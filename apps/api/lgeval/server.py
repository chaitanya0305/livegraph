"""Start/stop a LiveGraph API subprocess (uvicorn from apps/api) with env overrides on a free port."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import time
from pathlib import Path
from typing import IO

import httpx

API_DIR = Path(__file__).resolve().parents[1]  # apps/api


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ApiServer:
    """`async with ApiServer({"DISABLED_NODE_TYPES": "critic"}) as srv: ... srv.base_url`.

    The child inherits os.environ plus `env`; settings not overridden come from apps/api/.env as usual
    (real env vars win over .env in pydantic-settings). Runs share the dev Postgres and are isolated by
    run_id. The API runs migrations at startup, so start several servers one after another.
    """

    def __init__(
        self,
        env: dict[str, str] | None = None,
        *,
        port: int | None = None,
        log_path: Path | None = None,
        startup_timeout: float = 60.0,
        host: str = "127.0.0.1",
    ) -> None:
        self.env = dict(env or {})
        self.port = port or free_port()
        self.host = host
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.base_url = f"http://{host}:{self.port}"
        self._proc: asyncio.subprocess.Process | None = None
        self._log: IO[bytes] | None = None

    def command(self) -> list[str]:
        return ["uv", "run", "uvicorn", "livegraph.main:app", "--host", self.host, "--port", str(self.port)]

    async def start(self) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("ab")
        out = self._log if self._log is not None else asyncio.subprocess.DEVNULL
        self._proc = await asyncio.create_subprocess_exec(
            *self.command(),
            cwd=API_DIR,
            env={**os.environ, **self.env},
            stdout=out,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # own process group: `uv run` spawns uvicorn as a child
        )
        try:
            await self._wait_healthy()
        except BaseException:
            await self.stop()
            raise

    async def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        async with httpx.AsyncClient(timeout=2.0) as http:
            while time.monotonic() < deadline:
                assert self._proc is not None
                if self._proc.returncode is not None:
                    raise RuntimeError(f"API server exited with code {self._proc.returncode} (log: {self.log_path})")
                with contextlib.suppress(httpx.HTTPError):
                    r = await http.get(f"{self.base_url}/api/health")
                    if r.status_code == 200:
                        return
                await asyncio.sleep(0.5)
        raise TimeoutError(f"API server not healthy after {self.startup_timeout:.0f}s (log: {self.log_path})")

    def _signal(self, sig: int) -> None:
        assert self._proc is not None
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self._proc.pid, sig)

    async def stop(self, grace: float = 15.0) -> None:
        """SIGTERM the process group, then SIGKILL after `grace` seconds."""
        if self._proc is not None and self._proc.returncode is None:
            self._signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), grace)
            except TimeoutError:
                self._signal(signal.SIGKILL)
                await self._proc.wait()
        self._proc = None
        if self._log is not None:
            self._log.close()
            self._log = None

    async def __aenter__(self) -> ApiServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
