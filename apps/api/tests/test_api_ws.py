"""WebSocket replay (since=N) + live streaming, against a real uvicorn server in the test loop."""

from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest_asyncio
import uvicorn
import websockets

from tests.util import start_run, upload_docs, wait_run


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture
async def server(app):
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.01)
    yield f"127.0.0.1:{port}"
    srv.should_exit = True
    await task


async def _collect(url: str, until_done: bool = True, timeout: float = 15.0) -> list[dict]:
    out: list[dict] = []
    async with websockets.connect(url) as ws:
        async def reader():
            async for msg in ws:
                ev = json.loads(msg)
                out.append(ev)
                if until_done and ev["type"] == "run.status" and ev["payload"]["status"] == "done":
                    return
        await asyncio.wait_for(reader(), timeout)
    return out


async def test_ws_live_stream_and_replay_since(server, fake):
    fake["slow"]["editor"] = 0.01  # spread deltas over several 50ms batches
    async with httpx.AsyncClient(base_url=f"http://{server}") as client:
        await upload_docs(client)
        run_id = await start_run(client)
        live = await _collect(f"ws://{server}/api/runs/{run_id}/ws?since=0")
        snap = await wait_run(client, run_id)

    persisted = [e for e in live if e["seq"] is not None]
    assert [e["seq"] for e in persisted] == list(range(1, len(persisted) + 1))
    deltas = [e for e in live if e["type"] == "node.delta"]
    assert deltas and all(e["seq"] is None and e["node_id"] for e in deltas)
    editor_id = next(n["id"] for n in snap["nodes"] if n["key"] == "editor")
    streamed = "".join(e["payload"]["text"] for e in deltas if e["node_id"] == editor_id)
    assert streamed == snap["versions"][editor_id]["output_text"]
    assert len([e for e in deltas if e["node_id"] == editor_id]) > 1
    # deltas for a node arrive before its node.version
    idx_v = next(i for i, e in enumerate(live) if e["type"] == "node.version" and e["node_id"] == editor_id)
    assert max(i for i, e in enumerate(live) if e["type"] == "node.delta" and e["node_id"] == editor_id) < idx_v
    for k in ("seq", "run_id", "type", "node_id", "payload", "ts"):
        assert k in live[0]

    # replay from the middle: exactly the persisted tail, no ephemeral events
    n = len(persisted) // 2
    async with websockets.connect(f"ws://{server}/api/runs/{run_id}/ws?since={n}") as ws:
        got = []
        while len(got) < len(persisted) - n:
            got.append(json.loads(await asyncio.wait_for(ws.recv(), 5)))
    assert [e["seq"] for e in got] == list(range(n + 1, len(persisted) + 1))
    assert got == persisted[n:]
