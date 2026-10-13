from __future__ import annotations

import asyncio
import contextlib
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from livegraph.events import bus

router = APIRouter(tags=["ws"])


@router.websocket("/runs/{run_id}/ws")
async def run_ws(websocket: WebSocket, run_id: UUID, since: int = 0) -> None:
    await websocket.accept()

    async def pump() -> None:
        async for env in bus.subscribe(run_id, since):
            await websocket.send_text(env.model_dump_json())

    task = asyncio.create_task(pump())
    try:
        while True:
            recv = asyncio.create_task(websocket.receive_text())
            done, _ = await asyncio.wait({recv, task}, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                recv.cancel()
                break
            recv.result()  # client messages are ignored (keepalive pings etc.)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
