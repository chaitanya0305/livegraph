"""EventBus: persist events (per-run monotonically increasing seq) then fan out to subscribers.

Single-process MVP: seq allocation is serialized by a per-run asyncio.Lock and seeded from
``SELECT max(seq)`` the first time a run is seen by this process.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import func, select

from livegraph.db import base as db_base
from livegraph.db.models import Event
from livegraph.protocol import EventEnvelope

log = logging.getLogger(__name__)

DELTA_FLUSH_SECONDS = 0.05


def _jsonable(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    # round-trip through pydantic to make UUIDs/datetimes JSON-safe
    from pydantic_core import to_jsonable_python

    return to_jsonable_python(payload)


def envelope_from_row(ev: Event) -> EventEnvelope:
    return EventEnvelope(seq=ev.seq, run_id=ev.run_id, type=ev.type, node_id=ev.node_id, payload=ev.payload, ts=ev.ts)


class EventBus:
    def __init__(self) -> None:
        self._locks: dict[UUID, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_seq: dict[UUID, int] = {}
        self._subs: dict[UUID, set[asyncio.Queue[EventEnvelope]]] = defaultdict(set)
        # ephemeral delta batching: (run_id, node_id) -> buffered text
        self._delta_buf: dict[tuple[UUID, UUID], list[str]] = {}
        self._delta_tasks: dict[tuple[UUID, UUID], asyncio.Task] = {}

    def reset(self) -> None:
        """Forget cached state (tests / loop changes)."""
        for t in self._delta_tasks.values():
            t.cancel()
        self.__init__()

    # ------------------------------------------------------------ emit

    async def emit(
        self,
        run_id: UUID,
        type: str,
        node_id: UUID | None = None,
        payload: Any = None,
        persist: bool = True,
    ) -> EventEnvelope:
        data = _jsonable(payload)
        if not persist:
            env = EventEnvelope(seq=None, run_id=run_id, type=type, node_id=node_id, payload=data, ts=datetime.now(UTC))
            self._fanout(env)
            return env
        if node_id is not None:
            # keep ordering sane: buffered deltas for this node go out before its persisted events
            await self.flush_deltas(run_id, node_id)
        async with self._locks[run_id]:
            async with db_base.SessionLocal() as s:
                if run_id not in self._last_seq:
                    self._last_seq[run_id] = int(
                        await s.scalar(select(func.coalesce(func.max(Event.seq), 0)).where(Event.run_id == run_id))
                        or 0
                    )
                seq = self._last_seq[run_id] + 1
                row = Event(run_id=run_id, seq=seq, type=type, node_id=node_id, payload=data)
                s.add(row)
                try:
                    await s.commit()
                except Exception:
                    self._last_seq.pop(run_id, None)
                    raise
                await s.refresh(row)
                self._last_seq[run_id] = seq
                env = envelope_from_row(row)
            self._fanout(env)
        return env

    def _fanout(self, env: EventEnvelope) -> None:
        for q in list(self._subs.get(env.run_id, ())):
            try:
                q.put_nowait(env)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queues
                log.warning("subscriber queue full; dropping event")

    # ------------------------------------------------------------ deltas (ephemeral, batched)

    def delta(self, run_id: UUID, node_id: UUID, text: str) -> None:
        if not text:
            return
        key = (run_id, node_id)
        self._delta_buf.setdefault(key, []).append(text)
        if key not in self._delta_tasks:
            self._delta_tasks[key] = asyncio.create_task(self._delayed_flush(key))

    async def _delayed_flush(self, key: tuple[UUID, UUID]) -> None:
        try:
            await asyncio.sleep(DELTA_FLUSH_SECONDS)
        except asyncio.CancelledError:
            return
        self._delta_tasks.pop(key, None)
        self._flush_now(key)

    def _flush_now(self, key: tuple[UUID, UUID]) -> None:
        parts = self._delta_buf.pop(key, None)
        if parts:
            text = "".join(parts)
            env = EventEnvelope(
                seq=None, run_id=key[0], type="node.delta", node_id=key[1], payload={"text": text}, ts=datetime.now(UTC)
            )
            self._fanout(env)

    def delta_reset(self, run_id: UUID, node_id: UUID, text: str) -> None:
        """Replace the node's live buffer with ``text`` (streamed output superseded). Unsent deltas are dropped."""
        self.drop_deltas(run_id, node_id)
        env = EventEnvelope(
            seq=None, run_id=run_id, type="node.delta", node_id=node_id,
            payload={"text": text, "reset": True}, ts=datetime.now(UTC),
        )
        self._fanout(env)

    async def flush_deltas(self, run_id: UUID, node_id: UUID) -> None:
        key = (run_id, node_id)
        t = self._delta_tasks.pop(key, None)
        if t is not None:
            t.cancel()
        self._flush_now(key)

    def drop_deltas(self, run_id: UUID, node_id: UUID) -> None:
        key = (run_id, node_id)
        t = self._delta_tasks.pop(key, None)
        if t is not None:
            t.cancel()
        self._delta_buf.pop(key, None)

    # ------------------------------------------------------------ subscribe / replay

    async def load_events(self, run_id: UUID, since: int = 0) -> list[EventEnvelope]:
        async with db_base.SessionLocal() as s:
            rows = (
                await s.execute(select(Event).where(Event.run_id == run_id, Event.seq > since).order_by(Event.seq))
            ).scalars()
            return [envelope_from_row(r) for r in rows]

    async def subscribe(self, run_id: UUID, since: int = 0) -> AsyncIterator[EventEnvelope]:
        """Replay persisted events with seq > since, then live events, without gaps or duplicates.

        The queue is registered *before* loading the backlog; live events already covered by the
        backlog are dropped by seq.
        """
        q: asyncio.Queue[EventEnvelope] = asyncio.Queue()
        self._subs[run_id].add(q)
        try:
            last = since
            for env in await self.load_events(run_id, since):
                last = env.seq or last
                yield env
            while True:
                env = await q.get()
                if env.seq is not None:
                    if env.seq <= last:
                        continue
                    if env.seq > last + 1:
                        # should not happen (in-process ordering), but backfill defensively
                        for missed in await self.load_events(run_id, last):
                            if missed.seq is not None and missed.seq < env.seq:
                                last = missed.seq
                                yield missed
                    last = env.seq
                yield env
        finally:
            self._subs[run_id].discard(q)
            if not self._subs[run_id]:
                self._subs.pop(run_id, None)


bus = EventBus()
