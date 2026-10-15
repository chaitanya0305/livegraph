"""``NodeDeps``: what every agent run gets as Pydantic AI deps (one instance per node attempt).

Tools never share a DB session: each opens its own from ``session`` (``SessionLocal``). Graph mutations go
through ``sched`` (the run's ``RunScheduler``) under its mutex. ``guard()`` raises ``CancelledError`` as soon
as the attempt was superseded (steer/rerun/cancel/...), and tool events are only emitted while the attempt
still owns its node.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import UUID

from livegraph.agents.base import AgentContext
from livegraph.db import base as db_base
from livegraph.events import bus
from livegraph.rag.schemas import RetrievedChunk
from livegraph.trace import TraceBuilder


@dataclass
class NodeDeps:
    run_id: UUID
    node_id: UUID
    node_key: str
    node_type: str
    sched: Any  # RunScheduler (typed loosely: engine imports agents)
    attempt_id: str
    me: asyncio.Task[Any] | None
    ctx: AgentContext | None = None
    budgets: dict[str, int] = field(default_factory=dict)
    force: bool = False  # Rerun: bypass the web cache
    web_live: bool = True  # may hit the network on a web-cache miss (False for plain recomputes)
    agent: str = ""  # node type of the agent making calls ("prosecutor" inside the judge's ask_rebuttal)
    parent_call_id: str | None = None
    parent_ref: str | None = None  # context ref of the parent call (nested runs)
    # scope rule for the context tools: {node_key: current version id} of transitive active ancestors
    readable: dict[str, UUID] = field(default_factory=dict)
    recall_enabled: bool = False  # the agent has the context tools: its history may be compacted to stubs
    # shared with nested (agent-as-tool) runs
    retrieved: dict[str, RetrievedChunk] = field(default_factory=dict)
    via: dict[str, str] = field(default_factory=dict)
    cited_ok: set[str] = field(default_factory=set)
    tracer: TraceBuilder = field(default_factory=TraceBuilder)
    rebuttals: list[dict[str, Any]] = field(default_factory=list)
    # request_upstream_work calls accepted during this attempt ({target, instruction}); the engine turns them
    # into steers of the targets once this attempt's version has persisted
    upstream_requests: list[dict[str, str]] = field(default_factory=list)
    # per agent
    counts: dict[str, int] = field(default_factory=dict)
    cache: dict[str, str] = field(default_factory=dict)
    cache_refs: dict[str, str] = field(default_factory=dict)  # repeat-cache key -> the original call's ref
    call_refs: dict[str, str] = field(default_factory=dict)  # our call_id -> its ref (parent_ref of nested runs)
    # history compaction (agents/history.py): tool return (context.part_key) -> ref, and ref -> stub line
    refs: dict[str, str] = field(default_factory=dict)
    stubs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.agent = self.agent or self.node_type

    @property
    def session(self):
        return db_base.SessionLocal

    @property
    def trace(self) -> list[dict[str, Any]]:
        return self.tracer.dump()

    def still_mine(self) -> bool:
        return self.sched is None or self.me is None or self.sched._still_mine_task(self.node_id, self.me)

    def guard(self) -> None:
        if not self.still_mine():
            raise asyncio.CancelledError()

    def add_chunks(self, chunks: list[RetrievedChunk], via: str) -> None:
        for c in chunks:
            cid = str(c.chunk_id)
            prev = self.retrieved.get(cid)
            if prev is None or c.score > prev.score:
                self.retrieved[cid] = c
            self.via.setdefault(cid, via)

    def citable(self, cid: str) -> bool:
        return cid in self.retrieved or cid in self.cited_ok

    async def emit(self, type: str, payload: dict[str, Any], ts: str = "") -> bool:
        """Persist + fan out a tool event and fold it into this attempt's trace (only while still mine)."""
        if not self.still_mine():
            return False
        env = await bus.emit(self.run_id, type, self.node_id, payload)
        self.tracer.apply(type, env.payload, ts or env.ts.isoformat())
        return True

    def child(self, agent: str, parent_call_id: str, budgets: dict[str, int]) -> NodeDeps:
        """Deps for a nested agent run (agent-as-tool): shares evidence + trace, own budgets/counters/refs."""
        return replace(
            self, agent=agent, parent_call_id=parent_call_id, parent_ref=self.call_refs.get(parent_call_id),
            budgets=budgets, counts={}, cache={}, cache_refs={}, call_refs={}, refs={}, stubs={}, rebuttals=[],
            upstream_requests=[],
        )
