"""The LiveGraph engine: one ``RunScheduler`` (asyncio task) per active run.

Concurrency model (single process):
- ``RunScheduler.mutex`` serializes every *state mutation* of a run (scheduler ticks, node start /
  finalize, user actions). LLM calls and retrieval run *outside* the mutex, bounded by a per-run
  semaphore (``settings.max_concurrency``).
- A tick (triggered by ``wake()``) reloads nodes/edges, applies the skip rule, starts every ready node
  (status ``pending``/``stale``) as its own task, and recomputes the run status.
- Each node task: compute ``input_hash`` -> reuse an existing version with the same hash (no LLM call)
  or run the agent (a tool-using Pydantic AI agent loop for research/critic/debaters/judge/planner, a
  validated single shot for claims/editor; see ``agents/runner.py``) -> persist a new ``NodeVersion``
  (with the attempt's tool trace and every chunk the agent saw as retrievals) -> set current -> side
  effects (claims/judge/critic follow-ups) -> status ``done``. Changing a node's current version marks all its descendants ``stale``;
  they recompute in topological order through the normal ready rule, and recomputation stops early
  wherever the hash is unchanged (reuse).
- Early cutoff: input hashes name their inputs by content (``NodeVersion.output_hash``), and a new current
  version whose content equals the previous one does not stale anything; descendants that were already
  staled re-hash, find their current version again and go back to ``done`` ("upstream unchanged").
- Agent upstream requests (``request_upstream_work``) are queued on the attempt and applied as steers of the
  target (``SteerMessage.source = "agent:<key>"``) once the requester's version has persisted.
- Context store: every stored tool result of an attempt is a ``context_items`` row (``agents/context.py``);
  the rows are stamped with the version id once it persists (phase 4), and agents read their ancestors'
  outputs/traces/results through the context tools within ``deps.readable`` (built in ``_context``).
- Cancellation (steer/rerun/edit/restore/veto/cancel) cancels the node task; a cancelled task never
  writes (a task only finalizes if it is still the registered task for its node). Agent tools check the
  same ownership (``NodeDeps.guard``) before acting and before emitting their events, and every
  transition to ``running`` carries a fresh ``attempt_id`` so clients can drop stale tool events.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import re
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic_ai.usage import UsageLimits

from livegraph import rag
from livegraph.agents import SPECS, context
from livegraph.agents import critic as critic_mod
from livegraph.agents import research as research_mod
from livegraph.agents.base import AgentContext, AgentLimits, ParentOutput
from livegraph.agents.deps import NodeDeps
from livegraph.agents.guard import ToolOut
from livegraph.agents.human_input import prompt_for
from livegraph.agents.models import model_for
from livegraph.agents.runner import run_agent
from livegraph.agents.tools import context_tools
from livegraph.agents.tools.rag_tools import web_doc_ids
from livegraph.config import settings, tools_enabled
from livegraph.db import base as db_base
from livegraph.db.models import Claim, ClaimHistory, Edge, Node, NodeVersion, Run, SteerMessage
from livegraph.engine.hashing import input_hash, input_ref, output_hash
from livegraph.engine.serialize import claim_out, edge_out, node_out, version_out
from livegraph.events import bus
from livegraph.trace import truncate

log = logging.getLogger(__name__)

ACTIVE_RUN_STATUSES = ("planning", "running")
RUNNABLE = ("pending", "stale")
# node types whose output depends on the corpus (beyond their parents): their hash carries the corpus
# fingerprint and distrust/pin of a document they read makes them recompute
CORPUS_TYPES = ("research", "prosecutor", "defender", "judge")
SINGLE_SHOT_TYPES = ("claims", "editor")  # tool-less unless CONTEXT_TOOLS_SINGLE_SHOT=true
# node types that get their research ancestors' evidence and the claims ancestor's claims in their prompt
EVIDENCE_TYPES = ("claims", "prosecutor", "defender", "judge", "editor")
UPSTREAM_MAX_CHARS = 500  # request_upstream_work instruction cap
UPSTREAM_FORBIDDEN = ("plan", "human_input")  # request_upstream_work targets that cannot be steered by agents
IMPACT_ACTIONS = ("steer", "rerun", "config", "edit", "restore", "veto")


def _uses_corpus(node_type: str) -> bool:
    return node_type == "research" or (node_type in CORPUS_TYPES and tools_enabled(node_type))


def _cited_ids(output: Any) -> set[str]:
    out: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("chunk_ids", "support_chunk_ids") and isinstance(v, list):
                    out.update(str(i) for i in v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for i in x:
                walk(i)

    walk(output.model_dump(mode="json") if hasattr(output, "model_dump") else output)
    return out


def format_budgets(budgets: dict[str, int] | AgentLimits | None) -> str:
    if isinstance(budgets, AgentLimits):
        budgets = budgets.budgets
    if not budgets:
        return ""
    lines = "\n".join(f"- {t}: at most {n} calls" for t, n in budgets.items())
    return f"\n\n## Tool budgets\n{lines}\n"


def _uses_tools(node_type: str) -> bool:
    return tools_enabled(node_type) or (settings.context_tools_single_shot and node_type in SINGLE_SHOT_TYPES)


def _reads_context(node_type: str) -> bool:
    """The node's agent has the context tools, so it can read any transitive ancestor (``deps.readable``)."""
    spec = SPECS.get(node_type)
    return (
        spec is not None and _uses_tools(node_type) and settings.context_tools
        and context_tools.has_context_tools(spec.agents)
    )


class EngineError(Exception):
    """User-facing error (maps to 4xx)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _now() -> datetime:
    return datetime.now(UTC)


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", s.strip().lower()).strip("_")
    return s[:48] or "node"


class Graph:
    """In-memory view of a run's nodes and edges loaded in one session."""

    def __init__(self, nodes: list[Node], edges: list[Edge]) -> None:
        self.nodes = {n.id: n for n in nodes}
        self.by_key = {n.key: n for n in nodes}
        self.edges = edges
        self.parents: dict[UUID, list[UUID]] = defaultdict(list)
        self.children: dict[UUID, list[UUID]] = defaultdict(list)
        for e in edges:
            self.parents[e.target_node_id].append(e.source_node_id)
            self.children[e.source_node_id].append(e.target_node_id)

    @staticmethod
    def inactive(n: Node) -> bool:
        return n.removed or n.status == "skipped"

    def active_parents(self, nid: UUID) -> list[Node]:
        return [self.nodes[p] for p in self.parents.get(nid, []) if not self.inactive(self.nodes[p])]

    def descendants(self, nid: UUID) -> list[Node]:
        seen: set[UUID] = set()
        order: list[Node] = []
        q = deque(self.children.get(nid, []))
        while q:
            c = q.popleft()
            if c in seen:
                continue
            seen.add(c)
            order.append(self.nodes[c])
            q.extend(self.children.get(c, []))
        return order

    def ancestors(self, nid: UUID) -> list[Node]:
        seen: set[UUID] = set()
        out: list[Node] = []
        q = deque(self.parents.get(nid, []))
        while q:
            p = q.popleft()
            if p in seen:
                continue
            seen.add(p)
            out.append(self.nodes[p])
            q.extend(self.parents.get(p, []))
        return out

    def reachable(self, src: UUID, dst: UUID) -> bool:
        return src == dst or any(n.id == dst for n in self.descendants(src))

    def topo(self) -> list[Node]:
        """All nodes in topological order (Kahn); ties keep creation order."""
        rank = {nid: i for i, nid in enumerate(self.nodes)}
        indeg = {nid: 0 for nid in self.nodes}
        for e in self.edges:
            if e.target_node_id in indeg and e.source_node_id in indeg:
                indeg[e.target_node_id] += 1
        heap = [(rank[nid], nid) for nid, d in indeg.items() if d == 0]
        heapq.heapify(heap)
        out: list[Node] = []
        while heap:
            _, nid = heapq.heappop(heap)
            out.append(self.nodes[nid])
            for c in self.children.get(nid, []):
                if c in indeg:
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        heapq.heappush(heap, (rank[c], c))
        return out

    def impact(self, nid: UUID, action: str) -> tuple[list[UUID], list[UUID]]:
        """Upper bound of what an action on ``nid`` reruns: (node ids in topological order, skipped ids).

        steer/rerun/config rerun the node itself and (if its result changes) every active descendant;
        edit/restore set the node's version directly, so only descendants rerun; veto removes the node, so
        descendants left without active inputs are skipped and the others rerun. Early cutoff can make the
        real set smaller (a recompute with an unchanged result stops there)."""
        order = {n.id: i for i, n in enumerate(self.topo())}
        desc = sorted((d for d in self.descendants(nid) if not self.inactive(d)), key=lambda d: order[d.id])
        if action == "veto":
            gone = {nid}
            skipped: list[UUID] = []
            for d in desc:
                live = [p for p in self.parents.get(d.id, []) if p not in gone and not self.inactive(self.nodes[p])]
                if not live:
                    gone.add(d.id)
                    skipped.append(d.id)
            return [d.id for d in desc if d.id not in gone], skipped
        me = self.nodes[nid]
        own = [nid] if action in ("steer", "rerun", "config") and not self.inactive(me) else []
        return own + [d.id for d in desc], []


async def load_graph(s: AsyncSession, run_id: UUID) -> Graph:
    nodes = list((await s.execute(select(Node).where(Node.run_id == run_id).order_by(Node.created_at, Node.key))).scalars())
    edges = list((await s.execute(select(Edge).where(Edge.run_id == run_id))).scalars())
    return Graph(nodes, edges)


class RunScheduler:
    def __init__(self, engine: Engine, run_id: UUID) -> None:
        self.engine = engine
        self.run_id = run_id
        self.mutex = asyncio.Lock()
        self.sem = asyncio.Semaphore(max(1, settings.max_concurrency))
        self.tasks: dict[UUID, asyncio.Task] = {}
        self.force: set[UUID] = set()
        self.web_lock = asyncio.Lock()  # serializes web document ingest (check-then-insert) for this run
        self.web_live_once: set[UUID] = set()  # PATCHed nodes: their next run may hit the network once
        # ancestor evidence / claims keyed by the frozenset of the version ids they were built from
        self._evidence_cache: dict[frozenset, list[dict[str, Any]]] = {}
        self._claims_cache: dict[frozenset, list[dict[str, Any]]] = {}
        self._wake = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._stopped = False

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._loop(), name=f"run-{self.run_id}")
        self.wake()

    def wake(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        self._stopped = True
        for t in list(self.tasks.values()):
            t.cancel()
        self.tasks.clear()
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _loop(self) -> None:
        while not self._stopped:
            await self._wake.wait()
            self._wake.clear()
            try:
                async with self.mutex:
                    await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                log.exception("scheduler tick failed for run %s", self.run_id)

    def is_idle(self) -> bool:
        return not self.tasks and not self._wake.is_set()

    # ------------------------------------------------------------------ helpers (call with mutex held)

    async def set_node_status(self, s: AsyncSession, n: Node, status: str, reason: str | None = None) -> str | None:
        """Persist + emit a status change; entering ``running`` returns the new attempt id."""
        if n.status == status:
            return None
        n.status = status
        await s.commit()
        payload: dict[str, Any] = {"status": status}
        if reason:
            payload["reason"] = reason
        attempt_id = None
        if status == "running":
            attempt_id = uuid.uuid4().hex
            payload["attempt_id"] = attempt_id
        await bus.emit(self.run_id, "node.status", n.id, payload)
        return attempt_id

    async def set_run_status(self, s: AsyncSession, run: Run, status: str) -> None:
        if run.status == status:
            return
        run.status = status
        run.updated_at = _now()
        await s.commit()
        await bus.emit(self.run_id, "run.status", None, {"status": status})

    def cancel_task(self, nid: UUID) -> bool:
        t = self.tasks.pop(nid, None)
        if t is not None:
            t.cancel()
            bus.drop_deltas(self.run_id, nid)
            return True
        return False

    async def mark_descendants_stale(self, s: AsyncSession, g: Graph, nid: UUID, reason: str) -> None:
        hit = [d for d in g.descendants(nid) if not (d.removed or d.status in ("skipped", "pending", "stale"))]
        if not hit:
            return
        for d in hit:
            self.cancel_task(d.id)
            d.status = "stale"
        await s.commit()  # one commit for the whole batch
        for d in hit:
            await bus.emit(self.run_id, "node.status", d.id, {"status": "stale", "reason": reason})

    async def reopen_run(self, s: AsyncSession) -> None:
        run = await s.get(Run, self.run_id)
        if run is not None and run.status in ("done", "failed", "waiting_input"):
            await self.set_run_status(s, run, "running")

    async def queue_steer(
        self, s: AsyncSession, g: Graph, n: Node, text: str, *, source: str = "user", reason: str = "steer",
    ) -> SteerMessage:
        """Persist a steer, emit ``steer.queued``, and restart the node (a done/failed node's descendants go stale)."""
        m = SteerMessage(node_id=n.id, text=text, source=source)
        s.add(m)
        await s.commit()
        await s.refresh(m)
        await bus.emit(self.run_id, "steer.queued", n.id, {"steer_id": str(m.id), "text": text, "source": source})
        if n.status == "running":
            self.cancel_task(n.id)
            await self.set_node_status(s, n, "stale", reason)
        elif n.status in ("done", "failed"):
            await self.set_node_status(s, n, "stale", reason)
            await self.mark_descendants_stale(s, g, n.id, f"upstream {n.key} steered")
            await self.reopen_run(s)
        # pending/stale: the steer is picked up when it runs
        return m

    # ------------------------------------------------------------------ tick

    async def _tick(self) -> None:
        async with db_base.SessionLocal() as s:
            run = await s.get(Run, self.run_id)
            if run is None or run.status == "cancelled":
                return
            g = await load_graph(s, self.run_id)

            # skip rule: a node whose parents are all inactive (vetoed/skipped) is skipped
            changed = True
            while changed:
                changed = False
                for n in g.nodes.values():
                    if Graph.inactive(n) or not g.parents.get(n.id):
                        continue
                    if not g.active_parents(n.id):
                        self.cancel_task(n.id)
                        await self.set_node_status(s, n, "skipped", "inputs removed")
                        changed = True

            if run.status != "paused":
                plan_blocking = any(
                    n.type == "plan" and not n.removed and n.status in ("pending", "running", "stale")
                    for n in g.nodes.values()
                )
                for n in g.nodes.values():
                    if n.removed or n.status not in RUNNABLE or n.id in self.tasks:
                        continue
                    if plan_blocking and n.type != "plan":
                        continue
                    if self._ready(g, n):
                        self.tasks[n.id] = asyncio.create_task(self._run_node(n.id), name=f"node-{n.key}")

            await self._update_run_status(s, run, g)

    def _ready(self, g: Graph, n: Node) -> bool:
        for p in g.active_parents(n.id):
            if p.status != "done" or p.current_version_id is None or p.id in self.tasks:
                return False
        if n.type == "claims":
            # claims waits for every critic so follow-up research the critics add is included
            for c in g.nodes.values():
                if c.type == "critic" and not Graph.inactive(c) and (c.status not in ("done", "failed") or c.id in self.tasks):
                    return False
        return True

    async def _update_run_status(self, s: AsyncSession, run: Run, g: Graph) -> None:
        if run.status in ("paused", "cancelled"):
            return
        active = [n for n in g.nodes.values() if not Graph.inactive(n)]
        if self.tasks:
            plan_running = any(g.nodes[nid].type == "plan" for nid in self.tasks if nid in g.nodes)
            status = "planning" if plan_running else "running"
        elif any(n.status == "waiting_input" for n in active):
            status = "waiting_input"
        elif all(n.status == "done" for n in active):
            status = "done"
        else:
            status = "failed"
        await self.set_run_status(s, run, status)

    # ------------------------------------------------------------------ node execution

    async def _context(self, s: AsyncSession, g: Graph, n: Node, run: Run, steer_texts: list[str]) -> AgentContext:
        parents: list[ParentOutput] = []
        for p in g.active_parents(n.id):
            v = await s.get(NodeVersion, p.current_version_id) if p.current_version_id else None
            parents.append(
                ParentOutput(
                    key=p.key, type=p.type, title=p.title, config=p.config or {},
                    output=(v.output if v else {}), output_text=(v.output_text if v else ""),
                )
            )
        ctx = AgentContext(
            goal=run.goal, node_key=n.key, node_type=n.type, title=n.title, config=n.config or {},
            steer_texts=steer_texts, parents=parents,
            # scope rule of the context tools: only upstream work (covered by this node's input hash)
            readable={
                a.key: a.current_version_id for a in g.ancestors(n.id)
                if not Graph.inactive(a) and a.current_version_id is not None
            },
        )
        if n.type in ("claims", "prosecutor", "defender", "judge", "editor"):
            ctx.evidence = await self._ancestor_evidence(s, g, n)
            ctx.claims = await self._ancestor_claims(s, g, n, with_confidence=n.type == "editor")
        return ctx

    @staticmethod
    def _remember(cache: dict, key: frozenset, value: list) -> None:
        if len(cache) >= 64:
            cache.clear()
        cache[key] = value

    async def _ancestor_evidence(self, s: AsyncSession, g: Graph, n: Node) -> list[dict[str, Any]]:
        vids = [
            a.current_version_id for a in g.ancestors(n.id)
            if a.type == "research" and not Graph.inactive(a) and a.current_version_id
        ]
        key = frozenset(vids)
        if key in self._evidence_cache:  # versions are immutable; trust changes clear the cache
            return list(self._evidence_cache[key])
        rows = await rag.get_retrievals_many(s, vids)
        seen: dict[str, dict[str, Any]] = {}
        for vid in vids:
            for r in rows.get(vid, []):
                cid = str(r["chunk_id"])
                if cid not in seen:
                    seen[cid] = r
                elif r["score"] > seen[cid]["score"] or r.get("cited"):
                    seen[cid] = {**r, "cited": bool(r.get("cited") or seen[cid].get("cited"))}
        # chunks research actually cited come first, then the rest of what it saw
        out = sorted(seen.values(), key=lambda r: (not r.get("cited"), -r["score"]))
        self._remember(self._evidence_cache, key, out)
        return list(out)

    async def _ancestor_claims(self, s: AsyncSession, g: Graph, n: Node, *, with_confidence: bool) -> list[dict[str, Any]]:
        cnode = next((a for a in g.ancestors(n.id) if a.type == "claims" and not Graph.inactive(a)), None)
        if cnode is None or cnode.current_version_id is None:
            return []
        key = frozenset([cnode.current_version_id])
        if not with_confidence and key in self._claims_cache:
            return [dict(d) for d in self._claims_cache[key]]
        v = await s.get(NodeVersion, cnode.current_version_id)
        items = (v.output or {}).get("claims", []) if v else []
        # confidence/status live on the claim rows (set by the judge), so that variant is never cached
        rows = (
            {c.key: c for c in (await s.execute(select(Claim).where(Claim.run_id == self.run_id))).scalars()}
            if with_confidence else {}
        )
        out = []
        for it in items:
            d = {"key": it.get("key"), "text": it.get("text"), "support_chunk_ids": it.get("support_chunk_ids", [])}
            row = rows.get(d["key"])
            if with_confidence and row is not None:
                d["confidence"] = row.confidence
                d["status"] = row.status
            out.append(d)
        if not with_confidence:
            self._remember(self._claims_cache, key, [dict(d) for d in out])
        return out

    async def _steers(self, s: AsyncSession, nid: UUID) -> list[SteerMessage]:
        return list(
            (await s.execute(select(SteerMessage).where(SteerMessage.node_id == nid).order_by(SteerMessage.created_at))).scalars()
        )

    async def _hash(self, s: AsyncSession, g: Graph, n: Node, run: Run, steer_texts: list[str]) -> str:
        """Input hash. Inputs are named by content (``input_ref``): the direct parents, plus every ancestor the
        node can read beyond them: research evidence + claims for EVIDENCE_TYPES, and all readable ancestors
        (``deps.readable``) for agents with the context tools. So a change hidden behind an unchanged parent
        still recomputes whoever can see it, while everyone else is cut off."""
        docs = await rag.corpus_fingerprint(s, self.run_id) if _uses_corpus(n.type) else None
        parents = g.active_parents(n.id)
        parent_ids = {p.id for p in parents}
        reads_ctx = _reads_context(n.type)
        extra: list[Node] = []
        if reads_ctx or n.type in EVIDENCE_TYPES:
            extra = [
                a for a in g.ancestors(n.id)
                if a.id not in parent_ids and not Graph.inactive(a) and a.current_version_id is not None
                and (reads_ctx or a.type in ("research", "claims"))
            ]
        vids = [x.current_version_id for x in parents + extra if x.current_version_id is not None]
        hashes: dict[UUID, str | None] = {}
        if vids:
            hashes = dict(
                (await s.execute(select(NodeVersion.id, NodeVersion.output_hash).where(NodeVersion.id.in_(vids)))).all()
            )

        def ref(x: Node) -> str:
            return input_ref(x.key, x.current_version_id, hashes.get(x.current_version_id))

        config = n.config or {}
        return input_hash(
            node_type=n.type,
            config=config,
            goal=run.goal,
            parent_version_ids=[ref(p) for p in parents],
            steer_texts=steer_texts,
            documents=docs,
            title=n.title,
            question=(config.get("question") or n.title) if n.type == "research" else None,
            ancestors=[ref(a) for a in extra],
        )

    async def _find_version_by_hash(self, s: AsyncSession, nid: UUID, h: str) -> NodeVersion | None:
        return (
            await s.execute(
                select(NodeVersion)
                .where(NodeVersion.node_id == nid, NodeVersion.input_hash == h)
                .order_by(NodeVersion.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    def _still_mine(self, nid: UUID) -> bool:
        return self.tasks.get(nid) is asyncio.current_task()

    def _still_mine_task(self, nid: UUID, task: asyncio.Task | None) -> bool:
        """Like ``_still_mine`` for code running in another task (agent tools run in their own tasks)."""
        return task is not None and self.tasks.get(nid) is task

    async def _run_node(self, nid: UUID) -> None:
        try:
            await self._run_node_inner(nid)
        except asyncio.CancelledError:
            bus.drop_deltas(self.run_id, nid)
            raise
        except Exception as exc:  # node failure: mark failed, keep the run going
            log.exception("node %s failed", nid)
            try:
                async with self.mutex:
                    if self._still_mine(nid):
                        self.tasks.pop(nid, None)
                        async with db_base.SessionLocal() as s:
                            n = await s.get(Node, nid)
                            if n is not None:
                                await self.set_node_status(s, n, "failed", type(exc).__name__)
                        await bus.emit(
                            self.run_id, "error", nid, {"message": f"{type(exc).__name__}: {exc}", "node_id": str(nid)}
                        )
            except Exception:  # pragma: no cover
                log.exception("failed to record node failure")
        finally:
            if self.tasks.get(nid) is asyncio.current_task():
                self.tasks.pop(nid, None)
            self.wake()

    async def _run_node_inner(self, nid: UUID) -> None:
        spec = SPECS[(await self._node_type(nid))]
        # ---- phase 1 (mutex): decide, maybe reuse, mark running, build context
        async with self.mutex:
            if not self._still_mine(nid):
                return
            async with db_base.SessionLocal() as s:
                run = await s.get(Run, self.run_id)
                g = await load_graph(s, self.run_id)
                n = g.nodes[nid]
                if n.removed or n.status not in RUNNABLE:
                    self.tasks.pop(nid, None)
                    return
                steers = await self._steers(s, nid)
                steer_texts = [m.text for m in steers]
                unconsumed = [m for m in steers if m.consumed_at is None]
                force = nid in self.force
                live_once = nid in self.web_live_once  # PATCHed: one live web pass for the new question
                self.web_live_once.discard(nid)
                has_versions = bool(await s.scalar(select(func.count()).where(NodeVersion.node_id == nid)))
                reason = (
                    "retry" if force else "steer" if unconsumed else "recompute" if has_versions else "initial"
                )

                if n.type == "human_input":
                    h = await self._hash(s, g, n, run, steer_texts)
                    existing = None if force else await self._find_version_by_hash(s, nid, h)
                    self.force.discard(nid)
                    self.tasks.pop(nid, None)
                    if existing is not None:
                        await self.apply_version(s, g, n, existing, reused=True)
                        await self.set_node_status(s, n, "done")
                        return
                    await self.set_node_status(s, n, "waiting_input")
                    await bus.emit(self.run_id, "input.requested", nid, {"prompt": prompt_for(n.config, n.title)})
                    return

                h = await self._hash(s, g, n, run, steer_texts)
                if not force:
                    existing = await self._find_version_by_hash(s, nid, h)
                    if existing is not None:
                        unchanged = existing.id == n.current_version_id
                        await self.apply_version(s, g, n, existing, reused=True)
                        await self._consume_steers(s, unconsumed)
                        await self.set_node_status(s, n, "done", "upstream unchanged" if unchanged else None)
                        self.tasks.pop(nid, None)
                        return
                attempt_id = await self.set_node_status(s, n, "running", reason) or uuid.uuid4().hex
                ctx = await self._context(s, g, n, run, steer_texts)
                node_type, node_key = n.type, n.key
                question = (n.config or {}).get("question") or n.title

        uses_tools = _uses_tools(node_type)
        limits: AgentLimits | None = getattr(spec, "LIMITS", None)
        reads_context = uses_tools and settings.context_tools and context_tools.has_context_tools(spec.agents)
        budgets = {**(context_tools.BUDGETS if reads_context else {}), **(limits.budgets if limits else {})}
        deps = NodeDeps(
            run_id=self.run_id, node_id=nid, node_key=node_key, node_type=node_type, sched=self,
            attempt_id=attempt_id, me=asyncio.current_task(), ctx=ctx, readable=dict(ctx.readable),
            recall_enabled=reads_context,
            budgets=budgets, force=force,
            # the network is only used on a first run, a steer or a Rerun; recomputes read the web cache
            web_live=force or not has_versions or bool(unconsumed) or live_once,
        )

        # ---- phase 2 (no mutex): starting evidence for research nodes
        if node_type == "research":
            chunks = await self._research_evidence(deps, question, uses_tools, refresh_web=force or not has_versions or live_once)
            if not self._still_mine(nid):
                return
            deps.add_chunks(chunks, "seed")
            ctx.evidence = [c.model_dump(mode="json") for c in chunks]
        deps.cited_ok.update(str(c["chunk_id"]) for c in ctx.evidence)

        # ---- phase 3 (semaphore): the agent run, final output streamed
        def on_delta(text: str) -> None:
            if self._still_mine(nid):
                bus.delta(self.run_id, nid, text)

        def on_reset(text: str) -> None:
            if self._still_mine(nid):
                bus.delta_reset(self.run_id, nid, text)

        model = model_for(node_type, spec.TIER)
        prompt = spec.build_prompt(ctx)
        if uses_tools and spec.agents.tools:
            prompt += (context_tools.TOOL_SYSTEM if reads_context else "") + format_budgets(budgets)
        async with self.sem:
            result = await run_agent(
                spec.agents.for_model(model, with_tools=uses_tools),
                prompt,
                model=model,
                render=spec.render,
                on_delta=on_delta,
                on_reset=on_reset,
                deps=deps,
                limits=UsageLimits(request_limit=limits.request_limit) if limits else None,
                timeout=limits.scaled_timeout if limits else None,
            )
        output = result.output
        if hasattr(spec, "finalize"):
            output = spec.finalize(output, deps)
        valid = set(deps.retrieved) | deps.cited_ok
        if node_type == "research":  # last resort; the output validator already asked the model to fix it
            output = research_mod.sanitize(output, valid)
        elif node_type == "claims":
            output = SPECS["claims"].sanitize(output, valid)
        output_text = spec.render(output)
        if output_text != result.output_text:  # e.g. the judge's rebuttals appended after the verdicts
            if output_text.startswith(result.output_text):
                on_delta(output_text[len(result.output_text) :])
            else:
                on_reset(output_text)

        # ---- phase 4 (mutex): persist the version, side effects
        async with self.mutex:
            if not self._still_mine(nid):
                return
            async with db_base.SessionLocal() as s:
                g = await load_graph(s, self.run_id)
                n = g.nodes[nid]
                if node_type == "plan":
                    await self.ensure_default_shape(s, g, n, output)
                    g = await load_graph(s, self.run_id)
                    n = g.nodes[nid]
                out_json = output.model_dump(mode="json")
                cited = _cited_ids(output) & set(deps.retrieved)
                v = await self._new_version(
                    s, n, h, out_json, output_text, reason,
                    steer_text="\n".join(m.text for m in unconsumed) or None,
                    model=result.model_name, tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                    trace=deps.trace,
                    output_hash=output_hash(out_json, output_text, [(c, c in cited) for c in deps.retrieved]),
                )
                # the attempt's stored tool results now belong to this version (readable downstream)
                await context.backfill(s, self.run_id, attempt_id, v.id)
                if deps.retrieved:
                    ordered = sorted(deps.retrieved.values(), key=lambda c: (str(c.chunk_id) not in cited, -c.score))
                    await rag.record_retrieval(s, node_version_id=v.id, chunks=ordered, cited=cited, via=deps.via)
                    await s.commit()
                await self.apply_version(s, g, n, v, reused=False)
                if deps.retrieved:
                    await bus.emit(
                        self.run_id, "retrieval.done", nid,
                        {"node_version_id": str(v.id), "chunks": await rag.get_retrievals(s, v.id)},
                    )
                await self._consume_steers(s, unconsumed)
                self.force.discard(nid)
                self.tasks.pop(nid, None)
                await self.set_node_status(s, n, "done")
                if node_type == "critic" and not uses_tools:  # single-shot fallback: follow-up from the output
                    out = critic_mod.CriticOutput.model_validate(output.model_dump())
                    if out.verdict == "thin" and (out.follow_up_question or "").strip():
                        await self.critic_follow_up(s, n, out.follow_up_question)
                        await self._attach_follow_ups(s, n)
                if deps.upstream_requests:
                    await self._apply_upstream_requests(s, n, deps.upstream_requests)

    async def _research_evidence(self, deps: NodeDeps, question: str, uses_tools: bool, *, refresh_web: bool) -> list:
        if uses_tools:  # the agent gathers the rest itself
            async with db_base.SessionLocal() as s:
                return await rag.search(s, run_id=self.run_id, query=question, k=settings.research_seed_k)
        # single-shot fallback (research not in AGENT_TOOLS): engine-side web + search, as before agents
        if refresh_web:
            try:
                await web_doc_ids(deps, question)
            except Exception as exc:  # web search is best-effort
                log.warning("web_research failed: %s", exc)
        async with db_base.SessionLocal() as s:
            return await rag.search(s, run_id=self.run_id, query=question, k=8)

    async def _node_type(self, nid: UUID) -> str:
        async with db_base.SessionLocal() as s:
            n = await s.get(Node, nid)
            return n.type if n else "plan"

    async def _consume_steers(self, s: AsyncSession, msgs: list[SteerMessage]) -> None:
        for m in msgs:
            m.consumed_at = _now()
            await s.merge(m)
        if msgs:
            await s.commit()
        for m in msgs:
            await bus.emit(
                self.run_id, "steer.consumed", m.node_id,
                {"steer_id": str(m.id), "text": m.text, "source": m.source or "user"},
            )

    async def _new_version(self, s: AsyncSession, n: Node, h: str, output: dict, output_text: str, reason: str, **kw: Any) -> NodeVersion:
        kw.setdefault("output_hash", output_hash(output, output_text))  # edits/answers: no retrievals
        num = int(await s.scalar(select(func.coalesce(func.max(NodeVersion.version), 0)).where(NodeVersion.node_id == n.id)) or 0)
        v = NodeVersion(
            node_id=n.id, version=num + 1, input_hash=h, output=output, output_text=output_text, reason=reason, **kw
        )
        s.add(v)
        await s.commit()
        await s.refresh(v)
        return v

    async def apply_version(self, s: AsyncSession, g: Graph, n: Node, v: NodeVersion, *, reused: bool, restore: bool = False) -> None:
        """Make ``v`` the node's current version, emit, run side effects, stale descendants on change."""
        prev = n.current_version_id
        prev_hash = None
        if prev is not None and prev != v.id:
            pv = await s.get(NodeVersion, prev)
            prev_hash = pv.output_hash if pv else None
        n.current_version_id = v.id
        await s.commit()
        if restore:
            await bus.emit(self.run_id, "node.restored", n.id, {"version_id": str(v.id)})
        else:
            vo = await version_out(s, v)
            await bus.emit(self.run_id, "node.version", n.id, {"version": vo.model_dump(mode="json"), "reused": reused})
        if n.type == "claims":
            await self.apply_claims(s, n, v)
        elif n.type == "judge":
            await self.apply_judge(s, v, always=not reused and not restore)
        elif n.type == "critic":
            await self._attach_follow_ups(s, n)
        # early cutoff: a new current version with the same content changes nothing downstream
        same_content = prev_hash is not None and prev_hash == v.output_hash
        if prev is not None and prev != v.id and not same_content:
            await self.mark_descendants_stale(s, g, n.id, f"upstream {n.key} changed")

    # ------------------------------------------------------------------ side effects

    async def apply_claims(self, s: AsyncSession, n: Node, v: NodeVersion) -> None:
        items = (v.output or {}).get("claims") or []
        rows = {c.key: c for c in (await s.execute(select(Claim).where(Claim.run_id == self.run_id))).scalars()}
        changed: list[Claim] = []
        seen = set()
        for it in items:
            key, text = str(it.get("key") or "").strip(), str(it.get("text") or "").strip()
            if not key or not text or key in seen:
                continue
            seen.add(key)
            row = rows.get(key)
            if row is None:
                row = Claim(run_id=self.run_id, key=key, text=text, status="active", source_node_id=n.id)
                s.add(row)
                changed.append(row)
            elif row.text != text or row.status != "active" or row.source_node_id != n.id:
                row.text, row.status, row.source_node_id = text, "active", n.id
                changed.append(row)
        for key, row in rows.items():
            if key not in seen and row.status != "struck":
                row.status = "struck"
                changed.append(row)
        await s.commit()
        for row in changed:
            await s.refresh(row)
            co = await claim_out(s, row)
            await bus.emit(self.run_id, "claim.upserted", None, {"claim": co.model_dump(mode="json")})

    async def apply_judge(self, s: AsyncSession, v: NodeVersion, *, always: bool) -> None:
        verdicts = (v.output or {}).get("verdicts") or []
        rows = {c.key: c for c in (await s.execute(select(Claim).where(Claim.run_id == self.run_id))).scalars()}
        for vd in verdicts:
            row = rows.get(str(vd.get("claim_key", "")).strip())
            if row is None:
                continue
            conf = vd.get("confidence")
            conf = None if conf is None else max(0.0, min(1.0, float(conf)))
            new_status = "struck" if vd.get("strike") else "active"
            if not always and row.confidence == conf and row.status == new_status:
                continue
            prev = row.confidence
            status_changed = row.status != new_status
            row.confidence = conf
            row.status = new_status
            rationale = str(vd.get("rationale") or "")
            s.add(ClaimHistory(claim_id=row.id, confidence=conf, node_version_id=v.id, rationale=rationale))
            await s.commit()
            await bus.emit(
                self.run_id, "claim.confidence", None,
                {"claim_id": str(row.id), "key": row.key, "confidence": conf, "previous": prev, "rationale": rationale},
            )
            if status_changed:
                await s.refresh(row)
                co = await claim_out(s, row)
                await bus.emit(self.run_id, "claim.upserted", None, {"claim": co.model_dump(mode="json")})

    async def critic_follow_up(self, s: AsyncSession, critic: Node, question: str) -> ToolOut:
        """Add one follow-up research node (``{research_key}_fu``) and the edge critic -> follow-up.

        Never chains (no follow-ups of follow-ups) and at most one per research node. The follow-up's edges
        into claims are added by ``_attach_follow_ups`` once the critic's version persists, so a running or
        finished claims node is never disturbed by a critic that is still thinking."""
        q = (question or "").strip()
        if not q:
            return ToolOut(text="error: empty question", summary="empty question", ok=False, error="empty question")
        g = await load_graph(s, self.run_id)
        sources = [p for p in g.active_parents(critic.id) if p.type == "research"]
        if not sources:
            return ToolOut(text="error: no research to follow up", summary="no research input", ok=False,
                           error="no research input")
        if any((p.config.get("params") or {}).get("follow_up") for p in sources):
            return ToolOut(text="follow-ups of follow-up research are not allowed; return your verdict",
                           summary="no chaining", ok=False, error="no chaining")
        src = sources[0]
        key = f"{src.key}_fu"
        if key in g.by_key:
            return ToolOut(text=f"follow-up research {key} already exists; return your verdict",
                           summary=f"{key} already exists", node_ids=[str(g.by_key[key].id)])
        new = await self.add_node(
            s, key=key, type="research", title=f"Follow-up: {src.title}"[:120],
            question=q, created_by="critic", params={"follow_up": True, "critic": critic.key},
        )
        await self.add_edge(s, critic.key, new.key)
        return ToolOut(text=f"added follow-up research node {new.key}; it runs before the claims. Return your verdict.",
                       summary=f"+ research {new.key}", node_ids=[str(new.id)])

    async def _attach_follow_ups(self, s: AsyncSession, critic: Node) -> None:
        g = await load_graph(s, self.run_id)
        fus = [
            n for n in g.nodes.values()
            if n.type == "research" and not n.removed and (n.config.get("params") or {}).get("critic") == critic.key
        ]
        for fu in fus:
            for c in g.nodes.values():
                if c.type == "claims" and not c.removed and c.id not in g.children.get(fu.id, []):
                    await self.add_edge(s, fu.key, c.key)

    async def _upstream_counts(self, s: AsyncSession, requester_key: str) -> tuple[set[UUID], int]:
        """Persisted agent requests of this run: (target node ids ``requester_key`` already asked, run total)."""
        rows = (
            await s.execute(
                select(SteerMessage.node_id, SteerMessage.source)
                .join(Node, Node.id == SteerMessage.node_id)
                .where(Node.run_id == self.run_id, SteerMessage.source.like("agent:%"))
            )
        ).all()
        mine = {nid for nid, src in rows if src == f"agent:{requester_key}"}
        return mine, len(rows)

    async def _apply_upstream_requests(self, s: AsyncSession, requester: Node, reqs: list[dict[str, str]]) -> None:
        """After the requester's version persisted: each accepted request becomes a steer of its target.

        Re-checked here under the mutex (concurrent requesters, graph changes since the call): the target must
        still be an active strict ancestor, at most one per (requester, target) and MAX_UPSTREAM_REQUESTS per run.
        The target recomputes, its descendants (the requester included) go stale and re-hash; with early cutoff
        they reuse unless the target's result changed. A rerun requester cannot repeat a request, so the graph
        settles."""
        for r in reqs:
            g = await load_graph(s, self.run_id)
            t = g.by_key.get(r["target"])
            me = g.nodes.get(requester.id)
            if t is None or me is None or me.removed or Graph.inactive(t) or t.type in UPSTREAM_FORBIDDEN:
                continue
            if t.id not in {a.id for a in g.ancestors(me.id)}:
                continue
            mine, total = await self._upstream_counts(s, me.key)
            if t.id in mine or total >= settings.max_upstream_requests:
                continue
            await self.queue_steer(
                s, g, t, f"Request from {me.key}: {r['instruction']}", source=f"agent:{me.key}",
                reason=f"request from {me.key}",
            )
        self.wake()

    # ------------------------------------------------------------------ agent tool entry points

    async def tool_request_upstream(self, deps: NodeDeps, target_node_key: str, instruction: str) -> ToolOut:
        def no(msg: str, summary: str | None = None) -> ToolOut:
            return ToolOut(text=f"rejected: {msg}", summary=summary or truncate(msg), ok=False, error=truncate(msg))

        deps.guard()
        if deps.parent_call_id is not None:
            return no("request_upstream_work is not available inside a rebuttal", "not in a rebuttal")
        text = " ".join((instruction or "").split())
        if not text:
            return no("empty instruction", "empty instruction")
        if len(text) > UPSTREAM_MAX_CHARS:
            return no(f"instruction is {len(text)} characters; at most {UPSTREAM_MAX_CHARS}", "instruction too long")
        key = (target_node_key or "").strip()
        if key == deps.node_key:
            return no("that is your own node", "own node")
        async with db_base.SessionLocal() as s:
            g = await load_graph(s, self.run_id)
            t = g.by_key.get(key)
            if t is None:
                return no(f"unknown node {key}; call list_agent_runs()", "unknown node")
            if key not in deps.readable:
                return no(f"{key} is not upstream of you", "not upstream")
            if t.removed or t.type in UPSTREAM_FORBIDDEN:
                return no(f"{key} ({t.type}) cannot take requests", "not requestable")
            mine, total = await self._upstream_counts(s, deps.node_key)
        queued = {q["target"] for q in deps.upstream_requests}
        if t.id in mine or key in queued:
            return no(f"you already asked {key} once in this run; use its current output", "already requested")
        if total + len(deps.upstream_requests) >= settings.max_upstream_requests:
            return no(
                f"the run's limit of {settings.max_upstream_requests} upstream requests is used up; finish with "
                "your inputs", "request limit reached",
            )
        deps.upstream_requests.append({"target": key, "instruction": text})
        return ToolOut(
            text=f"queued: {key} will redo its work with your request after you return. Your current inputs are "
            "final for this answer; finish it now.",
            summary=truncate(f"→ {key}: {text}"), node_ids=[str(t.id)],
        )

    async def tool_add_node(
        self, deps: NodeDeps, *, key: str, type: str, title: str, question: str | None = None,
        instructions: str | None = None,
    ) -> ToolOut:
        if type == "plan":
            return ToolOut(text="error: the plan node already exists", summary="rejected", ok=False,
                           error="the plan node already exists")
        async with self.mutex:
            deps.guard()
            async with db_base.SessionLocal() as s:
                g = await load_graph(s, self.run_id)
                if _slug(key) in g.by_key:
                    n = g.by_key[_slug(key)]
                    return ToolOut(text=f"node {n.key} already exists", summary=f"{n.key} exists", node_ids=[str(n.id)])
                try:
                    n = await self.add_node(s, key=key, type=type, title=title, question=question, instructions=instructions)
                except EngineError as e:
                    return ToolOut(text=f"error: {e}", summary=str(e), ok=False, error=str(e))
        return ToolOut(text=f"added node {n.key}", summary=f"+ {n.type} {n.key}", node_ids=[str(n.id)])

    async def tool_add_edge(self, deps: NodeDeps, source_key: str, target_key: str) -> ToolOut:
        async with self.mutex:
            deps.guard()
            async with db_base.SessionLocal() as s:
                try:
                    await self.add_edge(s, source_key, target_key)
                except EngineError as e:
                    return ToolOut(text=f"error: {e}", summary=str(e), ok=False, error=str(e))
                g = await load_graph(s, self.run_id)
                ids = [str(g.by_key[_slug(k)].id) for k in (source_key, target_key) if _slug(k) in g.by_key]
        return ToolOut(text=f"added edge {source_key}->{target_key}", summary=f"{source_key} → {target_key}", node_ids=ids)

    async def tool_follow_up(self, deps: NodeDeps, question: str) -> ToolOut:
        async with self.mutex:
            deps.guard()
            async with db_base.SessionLocal() as s:
                critic = await s.get(Node, deps.node_id)
                return await self.critic_follow_up(s, critic, question)

    # ------------------------------------------------------------------ graph building

    async def add_node(
        self, s: AsyncSession, *, key: str, type: str, title: str, question: str | None = None,
        instructions: str | None = None, created_by: str = "planner", params: dict | None = None,
    ) -> Node:
        if type not in SPECS:
            raise EngineError(f"unknown node type {type!r}; expected one of {sorted(SPECS)}")
        config: dict[str, Any] = {"instructions": instructions, "params": params or {}}
        if question:
            config["question"] = question
        n = Node(
            run_id=self.run_id, key=_slug(key), type=type, title=title or key, status="pending",
            config=config, position=None, created_by=created_by, removed=False,
        )
        s.add(n)
        await s.commit()
        await s.refresh(n)
        await bus.emit(self.run_id, "node.added", n.id, {"node": node_out(n).model_dump(mode="json")})
        return n

    async def add_edge(self, s: AsyncSession, source_key: str, target_key: str) -> Edge | None:
        g = await load_graph(s, self.run_id)
        src, tgt = g.by_key.get(_slug(source_key)), g.by_key.get(_slug(target_key))
        if src is None or tgt is None:
            raise EngineError(f"unknown node key: {source_key if src is None else target_key}")
        if src.id == tgt.id or g.reachable(tgt.id, src.id):
            raise EngineError(f"edge {source_key}->{target_key} would create a cycle")
        if tgt.id in g.children.get(src.id, []):
            return None
        e = Edge(run_id=self.run_id, source_node_id=src.id, target_node_id=tgt.id)
        s.add(e)
        await s.commit()
        await s.refresh(e)
        await bus.emit(self.run_id, "edge.added", None, {"edge": edge_out(e).model_dump(mode="json")})
        # a new input invalidates a finished/running target
        if tgt.status in ("done", "running", "failed", "waiting_input") and not tgt.removed:
            self.cancel_task(tgt.id)
            await self.set_node_status(s, tgt, "stale", f"new input {src.key}")
            g = await load_graph(s, self.run_id)
            await self.mark_descendants_stale(s, g, tgt.id, f"upstream {tgt.key} changed")
        return e

    async def ensure_default_shape(self, s: AsyncSession, g: Graph, plan: Node, out: Any) -> None:
        """Make the Living Brief graph well-formed whatever the planner did (robustness for real models)."""

        def unique(key: str) -> str:
            k, i = key, 2
            while k in g.by_key:
                k, i = f"{key}{i}", i + 1
            return k

        async def refresh() -> Graph:
            return await load_graph(s, self.run_id)

        def of_type(t: str) -> list[Node]:
            return [n for n in g.nodes.values() if n.type == t and not n.removed]

        research = of_type("research")
        if not research:
            qs = list(getattr(out, "subquestions", []) or []) or [g.nodes[plan.id].title]
            run = await s.get(Run, self.run_id)
            if qs == [plan.title]:
                qs = [run.goal]
            for i, q in enumerate(qs[:5], 1):
                await self.add_node(s, key=unique(f"r{i}"), type="research", title=q[:80], question=q)
                g = await refresh()
            research = of_type("research")
        for r in research:
            if not g.parents.get(r.id):
                await self.add_edge(s, plan.key, r.key)
        g = await refresh()
        for i, r in enumerate(of_type("research"), 1):
            if (r.config.get("params") or {}).get("follow_up"):
                continue
            if not any(g.nodes[c].type == "critic" for c in g.children.get(r.id, [])):
                c = await self.add_node(s, key=unique(f"c{i}"), type="critic", title=f"Critique {r.key}")
                await self.add_edge(s, r.key, c.key)
                g = await refresh()
        claims = of_type("claims")
        if not claims:
            c = await self.add_node(s, key=unique("claims"), type="claims", title="Claims")
            g = await refresh()
            claims = [c]
        for r in of_type("research"):
            for c in claims:
                if c.id not in g.children.get(r.id, []):
                    await self.add_edge(s, r.key, c.key)
        g = await refresh()
        claim_key = claims[0].key
        for t, title in (("prosecutor", "Prosecutor"), ("defender", "Defender")):
            if not of_type(t):
                await self.add_node(s, key=unique(t), type=t, title=title)
                await self.add_edge(s, claim_key, of_type_key(await refresh(), t))
                g = await refresh()
        if not of_type("judge"):
            j = await self.add_node(s, key=unique("judge"), type="judge", title="Judge")
            await self.add_edge(s, claim_key, j.key)
            g = await refresh()
            for t in ("prosecutor", "defender"):
                for n in of_type(t):
                    await self.add_edge(s, n.key, j.key)
            g = await refresh()
        if not of_type("editor"):
            e = await self.add_node(s, key=unique("editor"), type="editor", title="Editor")
            for j in of_type("judge"):
                await self.add_edge(s, j.key, e.key)
        # any other orphan (no parents, not plan) hangs off the plan
        g = await refresh()
        for n in g.nodes.values():
            if n.type != "plan" and not n.removed and not g.parents.get(n.id):
                await self.add_edge(s, plan.key, n.key)


def of_type_key(g: Graph, t: str) -> str:
    return next(n.key for n in g.nodes.values() if n.type == t and not n.removed)


class Engine:
    def __init__(self) -> None:
        self.schedulers: dict[UUID, RunScheduler] = {}

    def scheduler(self, run_id: UUID) -> RunScheduler:
        sch = self.schedulers.get(run_id)
        if sch is None:
            sch = RunScheduler(self, run_id)
            self.schedulers[run_id] = sch
            sch.start()
        return sch

    async def shutdown(self) -> None:
        for sch in list(self.schedulers.values()):
            await sch.stop()
        self.schedulers.clear()

    # ------------------------------------------------------------------ startup recovery

    async def recover(self) -> None:
        """Runs that were mid-flight when the process died are marked ``paused`` (resume restarts them).

        Their ``running`` nodes go back to ``pending`` (``stale`` if they already had a version)."""
        async with db_base.SessionLocal() as s:
            runs = list((await s.execute(select(Run).where(Run.status.in_(ACTIVE_RUN_STATUSES)))).scalars())
            for run in runs:
                nodes = (await s.execute(select(Node).where(Node.run_id == run.id, Node.status == "running"))).scalars()
                for n in nodes:
                    n.status = "stale" if n.current_version_id else "pending"
                    await s.commit()
                    await bus.emit(run.id, "node.status", n.id, {"status": n.status, "reason": "server restart"})
                run.status = "paused"
                await s.commit()
                await bus.emit(run.id, "run.status", None, {"status": "paused"})

    # ------------------------------------------------------------------ runs

    async def create_run(self, goal: str, workflow: str = "living_brief", document_ids: list[UUID] | None = None) -> UUID:
        if workflow != "living_brief":
            raise EngineError(f"unknown workflow {workflow!r}", 422)
        async with db_base.SessionLocal() as s:
            titles = []
            for did in document_ids or []:
                d = await rag.get_document(s, did)
                if d is not None:
                    titles.append(d.title)
            run = Run(goal=goal, workflow=workflow, status="planning")
            s.add(run)
            await s.commit()
            await s.refresh(run)
            run_id = run.id
            await bus.emit(run_id, "run.status", None, {"status": "planning"})
            sch = RunScheduler(self, run_id)
            self.schedulers[run_id] = sch
            await sch.add_node(
                s, key="plan", type="plan", title="Plan", created_by="planner",
                params={"document_ids": [str(d) for d in document_ids or []], "document_titles": titles},
            )
        sch.start()
        return run_id

    async def _run_or_404(self, s: AsyncSession, run_id: UUID) -> Run:
        run = await s.get(Run, run_id)
        if run is None:
            raise EngineError("run not found", 404)
        return run

    async def pause(self, run_id: UUID) -> None:
        sch = self.scheduler(run_id)
        async with sch.mutex:
            async with db_base.SessionLocal() as s:
                run = await self._run_or_404(s, run_id)
                if run.status in ("done", "cancelled", "failed"):
                    return
                await sch.set_run_status(s, run, "paused")

    async def resume(self, run_id: UUID) -> None:
        sch = self.scheduler(run_id)
        async with sch.mutex:
            async with db_base.SessionLocal() as s:
                run = await self._run_or_404(s, run_id)
                if run.status in ("paused", "cancelled"):
                    await sch.set_run_status(s, run, "running")
        sch.wake()

    async def cancel(self, run_id: UUID) -> None:
        sch = self.scheduler(run_id)
        async with sch.mutex:
            async with db_base.SessionLocal() as s:
                run = await self._run_or_404(s, run_id)
                for nid in list(sch.tasks):
                    sch.cancel_task(nid)
                    n = await s.get(Node, nid)
                    if n is not None and n.status == "running":
                        await sch.set_node_status(s, n, "stale" if n.current_version_id else "pending", "cancelled")
                await sch.set_run_status(s, run, "cancelled")

    # ------------------------------------------------------------------ node actions

    async def _node_ctx(self, node_id: UUID) -> tuple[RunScheduler, UUID]:
        async with db_base.SessionLocal() as s:
            n = await s.get(Node, node_id)
            if n is None:
                raise EngineError("node not found", 404)
            return self.scheduler(n.run_id), n.run_id

    async def _reopen_run(self, sch: RunScheduler, s: AsyncSession) -> None:
        await sch.reopen_run(s)

    async def steer(self, node_id: UUID, text: str) -> None:
        sch, run_id = await self._node_ctx(node_id)
        async with sch.mutex:
            async with db_base.SessionLocal() as s:
                g = await load_graph(s, run_id)
                n = g.nodes[node_id]
                if n.removed:
                    raise EngineError("node was removed", 409)
                if n.type == "human_input":
                    raise EngineError("human_input nodes cannot be steered; answer them instead", 409)
                await sch.queue_steer(s, g, n, text)
        sch.wake()

    async def impact(self, node_id: UUID, action: str) -> dict[str, Any]:
        """Read-only preview of what an action would rerun (upper bound; early cutoff may stop sooner)."""
        if action not in IMPACT_ACTIONS:
            raise EngineError(f"unknown action {action!r}; expected one of {list(IMPACT_ACTIONS)}", 422)
        async with db_base.SessionLocal() as s:
            n = await s.get(Node, node_id)
            if n is None:
                raise EngineError("node not found", 404)
            g = await load_graph(s, n.run_id)
        node_ids, skipped = g.impact(node_id, action)
        return {"action": action, "node_ids": node_ids, "skipped_ids": skipped, "upper_bound": True}

    async def patch(
        self, node_id: UUID, *, title: str | None = None, question: str | None = None, instructions: str | None = None,
    ) -> Node:
        """Change a node's title/question/instructions: emit ``node.updated``, restart it and stale its descendants.

        Its next run may search the web live once (a new question needs fresh sources); changing a field back
        re-hashes to the old input hash and reuses the old version (no force)."""
        sch, run_id = await self._node_ctx(node_id)
        async with sch.mutex:
            async with db_base.SessionLocal() as s:
                g = await load_graph(s, run_id)
                n = g.nodes[node_id]
                if n.removed:
                    raise EngineError("node was removed", 409)
                if question is not None and n.type != "research":
                    raise EngineError("only research nodes have a question", 422)
                config = dict(n.config or {})
                changed = False
                if title is not None and title.strip() and title.strip() != n.title:
                    n.title = title.strip()
                    changed = True
                for k, val in (("question", question), ("instructions", instructions)):
                    if val is None:
                        continue
                    val = val.strip() or None
                    if val != config.get(k):
                        if val is None:
                            config.pop(k, None)
                        else:
                            config[k] = val
                        changed = True
                if not changed:
                    return n
                n.config = config
                await s.commit()
                await s.refresh(n)
                await bus.emit(run_id, "node.updated", node_id, {"node": node_out(n).model_dump(mode="json")})
                sch.cancel_task(node_id)
                if n.type != "human_input":
                    sch.web_live_once.add(node_id)
                if n.status in ("done", "failed", "running", "waiting_input"):
                    await sch.set_node_status(s, n, "stale", "config changed")
                await sch.mark_descendants_stale(s, g, node_id, f"upstream {n.key} changed")
                await self._reopen_run(sch, s)
        sch.wake()
        return n

    async def edit(self, node_id: UUID, output_text: str):
        sch, run_id = await self._node_ctx(node_id)
        async with sch.mutex:
            async with db_base.SessionLocal() as s:
                g = await load_graph(s, run_id)
                n = g.nodes[node_id]
                if n.removed:
                    raise EngineError("node was removed", 409)
                run = await s.get(Run, run_id)
                sch.cancel_task(node_id)
                prev = await s.get(NodeVersion, n.current_version_id) if n.current_version_id else None
                spec = SPECS[n.type]
                output = spec.parse_edit(output_text, dict(prev.output) if prev else {})
                steer_texts = [m.text for m in await sch._steers(s, node_id)]
                h = await sch._hash(s, g, n, run, steer_texts)
                v = await sch._new_version(s, n, h, output, output_text, "edit", model=None, tokens_in=None, tokens_out=None)
                if n.current_version_id is None:
                    n.current_version_id = v.id  # so apply_version sees a change and stales descendants
                    await s.commit()
                    await sch.apply_version(s, g, n, v, reused=False)
                    await sch.mark_descendants_stale(s, g, node_id, f"upstream {n.key} edited")
                else:
                    await sch.apply_version(s, g, n, v, reused=False)
                await sch.set_node_status(s, n, "done", "edit")
                await self._reopen_run(sch, s)
                vo = await version_out(s, v)
        sch.wake()
        return vo

    async def rerun(self, node_id: UUID) -> None:
        sch, run_id = await self._node_ctx(node_id)
        async with sch.mutex:
            async with db_base.SessionLocal() as s:
                n = await s.get(Node, node_id)
                if n.removed:
                    raise EngineError("node was removed", 409)
                sch.cancel_task(node_id)
                sch.force.add(node_id)
                await sch.set_node_status(s, n, "stale", "rerun")
                await self._reopen_run(sch, s)
        sch.wake()

    async def restore(self, node_id: UUID, version_id: UUID) -> None:
        sch, run_id = await self._node_ctx(node_id)
        async with sch.mutex:
            async with db_base.SessionLocal() as s:
                g = await load_graph(s, run_id)
                n = g.nodes[node_id]
                v = await s.get(NodeVersion, version_id)
                if v is None or v.node_id != node_id:
                    raise EngineError("version not found for this node", 404)
                if n.removed:
                    raise EngineError("node was removed", 409)
                sch.cancel_task(node_id)
                sch.force.discard(node_id)
                await sch.apply_version(s, g, n, v, reused=False, restore=True)
                # contract: descendants go stale on restore (even if it was already current)
                await sch.mark_descendants_stale(s, g, node_id, f"upstream {n.key} restored")
                await sch.set_node_status(s, n, "done", "restore")
                await self._reopen_run(sch, s)
        sch.wake()

    async def answer(self, node_id: UUID, answer: str) -> None:
        sch, run_id = await self._node_ctx(node_id)
        async with sch.mutex:
            async with db_base.SessionLocal() as s:
                g = await load_graph(s, run_id)
                n = g.nodes[node_id]
                if n.type != "human_input":
                    raise EngineError("node is not a human_input node", 409)
                if n.removed:
                    raise EngineError("node was removed", 409)
                run = await s.get(Run, run_id)
                h = await sch._hash(s, g, n, run, [])
                has_versions = n.current_version_id is not None
                v = await sch._new_version(
                    s, n, h, {"answer": answer}, answer, "recompute" if has_versions else "initial",
                    model=None, tokens_in=None, tokens_out=None,
                )
                await sch.apply_version(s, g, n, v, reused=False)
                await sch.set_node_status(s, n, "done", "answered")
                await self._reopen_run(sch, s)
        sch.wake()

    async def veto(self, node_id: UUID) -> None:
        sch, run_id = await self._node_ctx(node_id)
        async with sch.mutex:
            async with db_base.SessionLocal() as s:
                g = await load_graph(s, run_id)
                n = g.nodes[node_id]
                if n.removed:
                    return
                sch.cancel_task(node_id)
                n.removed = True
                await s.commit()
                await bus.emit(run_id, "node.removed", node_id, {})
                await sch.set_node_status(s, n, "skipped", "vetoed")
                g = await load_graph(s, run_id)
                # direct children that still have inputs recompute; ones with none get skipped by the tick
                await sch.mark_descendants_stale(s, g, node_id, f"input {n.key} removed")
                await self._reopen_run(sch, s)
        sch.wake()

    # ------------------------------------------------------------------ documents

    async def document_trust_changed(self, document_id: UUID) -> list[UUID]:
        """Nodes whose current retrievals touched the document (research, and debaters/judge that read it with
        tools) go stale and recompute; their input hash carries the corpus fingerprint, so they do not reuse.

        Returns the affected run ids (each gets a ``document.updated`` event from the caller)."""
        from livegraph.rag.models import Chunk, Retrieval

        async with db_base.SessionLocal() as s:
            rows = (
                await s.execute(
                    select(Node.id, Node.run_id)
                    .join(Retrieval, Retrieval.node_version_id == Node.current_version_id)
                    .join(Chunk, Chunk.id == Retrieval.chunk_id)
                    .where(Chunk.document_id == document_id, Node.type.in_(CORPUS_TYPES), Node.removed.is_(False))
                    .distinct()
                )
            ).all()
        by_run: dict[UUID, list[UUID]] = defaultdict(list)
        for nid, rid in rows:
            by_run[rid].append(nid)
        for rid, nids in by_run.items():
            sch = self.scheduler(rid)
            async with sch.mutex:
                sch._evidence_cache.clear()  # cached evidence carries document trust
                async with db_base.SessionLocal() as s:
                    g = await load_graph(s, rid)
                    for nid in nids:
                        n = g.nodes[nid]
                        if n.status in ("done", "failed"):
                            await sch.set_node_status(s, n, "stale", "document trust changed")
                            await sch.mark_descendants_stale(s, g, nid, f"upstream {n.key} changed")
                    await self._reopen_run(sch, s)
            sch.wake()
        return list(by_run)


engine = Engine()


__all__ = ["Engine", "EngineError", "RunScheduler", "engine", "load_graph"]
