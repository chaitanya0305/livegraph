"""The LiveGraph engine: one ``RunScheduler`` (asyncio task) per active run.

Concurrency model (single process):
- ``RunScheduler.mutex`` serializes every *state mutation* of a run (scheduler ticks, node start /
  finalize, user actions). LLM calls and retrieval run *outside* the mutex, bounded by a per-run
  semaphore (``settings.max_concurrency``).
- A tick (triggered by ``wake()``) reloads nodes/edges, applies the skip rule, starts every ready node
  (status ``pending``/``stale``) as its own task, and recomputes the run status.
- Each node task: compute ``input_hash`` -> reuse an existing version with the same hash (no LLM call)
  or stream the agent -> persist a new ``NodeVersion`` -> set current -> side effects (claims/judge/
  critic) -> status ``done``. Changing a node's current version marks all its descendants ``stale``;
  they recompute in topological order through the normal ready rule, and recomputation stops early
  wherever the hash is unchanged (reuse).
- Cancellation (steer/rerun/edit/restore/veto/cancel) cancels the node task; a cancelled task never
  writes (a task only finalizes if it is still the registered task for its node).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from livegraph import rag
from livegraph.agents import SPECS
from livegraph.agents import critic as critic_mod
from livegraph.agents import plan as plan_mod
from livegraph.agents import research as research_mod
from livegraph.agents.base import AgentContext, ParentOutput, stream_structured
from livegraph.agents.human_input import prompt_for
from livegraph.agents.models import model_for
from livegraph.config import settings
from livegraph.db import base as db_base
from livegraph.db.models import Claim, ClaimHistory, Edge, Node, NodeVersion, Run, SteerMessage
from livegraph.engine.hashing import input_hash
from livegraph.engine.serialize import claim_out, edge_out, node_out, version_out
from livegraph.events import bus

log = logging.getLogger(__name__)

ACTIVE_RUN_STATUSES = ("planning", "running")
RUNNABLE = ("pending", "stale")


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

    async def set_node_status(self, s: AsyncSession, n: Node, status: str, reason: str | None = None) -> None:
        if n.status == status:
            return
        n.status = status
        await s.commit()
        payload: dict[str, Any] = {"status": status}
        if reason:
            payload["reason"] = reason
        await bus.emit(self.run_id, "node.status", n.id, payload)

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
        for d in g.descendants(nid):
            if d.removed or d.status in ("skipped", "pending", "stale"):
                continue
            self.cancel_task(d.id)
            await self.set_node_status(s, d, "stale", reason)

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
        )
        if n.type in ("claims", "prosecutor", "defender", "judge", "editor"):
            ctx.evidence = await self._ancestor_evidence(s, g, n)
            ctx.claims = await self._ancestor_claims(s, g, n, with_confidence=n.type == "editor")
        return ctx

    async def _ancestor_evidence(self, s: AsyncSession, g: Graph, n: Node) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for a in g.ancestors(n.id):
            if a.type != "research" or Graph.inactive(a) or not a.current_version_id:
                continue
            for r in await rag.get_retrievals(s, a.current_version_id):
                cid = str(r["chunk_id"])
                if cid not in seen or r["score"] > seen[cid]["score"]:
                    seen[cid] = r
        return sorted(seen.values(), key=lambda r: r["score"], reverse=True)

    async def _ancestor_claims(self, s: AsyncSession, g: Graph, n: Node, *, with_confidence: bool) -> list[dict[str, Any]]:
        cnode = next((a for a in g.ancestors(n.id) if a.type == "claims" and not Graph.inactive(a)), None)
        if cnode is None or cnode.current_version_id is None:
            return []
        v = await s.get(NodeVersion, cnode.current_version_id)
        items = (v.output or {}).get("claims", []) if v else []
        rows = {c.key: c for c in (await s.execute(select(Claim).where(Claim.run_id == self.run_id))).scalars()}
        out = []
        for it in items:
            d = {"key": it.get("key"), "text": it.get("text"), "support_chunk_ids": it.get("support_chunk_ids", [])}
            row = rows.get(d["key"])
            if with_confidence and row is not None:
                d["confidence"] = row.confidence
                d["status"] = row.status
            out.append(d)
        return out

    async def _steers(self, s: AsyncSession, nid: UUID) -> list[SteerMessage]:
        return list(
            (await s.execute(select(SteerMessage).where(SteerMessage.node_id == nid).order_by(SteerMessage.created_at))).scalars()
        )

    async def _hash(self, g: Graph, n: Node, run: Run, steer_texts: list[str], docs: list[tuple[str, str]] | None = None) -> str:
        return input_hash(
            node_type=n.type,
            config=n.config or {},
            goal=run.goal,
            parent_version_ids=[str(p.current_version_id) for p in g.active_parents(n.id)],
            steer_texts=steer_texts,
            documents=docs,
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
                has_versions = bool(await s.scalar(select(func.count()).where(NodeVersion.node_id == nid)))
                reason = (
                    "retry" if force else "steer" if unconsumed else "recompute" if has_versions else "initial"
                )

                if n.type == "human_input":
                    h = await self._hash(g, n, run, steer_texts)
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

                if n.type != "research":
                    h = await self._hash(g, n, run, steer_texts)
                    if not force:
                        existing = await self._find_version_by_hash(s, nid, h)
                        if existing is not None:
                            await self.apply_version(s, g, n, existing, reused=True)
                            await self._consume_steers(s, unconsumed)
                            await self.set_node_status(s, n, "done")
                            self.tasks.pop(nid, None)
                            return
                await self.set_node_status(s, n, "running", reason)
                ctx = await self._context(s, g, n, run, steer_texts)
                node_type, question, goal = n.type, (n.config or {}).get("question") or n.title, run.goal

        # ---- phase 2 (no mutex): retrieval for research nodes
        chunks: list[Any] = []
        if node_type == "research":
            async with db_base.SessionLocal() as s:
                if force or not has_versions:
                    try:
                        await rag.web_research(s, run_id=self.run_id, query=question)
                        await s.commit()
                    except Exception as exc:  # web search is best-effort
                        await s.rollback()
                        log.warning("web_research failed: %s", exc)
                chunks = await rag.search(s, run_id=self.run_id, query=question, k=8)
            docs = sorted({(str(c.document_id), c.trust) for c in chunks})
            async with self.mutex:
                if not self._still_mine(nid):
                    return
                async with db_base.SessionLocal() as s:
                    run = await s.get(Run, self.run_id)
                    g = await load_graph(s, self.run_id)
                    n = g.nodes[nid]
                    h = await self._hash(g, n, run, steer_texts, docs)
                    if not force:
                        existing = await self._find_version_by_hash(s, nid, h)
                        if existing is not None:
                            await self.apply_version(s, g, n, existing, reused=True)
                            await self._consume_steers(s, unconsumed)
                            await self.set_node_status(s, n, "done")
                            self.tasks.pop(nid, None)
                            return
            ctx.evidence = [{**c.model_dump(mode="json")} for c in chunks]

        # ---- phase 3 (semaphore): the LLM call, streamed
        def on_delta(text: str) -> None:
            bus.delta(self.run_id, nid, text)

        deps = None
        if node_type == "plan":
            deps = self._planner_deps(nid)
        async with self.sem:
            result = await stream_structured(
                spec.agent,
                spec.build_prompt(ctx),
                model=model_for(node_type, spec.TIER),
                render=spec.render,
                on_delta=on_delta,
                deps=deps,
            )
        output = result.output
        if node_type == "research":
            output = research_mod.sanitize(output, {str(c.chunk_id) for c in chunks})
        elif node_type == "claims":
            output = SPECS["claims"].sanitize(output, {str(c["chunk_id"]) for c in ctx.evidence})
        output_text = spec.render(output)

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
                v = await self._new_version(
                    s, n, h, output.model_dump(mode="json"), output_text, reason,
                    steer_text="\n".join(m.text for m in unconsumed) or None,
                    model=result.model_name, tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                )
                if node_type == "research":
                    await rag.record_retrieval(s, node_version_id=v.id, chunks=chunks)
                    await s.commit()
                await self.apply_version(s, g, n, v, reused=False)
                if node_type == "research":
                    await bus.emit(
                        self.run_id, "retrieval.done", nid,
                        {"node_version_id": str(v.id), "chunks": await rag.get_retrievals(s, v.id)},
                    )
                await self._consume_steers(s, unconsumed)
                self.force.discard(nid)
                self.tasks.pop(nid, None)
                await self.set_node_status(s, n, "done")
                if node_type == "critic":
                    await self.critic_follow_up(s, n, output)

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
            await bus.emit(self.run_id, "steer.consumed", m.node_id, {"steer_id": str(m.id), "text": m.text})

    async def _new_version(self, s: AsyncSession, n: Node, h: str, output: dict, output_text: str, reason: str, **kw: Any) -> NodeVersion:
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
        if prev is not None and prev != v.id:
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

    async def critic_follow_up(self, s: AsyncSession, critic: Node, output: Any) -> None:
        out = output if isinstance(output, critic_mod.CriticOutput) else critic_mod.CriticOutput.model_validate(output)
        if out.verdict != "thin" or not (out.follow_up_question or "").strip():
            return
        g = await load_graph(s, self.run_id)
        sources = [p for p in g.active_parents(critic.id) if p.type == "research"]
        if not sources or any((p.config.get("params") or {}).get("follow_up") for p in sources):
            return  # never chain follow-ups of follow-ups
        src = sources[0]
        key = f"{src.key}_fu"
        if key in g.by_key:
            return
        new = await self.add_node(
            s, key=key, type="research", title=f"Follow-up: {src.title}"[:120],
            question=out.follow_up_question.strip(), created_by="critic",
            params={"follow_up": True, "critic": critic.key},
        )
        await self.add_edge(s, critic.key, new.key)
        g = await load_graph(s, self.run_id)
        for c in g.nodes.values():
            if c.type == "claims" and not c.removed:
                await self.add_edge(s, new.key, c.key)

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

    def _planner_deps(self, plan_id: UUID) -> plan_mod.PlannerDeps:
        me = asyncio.current_task()  # the plan node task (tools may run in their own tasks)

        def guard() -> None:
            if self.tasks.get(plan_id) is not me:
                raise asyncio.CancelledError()

        async def add_node(key: str, type: str, title: str, question: str | None = None, instructions: str | None = None) -> str:
            guard()
            if type == "plan":
                return "error: the plan node already exists"
            async with self.mutex:
                async with db_base.SessionLocal() as s:
                    g = await load_graph(s, self.run_id)
                    if _slug(key) in g.by_key:
                        return f"node {_slug(key)} already exists"
                    try:
                        n = await self.add_node(s, key=key, type=type, title=title, question=question, instructions=instructions)
                    except EngineError as e:
                        return f"error: {e}"
            bus.delta(self.run_id, plan_id, f"+ {n.type} `{n.key}`: {n.title}\n")
            return f"added node {n.key}"

        async def add_edge(source_key: str, target_key: str) -> str:
            guard()
            async with self.mutex:
                async with db_base.SessionLocal() as s:
                    try:
                        await self.add_edge(s, source_key, target_key)
                    except EngineError as e:
                        return f"error: {e}"
            return f"added edge {source_key}->{target_key}"

        async def finish_plan() -> str:
            guard()
            return "plan finished; now return the final output"

        return plan_mod.PlannerDeps(add_node=add_node, add_edge=add_edge, finish_plan=finish_plan)

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
        run = await s.get(Run, sch.run_id)
        if run is not None and run.status in ("done", "failed", "waiting_input"):
            await sch.set_run_status(s, run, "running")

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
                m = SteerMessage(node_id=node_id, text=text)
                s.add(m)
                await s.commit()
                await s.refresh(m)
                await bus.emit(run_id, "steer.queued", node_id, {"steer_id": str(m.id), "text": text})
                if n.status == "running":
                    sch.cancel_task(node_id)
                    await sch.set_node_status(s, n, "stale", "steer")
                elif n.status in ("done", "failed"):
                    await sch.set_node_status(s, n, "stale", "steer")
                    await sch.mark_descendants_stale(s, g, node_id, f"upstream {n.key} steered")
                    await self._reopen_run(sch, s)
                # pending/stale: the steer is picked up when it runs
        sch.wake()

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
                h = await sch._hash(g, n, run, steer_texts)
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
                h = await sch._hash(g, n, run, [])
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
        """Research nodes whose current retrieval touched the document go stale and recompute.

        Returns the affected run ids (each gets a ``document.updated`` event from the caller)."""
        from livegraph.rag.models import Chunk, Retrieval

        async with db_base.SessionLocal() as s:
            rows = (
                await s.execute(
                    select(Node.id, Node.run_id)
                    .join(Retrieval, Retrieval.node_version_id == Node.current_version_id)
                    .join(Chunk, Chunk.id == Retrieval.chunk_id)
                    .where(Chunk.document_id == document_id, Node.type == "research", Node.removed.is_(False))
                    .distinct()
                )
            ).all()
        by_run: dict[UUID, list[UUID]] = defaultdict(list)
        for nid, rid in rows:
            by_run[rid].append(nid)
        for rid, nids in by_run.items():
            sch = self.scheduler(rid)
            async with sch.mutex:
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
