from __future__ import annotations

import contextlib
from uuid import UUID

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select

from livegraph import rag
from livegraph.db import base as db_base
from livegraph.db.models import Edge, Event, Node, NodeVersion, Run
from livegraph.engine import engine
from livegraph.engine.serialize import claims_for_run, edge_out, node_out, run_out, version_out
from livegraph.events import bus
from livegraph.protocol import CreateRunBody, EventEnvelope, RunSnapshot, RunSummary

router = APIRouter(tags=["runs"])


async def build_snapshot(run_id: UUID) -> RunSnapshot:
    sch = engine.schedulers.get(run_id)
    # hold the run mutex (all state changes + their events happen under it) for a consistent cut
    lock = sch.mutex if sch is not None else contextlib.nullcontext()
    async with lock:
        async with db_base.SessionLocal() as s:
            run = await s.get(Run, run_id)
            if run is None:
                raise HTTPException(404, "run not found")
            last_seq = int(await s.scalar(select(func.coalesce(func.max(Event.seq), 0)).where(Event.run_id == run_id)) or 0)
            nodes = list((await s.execute(select(Node).where(Node.run_id == run_id).order_by(Node.created_at, Node.key))).scalars())
            edges = list((await s.execute(select(Edge).where(Edge.run_id == run_id))).scalars())
            versions = {}
            for n in nodes:
                if n.current_version_id:
                    v = await s.get(NodeVersion, n.current_version_id)
                    if v is not None:
                        versions[str(n.id)] = await version_out(s, v)
            return RunSnapshot(
                run=run_out(run),
                nodes=[node_out(n) for n in nodes],
                edges=[edge_out(e) for e in edges],
                versions=versions,
                claims=await claims_for_run(s, run_id),
                documents=await rag.list_documents(s, run_id),
                last_seq=last_seq,
            )


async def _summary(run_id: UUID) -> RunSummary:
    async with db_base.SessionLocal() as s:
        run = await s.get(Run, run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return run_out(run)


@router.post("/runs", response_model=RunSnapshot)
async def create_run(body: CreateRunBody) -> RunSnapshot:
    run_id = await engine.create_run(body.goal, body.workflow, body.document_ids)
    return await build_snapshot(run_id)


@router.get("/runs", response_model=list[RunSummary])
async def list_runs() -> list[RunSummary]:
    async with db_base.SessionLocal() as s:
        rows = (await s.execute(select(Run).order_by(Run.created_at.desc()))).scalars()
        return [run_out(r) for r in rows]


@router.get("/runs/{run_id}", response_model=RunSnapshot)
async def get_run(run_id: UUID) -> RunSnapshot:
    return await build_snapshot(run_id)


@router.get("/runs/{run_id}/events", response_model=list[EventEnvelope])
async def get_events(run_id: UUID, since: int = 0) -> list[EventEnvelope]:
    await _summary(run_id)
    return await bus.load_events(run_id, since)


@router.post("/runs/{run_id}/pause", response_model=RunSummary)
async def pause(run_id: UUID) -> RunSummary:
    await _summary(run_id)
    await engine.pause(run_id)
    return await _summary(run_id)


@router.post("/runs/{run_id}/resume", response_model=RunSummary)
async def resume(run_id: UUID) -> RunSummary:
    await _summary(run_id)
    await engine.resume(run_id)
    return await _summary(run_id)


@router.post("/runs/{run_id}/cancel", response_model=RunSummary)
async def cancel(run_id: UUID) -> RunSummary:
    await _summary(run_id)
    await engine.cancel(run_id)
    return await _summary(run_id)
