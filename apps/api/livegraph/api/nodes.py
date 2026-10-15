from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from livegraph.db import base as db_base
from livegraph.db.models import Node, NodeVersion
from livegraph.engine import engine
from livegraph.engine.serialize import node_out, version_out
from livegraph.protocol import (
    Accepted,
    EditBody,
    ImpactAction,
    ImpactOut,
    InputBody,
    NodeOut,
    NodePatch,
    NodeVersionOut,
    RestoreBody,
    SteerBody,
)

router = APIRouter(tags=["nodes"])


@router.get("/nodes/{node_id}/versions", response_model=list[NodeVersionOut])
async def versions(node_id: UUID) -> list[NodeVersionOut]:
    async with db_base.SessionLocal() as s:
        if await s.get(Node, node_id) is None:
            raise HTTPException(404, "node not found")
        rows = (
            await s.execute(select(NodeVersion).where(NodeVersion.node_id == node_id).order_by(NodeVersion.version))
        ).scalars()
        return [await version_out(s, v) for v in rows]


@router.get("/nodes/{node_id}/impact", response_model=ImpactOut)
async def impact(node_id: UUID, action: ImpactAction = Query(...)) -> ImpactOut:
    """Read-only preview of the nodes an action would rerun (no state change, no events)."""
    return ImpactOut.model_validate(await engine.impact(node_id, action))


@router.patch("/nodes/{node_id}", response_model=NodeOut)
async def patch(node_id: UUID, body: NodePatch) -> NodeOut:
    if body.title is None and body.question is None and body.instructions is None:
        raise HTTPException(422, "nothing to change: give title, question or instructions")
    n = await engine.patch(node_id, title=body.title, question=body.question, instructions=body.instructions)
    return node_out(n)


@router.post("/nodes/{node_id}/steer", status_code=202, response_model=Accepted)
async def steer(node_id: UUID, body: SteerBody) -> Accepted:
    await engine.steer(node_id, body.text)
    return Accepted()


@router.post("/nodes/{node_id}/edit", response_model=NodeVersionOut)
async def edit(node_id: UUID, body: EditBody) -> NodeVersionOut:
    return await engine.edit(node_id, body.output_text)


@router.post("/nodes/{node_id}/rerun", status_code=202, response_model=Accepted)
async def rerun(node_id: UUID) -> Accepted:
    await engine.rerun(node_id)
    return Accepted()


@router.post("/nodes/{node_id}/restore", status_code=202, response_model=Accepted)
async def restore(node_id: UUID, body: RestoreBody) -> Accepted:
    await engine.restore(node_id, body.version_id)
    return Accepted()


@router.post("/nodes/{node_id}/input", status_code=202, response_model=Accepted)
async def answer(node_id: UUID, body: InputBody) -> Accepted:
    await engine.answer(node_id, body.answer)
    return Accepted()


@router.delete("/nodes/{node_id}", status_code=202, response_model=Accepted)
async def veto(node_id: UUID) -> Accepted:
    await engine.veto(node_id)
    return Accepted()
