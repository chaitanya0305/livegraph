"""ORM -> protocol conversions shared by the engine and the API."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from livegraph import rag
from livegraph.db.models import Claim, ClaimHistory, Edge, Node, NodeVersion, Run
from livegraph.protocol import (
    ClaimHistoryOut,
    ClaimOut,
    EdgeOut,
    NodeOut,
    NodeVersionOut,
    RetrievedChunkRanked,
    RunSummary,
)


def run_out(r: Run) -> RunSummary:
    return RunSummary.model_validate(r)


def node_out(n: Node) -> NodeOut:
    return NodeOut.model_validate(n)


def edge_out(e: Edge) -> EdgeOut:
    return EdgeOut.model_validate(e)


async def version_out(s: AsyncSession, v: NodeVersion, *, with_retrievals: bool = True) -> NodeVersionOut:
    out = NodeVersionOut.model_validate(v)
    if with_retrievals:
        rows = await rag.get_retrievals(s, v.id)
        out.retrievals = [RetrievedChunkRanked.model_validate(r) for r in rows] if rows else []
    return out


async def claim_out(s: AsyncSession, c: Claim) -> ClaimOut:
    hist = (
        await s.execute(select(ClaimHistory).where(ClaimHistory.claim_id == c.id).order_by(ClaimHistory.ts, ClaimHistory.id))
    ).scalars()
    return ClaimOut(
        id=c.id,
        key=c.key,
        text=c.text,
        confidence=c.confidence,
        status=c.status,  # type: ignore[arg-type]
        history=[ClaimHistoryOut(confidence=h.confidence, rationale=h.rationale, ts=h.ts) for h in hist],
    )


async def claims_for_run(s: AsyncSession, run_id: UUID) -> list[ClaimOut]:
    rows = (await s.execute(select(Claim).where(Claim.run_id == run_id))).scalars().all()
    rows = sorted(rows, key=lambda c: (_claim_sort_key(c.key)))
    return [await claim_out(s, c) for c in rows]


def _claim_sort_key(key: str) -> tuple[int, str]:
    digits = "".join(ch for ch in key if ch.isdigit())
    return (int(digits) if digits else 10**9, key)
