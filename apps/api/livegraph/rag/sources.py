"""Read access to individual chunks for agents: a chunk plus its neighbours, respecting run scope + trust."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import String, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from livegraph.rag.models import Chunk, Document
from livegraph.rag.schemas import RetrievedChunk


async def resolve_chunk_id(session: AsyncSession, *, run_id: UUID, ref: str) -> UUID | None:
    """A full chunk UUID, or an 8+ char prefix (rendered citations show 8), within the run's scope."""
    ref = ref.strip().strip("[]").lower()
    try:
        return UUID(ref)
    except ValueError:
        pass
    if len(ref) < 8 or any(ch not in "0123456789abcdef-" for ch in ref):
        return None
    stmt = (
        select(Chunk.id)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            or_(Document.run_id == run_id, Document.run_id.is_(None)),
            cast(Chunk.id, String).like(f"{ref}%"),
        )
        .limit(2)
    )
    rows = list((await session.execute(stmt)).scalars())
    return rows[0] if len(rows) == 1 else None


async def get_chunk_window(
    session: AsyncSession, *, run_id: UUID, chunk_id: UUID, before: int = 1, after: int = 1
) -> list[RetrievedChunk]:
    """The chunk and up to ``before``/``after`` neighbours of the same document, in document order.

    Empty when the chunk is unknown, outside the run's scope (run docs + global library) or its
    document is distrusted. ``score`` is 0 (these are reads, not ranked hits).
    """
    row = (
        await session.execute(
            select(Chunk.idx, Chunk.document_id, Document.trust, Document.run_id)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id == chunk_id)
        )
    ).first()
    if row is None or row.trust == "distrusted" or row.run_id not in (run_id, None):
        return []
    stmt = (
        select(Chunk.id, Chunk.document_id, Chunk.text, Document.title, Document.source_url, Document.trust)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.document_id == row.document_id, Chunk.idx.between(row.idx - before, row.idx + after))
        .order_by(Chunk.idx)
    )
    return [
        RetrievedChunk(
            chunk_id=r.id, document_id=r.document_id, document_title=r.title, source_url=r.source_url,
            text=r.text, score=0.0, trust=r.trust,
        )
        for r in (await session.execute(stmt)).all()
    ]
