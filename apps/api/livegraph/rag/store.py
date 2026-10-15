"""Ingest, search, trust and retrieval bookkeeping.

All functions take an ``AsyncSession`` and only ``flush()`` -- the caller owns the transaction
and must ``commit()``.
"""

from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession

from livegraph.rag.chunking import chunk_text, extract_text
from livegraph.rag.embeddings import get_embedder
from livegraph.rag.models import Chunk, Document, Retrieval
from livegraph.rag.schemas import DocumentOut, RetrievedChunk

log = logging.getLogger(__name__)

PIN_BOOST = 0.15
PIN_MIN_RAW_SCORE = 0.2
TRUST_VALUES = ("normal", "distrusted", "pinned")
SUPPORTED_EXTS = (".txt", ".md", ".markdown", ".pdf")


async def ingest_text(
    session: AsyncSession,
    *,
    run_id: UUID | None,
    title: str,
    text: str,
    source_url: str | None = None,
    origin: Literal["upload", "web"] = "upload",
) -> Document:
    pieces = chunk_text(text)
    vectors = await get_embedder().embed(pieces) if pieces else []
    doc = Document(run_id=run_id, title=title[:500] or "Untitled", source_url=source_url, origin=origin)
    session.add(doc)
    await session.flush()
    session.add_all(
        Chunk(document_id=doc.id, idx=i, text=t, embedding=v) for i, (t, v) in enumerate(zip(pieces, vectors))
    )
    await session.flush()
    await session.refresh(doc)
    return doc


async def ingest_file(session: AsyncSession, *, run_id: UUID | None, filename: str, data: bytes) -> Document:
    """Ingest a .txt / .md / .pdf upload. Raises ValueError for unsupported types or empty text."""
    lower = filename.lower()
    if not lower.endswith(SUPPORTED_EXTS):
        raise ValueError(f"unsupported file type: {filename} (expected .txt, .md or .pdf)")
    text = extract_text(filename, data)
    if not text.strip():
        raise ValueError(f"no extractable text in {filename}")
    return await ingest_text(session, run_id=run_id, title=filename, text=text, origin="upload")


def _scope(run_id: UUID, document_ids: list[UUID] | None = None):
    conds = [or_(Document.run_id == run_id, Document.run_id.is_(None)), Document.trust != "distrusted"]
    if document_ids is not None:
        conds.append(Document.id.in_(document_ids))
    return conds


async def search(
    session: AsyncSession, *, run_id: UUID, query: str, k: int = 6, document_ids: list[UUID] | None = None
) -> list[RetrievedChunk]:
    """Cosine search over run-scoped + global docs, excluding distrusted ones.

    score = 1 - cosine_distance (+0.15 for pinned docs). Each pinned doc's best chunk is
    guaranteed a slot if its raw score > 0.2. ``document_ids`` restricts the search to those docs.
    """
    if k <= 0 or not query.strip() or document_ids == []:
        return []
    [qvec] = await get_embedder().embed([query])
    dist = Chunk.embedding.cosine_distance(qvec).label("dist")
    cols = (Chunk.id, Chunk.document_id, Chunk.text, Document.title, Document.source_url, Document.trust, dist)

    stmt = (
        select(*cols)
        .join(Document, Chunk.document_id == Document.id)
        .where(*_scope(run_id, document_ids))
        .order_by(dist)
        .limit(max(k * 4, 24))
    )
    rows = (await session.execute(stmt)).all()

    def to_rc(r) -> RetrievedChunk:
        raw = 1.0 - float(r.dist)
        return RetrievedChunk(
            chunk_id=r.id, document_id=r.document_id, document_title=r.title, source_url=r.source_url,
            text=r.text, score=raw + (PIN_BOOST if r.trust == "pinned" else 0.0), trust=r.trust,
        )

    # chunk id breaks score ties so equal-scoring chunks always rank the same way (stable input hashes)
    cands = sorted((to_rc(r) for r in rows), key=lambda c: (-c.score, str(c.chunk_id)))
    results = cands[:k]

    # Guarantee each relevant pinned doc's best chunk.
    pinned_stmt = (
        select(*cols)
        .join(Document, Chunk.document_id == Document.id)
        .where(*_scope(run_id, document_ids), Document.trust == "pinned")
        .order_by(Chunk.document_id, dist)
        .ext(distinct_on(Chunk.document_id))
    )
    present_docs = {c.document_id for c in results}
    must: list[RetrievedChunk] = []
    for r in (await session.execute(pinned_stmt)).all():
        if r.document_id in present_docs or 1.0 - float(r.dist) <= PIN_MIN_RAW_SCORE:
            continue
        must.append(to_rc(r))
    for m in must:
        # Replace the weakest non-pinned result, or append if every slot is pinned.
        idx = next((i for i in range(len(results) - 1, -1, -1) if results[i].trust != "pinned"), None)
        if idx is not None and len(results) >= k:
            results.pop(idx)
        results.append(m)
    results.sort(key=lambda c: c.score, reverse=True)
    return results


async def record_retrieval(
    session: AsyncSession,
    *,
    node_version_id: UUID,
    chunks: list[RetrievedChunk],
    cited: set[str] | None = None,
    via: dict[str, str] | None = None,
) -> None:
    """Persist retrievals for a node version; rank is the 1-based position in ``chunks``."""
    cited, via = cited or set(), via or {}
    session.add_all(
        Retrieval(
            node_version_id=node_version_id, chunk_id=c.chunk_id, score=c.score, rank=i,
            cited=str(c.chunk_id) in cited, via=via.get(str(c.chunk_id)),
        )
        for i, c in enumerate(chunks, start=1)
    )
    await session.flush()


async def get_retrievals(session: AsyncSession, node_version_id: UUID) -> list[dict]:
    """Retrievals of a node version as ``RetrievedChunk`` dicts plus ``rank``, ordered by rank."""
    stmt = (
        select(Retrieval, Chunk, Document)
        .join(Chunk, Retrieval.chunk_id == Chunk.id)
        .join(Document, Chunk.document_id == Document.id)
        .where(Retrieval.node_version_id == node_version_id)
        .order_by(Retrieval.rank)
    )
    out = []
    for r, c, d in (await session.execute(stmt)).all():
        rc = RetrievedChunk(
            chunk_id=c.id, document_id=d.id, document_title=d.title, source_url=d.source_url,
            text=c.text, score=r.score, trust=d.trust,
        )
        out.append({**rc.model_dump(mode="json"), "rank": r.rank, "cited": bool(r.cited), "via": r.via})
    return out


async def get_retrievals_many(session: AsyncSession, node_version_ids: list[UUID]) -> dict[UUID, list[dict]]:
    """``get_retrievals`` for several versions in one query: {version_id: [dict, ...]} (each ordered by rank)."""
    out: dict[UUID, list[dict]] = {vid: [] for vid in node_version_ids}
    if not node_version_ids:
        return out
    stmt = (
        select(Retrieval, Chunk, Document)
        .join(Chunk, Retrieval.chunk_id == Chunk.id)
        .join(Document, Chunk.document_id == Document.id)
        .where(Retrieval.node_version_id.in_(list(node_version_ids)))
        .order_by(Retrieval.node_version_id, Retrieval.rank)
    )
    for r, c, d in (await session.execute(stmt)).all():
        rc = RetrievedChunk(
            chunk_id=c.id, document_id=d.id, document_title=d.title, source_url=d.source_url,
            text=c.text, score=r.score, trust=d.trust,
        )
        out[r.node_version_id].append({**rc.model_dump(mode="json"), "rank": r.rank, "cited": bool(r.cited), "via": r.via})
    return out


async def get_document(session: AsyncSession, document_id: UUID) -> Document | None:
    return await session.get(Document, document_id)


async def set_trust(session: AsyncSession, document_id: UUID, trust: str) -> Document:
    """Set a document's trust. Raises ValueError for bad trust, LookupError if not found."""
    if trust not in TRUST_VALUES:
        raise ValueError(f"invalid trust {trust!r}; expected one of {TRUST_VALUES}")
    doc = await session.get(Document, document_id)
    if doc is None:
        raise LookupError(f"document {document_id} not found")
    doc.trust = trust
    await session.flush()
    return doc


async def _chunk_counts(session: AsyncSession, ids: list[UUID]) -> dict[UUID, int]:
    if not ids:
        return {}
    stmt = select(Chunk.document_id, func.count()).where(Chunk.document_id.in_(ids)).group_by(Chunk.document_id)
    return dict((await session.execute(stmt)).all())


async def document_out(session: AsyncSession, doc: Document) -> DocumentOut:
    counts = await _chunk_counts(session, [doc.id])
    return _to_out(doc, counts.get(doc.id, 0))


def _to_out(doc: Document, n: int) -> DocumentOut:
    return DocumentOut(
        id=doc.id, run_id=doc.run_id, title=doc.title, source_url=doc.source_url, origin=doc.origin,
        trust=doc.trust, created_at=doc.created_at, chunk_count=n,
    )


async def list_documents(session: AsyncSession, run_id: UUID | None = None) -> list[DocumentOut]:
    """Global docs plus (if run_id given) docs scoped to that run, oldest first."""
    cond = Document.run_id.is_(None) if run_id is None else or_(Document.run_id == run_id, Document.run_id.is_(None))
    docs = list((await session.execute(select(Document).where(cond).order_by(Document.created_at))).scalars())
    counts = await _chunk_counts(session, [d.id for d in docs])
    return [_to_out(d, counts.get(d.id, 0)) for d in docs]


async def corpus_fingerprint(session: AsyncSession, run_id: UUID) -> list[tuple[str, str]]:
    """Sorted (document_id, trust) of in-scope docs that are uploads or have a non-default trust.

    Part of a corpus-reading node's input hash: an upload or a distrust/pin changes it (so the node
    recomputes), while web pages the agents themselves ingest (origin=web, trust normal) do not.
    """
    stmt = select(Document.id, Document.trust).where(
        or_(Document.run_id == run_id, Document.run_id.is_(None)),
        or_(Document.origin == "upload", Document.trust != "normal"),
    )
    return sorted((str(i), t) for i, t in (await session.execute(stmt)).all())
