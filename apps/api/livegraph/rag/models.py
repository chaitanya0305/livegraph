"""SQLAlchemy models for the RAG module: documents, chunks, retrievals."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from livegraph.config import settings
from livegraph.db.base import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # one document per (run, url): concurrent web ingests of the same page cannot duplicate it
        Index(
            "uq_documents_run_source_url", "run_id", "source_url", unique=True,
            postgresql_where=text("source_url IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    # Plain UUID (no FK) so the RAG tables don't depend on runs existing; NULL = global library.
    run_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    origin: Mapped[str] = mapped_column(String(16), nullable=False, default="upload", server_default="upload")
    trust: Mapped[str] = mapped_column(String(16), nullable=False, default="normal", server_default="normal")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True,
        order_by="Chunk.idx",
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embed_dim), nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Retrieval(Base):
    __tablename__ = "retrievals"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    node_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("node_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    # cited by the node's output (cited rows rank first); how the agent got it: seed|search|read|web
    cited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    via: Mapped[str | None] = mapped_column(String(8), nullable=True)


class WebCache(Base):
    """Per-run web search results by normalized query, so recomputes never re-hit the web (Rerun bypasses)."""

    __tablename__ = "web_cache"
    __table_args__ = (UniqueConstraint("run_id", "query_norm", name="uq_web_cache_run_query"),)

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    run_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    query_norm: Mapped[str] = mapped_column(Text, nullable=False)
    doc_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
