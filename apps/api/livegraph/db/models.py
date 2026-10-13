"""Core LiveGraph tables (see docs/CONTRACT.md §Data model)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from livegraph.db.base import Base


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )


def _ts() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    workflow: Mapped[str] = mapped_column(String(64), nullable=False, default="living_brief")
    # planning|running|paused|waiting_input|done|failed|cancelled
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="planning")
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Node(Base):
    __tablename__ = "nodes"
    __table_args__ = (UniqueConstraint("run_id", "key", name="uq_nodes_run_key"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # pending|running|waiting_input|done|failed|stale|skipped
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    position: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_by: Mapped[str] = mapped_column(String(16), nullable=False, default="planner")
    # No FK: avoids a nodes<->node_versions cycle; integrity is maintained by the engine.
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    removed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = _ts()


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (UniqueConstraint("source_node_id", "target_node_id", name="uq_edges_pair"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_node_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False
    )
    target_node_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False
    )


class NodeVersion(Base):
    __tablename__ = "node_versions"
    __table_args__ = (UniqueConstraint("node_id", "version", name="uq_node_versions_node_version"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    node_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    output_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # initial|steer|edit|recompute|retry|restore
    reason: Mapped[str] = mapped_column(String(16), nullable=False, default="initial")
    steer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = _ts()


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (PrimaryKeyConstraint("run_id", "seq", name="pk_events"),)

    run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    node_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    ts: Mapped[datetime] = _ts()


class SteerMessage(Base):
    __tablename__ = "steer_messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    node_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _ts()
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (UniqueConstraint("run_id", "key", name="uq_claims_run_key"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    key: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")  # active|struck
    source_node_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("nodes.id", ondelete="SET NULL"), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ClaimHistory(Base):
    __tablename__ = "claim_history"

    id: Mapped[uuid.UUID] = _uuid_pk()
    claim_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    node_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("node_versions.id", ondelete="SET NULL"), nullable=True
    )
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    ts: Mapped[datetime] = _ts()


# Make sure the RAG tables (documents, chunks, retrievals) are registered on Base.metadata
# so `create_all` creates them too.
import livegraph.rag.models  # noqa: E402,F401
