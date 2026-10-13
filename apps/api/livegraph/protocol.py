"""Wire schemas: REST bodies/responses and event payloads (docs/CONTRACT.md)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from livegraph.rag.schemas import DocumentOut, RetrievedChunk

RunStatus = Literal["planning", "running", "paused", "waiting_input", "done", "failed", "cancelled"]
NodeStatus = Literal["pending", "running", "waiting_input", "done", "failed", "stale", "skipped"]
NodeType = Literal["plan", "research", "claims", "prosecutor", "defender", "judge", "human_input", "editor", "critic"]
VersionReason = Literal["initial", "steer", "edit", "recompute", "retry", "restore"]
CreatedBy = Literal["planner", "critic", "user"]
Trust = Literal["normal", "distrusted", "pinned"]

EventType = Literal[
    "run.status",
    "node.added",
    "edge.added",
    "node.status",
    "node.delta",
    "node.version",
    "node.restored",
    "node.removed",
    "retrieval.done",
    "claim.upserted",
    "claim.confidence",
    "steer.queued",
    "steer.consumed",
    "input.requested",
    "document.updated",
    "error",
]


class _ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------- resources


class RunSummary(_ORM):
    id: UUID
    goal: str
    workflow: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime


class NodeOut(_ORM):
    id: UUID
    run_id: UUID
    key: str
    type: NodeType
    title: str
    status: NodeStatus
    config: dict[str, Any]
    position: dict[str, Any] | None = None
    created_by: CreatedBy
    current_version_id: UUID | None = None
    removed: bool


class EdgeOut(_ORM):
    id: UUID
    run_id: UUID
    source_node_id: UUID
    target_node_id: UUID


class RetrievedChunkRanked(RetrievedChunk):
    rank: int


class NodeVersionOut(_ORM):
    id: UUID
    node_id: UUID
    version: int
    input_hash: str
    output: dict[str, Any]
    output_text: str
    reason: VersionReason
    steer_text: str | None = None
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    created_at: datetime
    retrievals: list[RetrievedChunkRanked] | None = None


class ClaimHistoryOut(BaseModel):
    confidence: float | None
    rationale: str
    ts: datetime


class ClaimOut(BaseModel):
    id: UUID
    key: str
    text: str
    confidence: float | None
    status: Literal["active", "struck"]
    history: list[ClaimHistoryOut] = Field(default_factory=list)


class RunSnapshot(BaseModel):
    run: RunSummary
    nodes: list[NodeOut]
    edges: list[EdgeOut]
    versions: dict[str, NodeVersionOut]
    claims: list[ClaimOut]
    documents: list[DocumentOut]
    last_seq: int


class EventEnvelope(BaseModel):
    seq: int | None
    run_id: UUID
    type: EventType
    node_id: UUID | None = None
    payload: dict[str, Any]
    ts: datetime


# ---------------------------------------------------------------- request bodies


class CreateRunBody(BaseModel):
    goal: str = Field(min_length=1)
    workflow: str = "living_brief"
    document_ids: list[UUID] = Field(default_factory=list)


class SteerBody(BaseModel):
    text: str = Field(min_length=1)


class EditBody(BaseModel):
    output_text: str


class RestoreBody(BaseModel):
    version_id: UUID


class InputBody(BaseModel):
    answer: str


class TrustBody(BaseModel):
    trust: Trust


class Accepted(BaseModel):
    ok: bool = True


# ---------------------------------------------------------------- event payloads


class RunStatusPayload(BaseModel):
    status: RunStatus


class NodeAddedPayload(BaseModel):
    node: NodeOut


class EdgeAddedPayload(BaseModel):
    edge: EdgeOut


class NodeStatusPayload(BaseModel):
    status: NodeStatus
    reason: str | None = None


class NodeDeltaPayload(BaseModel):
    text: str


class NodeVersionPayload(BaseModel):
    version: NodeVersionOut
    reused: bool


class NodeRestoredPayload(BaseModel):
    version_id: UUID


class RetrievalDonePayload(BaseModel):
    node_version_id: UUID
    chunks: list[RetrievedChunkRanked]


class ClaimUpsertedPayload(BaseModel):
    claim: ClaimOut


class ClaimConfidencePayload(BaseModel):
    claim_id: UUID
    key: str
    confidence: float | None
    previous: float | None
    rationale: str


class SteerPayload(BaseModel):
    steer_id: UUID
    text: str


class InputRequestedPayload(BaseModel):
    prompt: str


class DocumentUpdatedPayload(BaseModel):
    document: DocumentOut


class ErrorPayload(BaseModel):
    message: str
    node_id: UUID | None = None
