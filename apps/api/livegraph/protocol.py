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
    "node.updated",
    "retrieval.done",
    "claim.upserted",
    "claim.confidence",
    "steer.queued",
    "steer.consumed",
    "input.requested",
    "document.updated",
    "tool.call",
    "tool.result",
    "agent.message",
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
    cited: bool = False  # the node's output cites this chunk
    via: str | None = None  # how the agent got it: seed|search|read|web


class TraceExchange(BaseModel):
    from_agent: str
    to_agent: str
    kind: Literal["question", "answer"]
    text: str


class TraceRead(BaseModel):
    node_key: str  # an upstream agent whose work a context tool read


class TraceStep(BaseModel):
    """One tool call of an agent attempt (persisted on the version; folded from tool.* events live)."""

    call_id: str
    tool: str
    agent: str | None = None
    parent_call_id: str | None = None
    args_summary: str = ""
    ok: bool | None = None  # null while the call is still in flight (snapshot activity only)
    summary: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)
    ms: int = 0
    ts: str = ""
    exchange: list[TraceExchange] = Field(default_factory=list)
    ref: str | None = None  # context store ref of the stored result (GET /runs/{id}/context/{ref})
    reads: list[TraceRead] = Field(default_factory=list)


class NodeVersionOut(_ORM):
    id: UUID
    node_id: UUID
    version: int
    input_hash: str
    output_hash: str | None = None  # content hash (early cutoff); null for versions from before migration 0003
    output: dict[str, Any]
    output_text: str
    reason: VersionReason
    steer_text: str | None = None
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    created_at: datetime
    retrievals: list[RetrievedChunkRanked] | None = None
    trace: list[TraceStep] = Field(default_factory=list)


class ContextItemOut(_ORM):
    """One stored tool result / rebuttal (``context_items``)."""

    run_id: UUID
    ref: str
    node_id: UUID
    attempt_id: str
    node_version_id: UUID | None = None
    agent: str
    parent_ref: str | None = None
    kind: Literal["tool_result", "rebuttal"]
    tool: str
    args: dict[str, Any]
    ok: bool
    text: str
    summary: str
    chunk_ids: list[str] = Field(default_factory=list)
    tokens_est: int
    created_at: datetime


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
    # tool steps of the current attempt of every running node, and that attempt's id
    activity: dict[str, list[TraceStep]] = Field(default_factory=dict)
    attempts: dict[str, str] = Field(default_factory=dict)


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


class NodePatch(BaseModel):
    """PATCH /nodes/{id}: any subset; omitted fields are unchanged ("" clears question/instructions)."""

    title: str | None = Field(default=None, max_length=200)
    question: str | None = Field(default=None, max_length=2000)  # research nodes only
    instructions: str | None = Field(default=None, max_length=4000)


ImpactAction = Literal["steer", "rerun", "config", "edit", "restore", "veto"]


class ImpactOut(BaseModel):
    """GET /nodes/{id}/impact?action=…: nodes that would rerun (topological order) and nodes a veto would skip.

    Always an upper bound: early cutoff stops a recompute wherever a result comes out unchanged."""

    action: ImpactAction
    node_ids: list[UUID]
    skipped_ids: list[UUID] = Field(default_factory=list)
    upper_bound: bool = True


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
    attempt_id: str | None = None  # fresh on every transition to running


class NodeDeltaPayload(BaseModel):
    text: str
    reset: bool = False  # true: replace the live buffer with `text` (streamed output was superseded)


class NodeVersionPayload(BaseModel):
    version: NodeVersionOut
    reused: bool


class NodeRestoredPayload(BaseModel):
    version_id: UUID


class NodeUpdatedPayload(BaseModel):
    node: NodeOut  # title/config after a PATCH


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
    source: str = "user"  # "user" or "agent:<node_key>" (request_upstream_work)


class InputRequestedPayload(BaseModel):
    prompt: str


class DocumentUpdatedPayload(BaseModel):
    document: DocumentOut


class ToolChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    score: float
    trust: str
    snippet: str


class ToolCallPayload(BaseModel):
    attempt_id: str
    call_id: str
    tool: str
    args: dict[str, Any]
    args_summary: str
    agent: str | None = None
    parent_call_id: str | None = None
    step: int
    budget: int | None = None


class ToolResultPayload(BaseModel):
    attempt_id: str
    call_id: str
    ok: bool
    summary: str
    error: str | None = None
    ms: int
    chunks: list[ToolChunk] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)
    ref: str | None = None
    reads: list[TraceRead] = Field(default_factory=list)


class AgentMessagePayload(BaseModel):
    attempt_id: str
    call_id: str
    from_agent: str
    to_agent: str
    kind: Literal["question", "answer"]
    text: str
    chunk_ids: list[str] = Field(default_factory=list)


class ErrorPayload(BaseModel):
    message: str
    node_id: UUID | None = None
