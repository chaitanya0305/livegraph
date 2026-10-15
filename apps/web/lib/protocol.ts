/**
 * Wire types mirroring docs/CONTRACT.md exactly (snake_case, ISO datetimes, UUID strings).
 * Keep this file free of runtime logic beyond tiny type guards.
 */

export type UUID = string;
export type ISODateTime = string;

export type RunStatus =
  | "planning"
  | "running"
  | "paused"
  | "waiting_input"
  | "done"
  | "failed"
  | "cancelled";

export type NodeStatus =
  | "pending"
  | "running"
  | "waiting_input"
  | "done"
  | "failed"
  | "stale"
  | "skipped";

export type NodeType =
  | "plan"
  | "research"
  | "claims"
  | "prosecutor"
  | "defender"
  | "judge"
  | "human_input"
  | "editor"
  | "critic";

export type CreatedBy = "planner" | "critic" | "user";
export type VersionReason = "initial" | "steer" | "edit" | "recompute" | "retry" | "restore";
export type Trust = "normal" | "distrusted" | "pinned";
export type DocumentOrigin = "upload" | "web";
export type ClaimStatus = "active" | "struck";

export interface Run {
  id: UUID;
  goal: string;
  workflow: string;
  status: RunStatus;
  created_at: ISODateTime;
  updated_at: ISODateTime;
}

/**
 * The contract does not spell out RunSummary's fields. We treat it as the Run row
 * (optionally with extra counters the backend may add).
 */
export interface RunSummary extends Run {
  node_count?: number;
  claim_count?: number;
}

export interface NodeConfig {
  instructions?: string | null;
  question?: string | null;
  prompt?: string | null;
  params?: Record<string, unknown> | null;
  [k: string]: unknown;
}

export interface Position {
  x: number;
  y: number;
}

export interface Node {
  id: UUID;
  run_id: UUID;
  key: string;
  type: NodeType;
  title: string;
  status: NodeStatus;
  config: NodeConfig;
  position: Position | null;
  created_by: CreatedBy;
  current_version_id: UUID | null;
  removed: boolean;
}

export interface Edge {
  id: UUID;
  run_id: UUID;
  source_node_id: UUID;
  target_node_id: UUID;
}

export interface RetrievedChunk {
  chunk_id: UUID;
  document_id: UUID;
  document_title: string;
  source_url: string | null;
  text: string;
  score: number;
  trust: Trust;
}

export type RankedChunk = RetrievedChunk & { rank: number };

export interface NodeVersion {
  id: UUID;
  node_id: UUID;
  version: number;
  input_hash: string;
  /** Content hash (early cutoff); null for versions from before migration 0003. */
  output_hash?: string | null;
  output: unknown;
  output_text: string;
  reason: VersionReason;
  steer_text: string | null;
  model: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
  created_at: ISODateTime;
  retrievals?: RankedChunk[];
  /** Tool calls the agent made while producing this version (empty for single-shot steps). */
  trace?: TraceStep[];
}

// ---------------------------------------------------------------------------
// Agent tool activity
// ---------------------------------------------------------------------------

/** Known agent tools. The wire type is a plain string so unknown tools still render. */
export type ToolName =
  | "add_node"
  | "add_edge"
  | "finish_plan"
  | "list_documents"
  | "search_documents"
  | "web_search"
  | "read_source"
  | "fetch_page"
  | "request_follow_up_research"
  | "ask_rebuttal"
  | "list_agent_runs"
  | "read_agent_output"
  | "read_agent_trace"
  | "recall"
  | "search_context"
  | "request_upstream_work";

/** A passage a tool returned (lighter than RetrievedChunk; `snippet` is truncated text). */
export interface ToolChunk {
  chunk_id: UUID;
  document_id: UUID;
  title: string;
  score: number;
  trust: Trust;
  snippet: string;
}

export interface ExchangeMessage {
  from_agent: string;
  to_agent: string;
  kind: "question" | "answer";
  text: string;
  chunk_ids?: UUID[];
}

/** One tool call as persisted on a NodeVersion (and in RunSnapshot.activity). */
export interface TraceStep {
  call_id: string;
  tool: string;
  agent: string | null;
  parent_call_id: string | null;
  args_summary: string;
  /** null = still in flight (only ever seen in live activity, never in a persisted trace). */
  ok: boolean | null;
  summary: string;
  chunk_ids: UUID[];
  node_ids: UUID[];
  ms: number | null;
  ts: ISODateTime;
  exchange: ExchangeMessage[];
  /** Context store ref of the stored result (GET /api/runs/{id}/context/{ref}); absent/null when not stored. */
  ref?: string | null;
  /** Upstream agents whose work a context tool read. */
  reads?: TraceRead[];
}

export interface TraceRead {
  node_key: string;
}

/** GET /api/runs/{id}/context/{ref}: one stored tool result in full (inspector use). */
export interface ContextItem {
  run_id: UUID;
  ref: string;
  node_id: UUID;
  attempt_id: string;
  node_version_id: UUID | null;
  agent: string;
  parent_ref: string | null;
  kind: "tool_result" | "rebuttal";
  tool: string;
  args: Record<string, unknown>;
  ok: boolean;
  text: string;
  summary: string;
  chunk_ids: string[];
  tokens_est: number;
  created_at: ISODateTime;
}

export interface ClaimHistoryEntry {
  confidence: number | null;
  rationale: string | null;
  ts: ISODateTime;
}

export interface Claim {
  id: UUID;
  key: string;
  text: string;
  confidence: number | null;
  status: ClaimStatus;
  history: ClaimHistoryEntry[];
}

export interface Document {
  id: UUID;
  run_id: UUID | null;
  title: string;
  source_url: string | null;
  origin: DocumentOrigin;
  trust: Trust;
  created_at: ISODateTime;
}

export interface RunSnapshot {
  run: Run;
  nodes: Node[];
  edges: Edge[];
  versions: Record<UUID, NodeVersion>;
  claims: Claim[];
  documents: Document[];
  last_seq: number;
  /** Live tool steps of currently running nodes (optional for older backends). */
  activity?: Record<UUID, TraceStep[]>;
  /** Current attempt id per running node. */
  attempts?: Record<UUID, string>;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

export interface EventPayloads {
  "run.status": { status: RunStatus };
  "node.added": { node: Node };
  "edge.added": { edge: Edge };
  /** `attempt_id` is new each time a node enters `running`; tool events carry it. */
  "node.status": { status: NodeStatus; reason?: string | null; attempt_id?: string | null };
  /** `reset: true` replaces the live buffer with `text` instead of appending. */
  "node.delta": { text: string; reset?: boolean };
  "node.version": { version: NodeVersion; reused: boolean };
  "node.restored": { version_id: UUID };
  "node.removed": Record<string, never>;
  /** Title/config changed (PATCH /nodes/{id}); the node goes stale and recomputes. */
  "node.updated": { node: Node };
  "retrieval.done": { node_version_id: UUID; chunks: RankedChunk[] };
  "claim.upserted": { claim: Claim };
  "claim.confidence": {
    claim_id: UUID;
    key: string;
    confidence: number | null;
    previous: number | null;
    rationale: string | null;
  };
  /** `source`: "user", or "agent:<node_key>" for a downstream agent's request_upstream_work. */
  "steer.queued": { steer_id: UUID; text: string; source?: SteerSource };
  "steer.consumed": { steer_id: UUID; text: string; source?: SteerSource };
  "input.requested": { prompt: string };
  "document.updated": { document: Document };
  "tool.call": {
    attempt_id: string;
    call_id: string;
    tool: string;
    args: Record<string, unknown>;
    args_summary: string;
    agent: string | null;
    parent_call_id: string | null;
    step: number;
    budget: number | null;
  };
  "tool.result": {
    attempt_id: string;
    call_id: string;
    ok: boolean;
    summary: string;
    error: string | null;
    ms: number;
    chunks: ToolChunk[];
    node_ids: UUID[];
    ref?: string | null;
    reads?: TraceRead[];
  };
  "agent.message": {
    attempt_id: string;
    call_id: string;
    from_agent: string;
    to_agent: string;
    kind: "question" | "answer";
    text: string;
    chunk_ids: UUID[];
  };
  error: { message: string; node_id?: UUID | null };
}

export type EventType = keyof EventPayloads;

/** "user" or "agent:<node_key>". */
export type SteerSource = "user" | `agent:${string}`;

export const EVENT_TYPES = [
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
] as const satisfies readonly EventType[];

interface EnvelopeBase<T extends EventType> {
  /** null = ephemeral (not persisted, not replayable). */
  seq: number | null;
  run_id: UUID;
  type: T;
  node_id: UUID | null;
  payload: EventPayloads[T];
  ts: ISODateTime;
}

/** Discriminated union over `type`. */
export type EventEnvelope = { [T in EventType]: EnvelopeBase<T> }[EventType];

export type EventOf<T extends EventType> = Extract<EventEnvelope, { type: T }>;

/** A persisted event always carries a numeric seq. */
export type PersistedEvent = EventEnvelope & { seq: number };

export function isPersisted(e: EventEnvelope): e is PersistedEvent {
  return typeof e.seq === "number";
}

// REST bodies
export interface CreateRunBody {
  goal: string;
  workflow?: string;
  document_ids?: UUID[];
}

/** PATCH /api/nodes/{id} body (any subset; "" clears question/instructions). Response: Node. */
export interface NodePatch {
  title?: string;
  /** research nodes only */
  question?: string;
  instructions?: string;
}

export type ImpactAction = "steer" | "rerun" | "config" | "edit" | "restore" | "veto";

/** GET /api/nodes/{id}/impact?action=… — read-only preview; always an upper bound (early cutoff may stop sooner). */
export interface ImpactOut {
  action: ImpactAction;
  /** Nodes that would rerun, in topological order. */
  node_ids: UUID[];
  /** Nodes a veto would leave without inputs (skipped). */
  skipped_ids: UUID[];
  upper_bound: boolean;
}
