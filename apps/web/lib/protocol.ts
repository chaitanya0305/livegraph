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
  output: unknown;
  output_text: string;
  reason: VersionReason;
  steer_text: string | null;
  model: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
  created_at: ISODateTime;
  retrievals?: RankedChunk[];
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
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

export interface EventPayloads {
  "run.status": { status: RunStatus };
  "node.added": { node: Node };
  "edge.added": { edge: Edge };
  "node.status": { status: NodeStatus; reason?: string | null };
  "node.delta": { text: string };
  "node.version": { version: NodeVersion; reused: boolean };
  "node.restored": { version_id: UUID };
  "node.removed": Record<string, never>;
  "retrieval.done": { node_version_id: UUID; chunks: RankedChunk[] };
  "claim.upserted": { claim: Claim };
  "claim.confidence": {
    claim_id: UUID;
    key: string;
    confidence: number | null;
    previous: number | null;
    rationale: string | null;
  };
  "steer.queued": { steer_id: UUID; text: string };
  "steer.consumed": { steer_id: UUID; text: string };
  "input.requested": { prompt: string };
  "document.updated": { document: Document };
  error: { message: string; node_id?: UUID | null };
}

export type EventType = keyof EventPayloads;

export const EVENT_TYPES = [
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
