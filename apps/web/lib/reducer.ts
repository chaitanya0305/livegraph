/**
 * The single pure fold that powers both live updates and the timeline scrubber.
 *
 *   state(N) = events.filter(e => e.seq <= N).reduce(applyEvent, emptyState(base))
 *
 * Rules:
 * - Never mutates its inputs; returns a new object when anything changes.
 * - Persisted events with seq <= state.lastSeq are ignored (idempotent replay/resume).
 * - Ephemeral events (seq === null, i.e. node.delta) only touch the live text buffer.
 * - Tool events (tool.call / tool.result / agent.message) are persisted, so the scrubber
 *   shows an agent mid-tool. They only apply to a running node whose current attempt
 *   matches their `attempt_id`; events from a superseded attempt are dropped.
 */
import type {
  Claim,
  Document,
  Edge,
  EventEnvelope,
  EventOf,
  Node,
  NodeStatus,
  NodeVersion,
  RankedChunk,
  Run,
  RunSnapshot,
  RunStatus,
  SteerSource,
  ToolChunk,
  TraceStep,
  UUID,
} from "./protocol";

/** A tool step as tracked live; extends the persisted TraceStep with in-flight details. */
export interface ActivityStep extends TraceStep {
  step?: number;
  budget?: number | null;
  error?: string | null;
  args?: Record<string, unknown>;
  chunks?: ToolChunk[];
}

export type StepStatus = "running" | "ok" | "error";

export function stepStatus(step: TraceStep): StepStatus {
  if (step.ok === null || step.ok === undefined) return "running";
  return step.ok ? "ok" : "error";
}

export interface SteerEntry {
  steer_id: UUID;
  text: string;
  consumed: boolean;
  /** "user", or "agent:<node_key>" when a downstream agent asked for this (request_upstream_work). */
  source: SteerSource;
}

/** The agent key behind a steer, or null when it came from the user. */
export function steerAgent(source: SteerSource | string | null | undefined): string | null {
  return typeof source === "string" && source.startsWith("agent:") ? source.slice(6) || null : null;
}

/** "from you" / "from judge" */
export function steerFromLabel(source: SteerSource | string | null | undefined): string {
  const agent = steerAgent(source);
  return agent ? `from ${agent}` : "from you";
}

const AGENT_STEER_RE = /^Request from ([^:]+):\s*/;

/** Split an agent-originated steer text ("Request from judge: …") into its agent and instruction. */
export function splitSteerText(text: string): { agent: string | null; text: string } {
  const m = AGENT_STEER_RE.exec(text);
  return m ? { agent: m[1]!.trim(), text: text.slice(m[0].length) } : { agent: null, text };
}

export interface UpstreamRequest {
  target: string | null;
  instruction: string;
  state: "pending" | "accepted" | "rejected";
  /** rejection reason (tool summary/error) */
  reason: string | null;
}

/**
 * Details of a request_upstream_work step. Live steps carry `args`; persisted traces only have
 * `args_summary` ("target_node_key=research-1, instruction=…") and the summary ("→ research-1: …").
 */
export function upstreamRequest(step: ActivityStep): UpstreamRequest | null {
  if (step.tool !== "request_upstream_work") return null;
  let target = typeof step.args?.target_node_key === "string" ? step.args.target_node_key : null;
  let instruction = typeof step.args?.instruction === "string" ? step.args.instruction : "";
  if (!target || !instruction) {
    const m = /^target_node_key=(.*?), instruction=([\s\S]*)$/.exec(step.args_summary ?? "");
    if (m) {
      target ||= m[1]!.trim();
      instruction ||= m[2]!.trim();
    }
  }
  if (step.ok && step.summary) {
    const m = /^→\s*([^:]+):\s*([\s\S]*)$/.exec(step.summary);
    if (m) {
      target ||= m[1]!.trim();
      instruction ||= m[2]!.trim();
    }
  }
  const state = step.ok === null || step.ok === undefined ? "pending" : step.ok ? "accepted" : "rejected";
  return {
    target,
    instruction: instruction || step.args_summary || "",
    state,
    reason: state === "rejected" ? step.error || step.summary || null : null,
  };
}

export interface RunErrorEntry {
  seq: number | null;
  message: string;
  node_id: UUID | null;
  ts: string;
}

export interface RunState {
  run: Run | null;
  runStatus: RunStatus;
  nodes: Record<UUID, Node>;
  /** insertion order of nodes (stable for layout + lists) */
  nodeOrder: UUID[];
  edges: Record<UUID, Edge>;
  /** all known versions per node, sorted by version number */
  versions: Record<UUID, NodeVersion[]>;
  /** retrievals keyed by node_version_id */
  retrievals: Record<UUID, RankedChunk[]>;
  claims: Record<UUID, Claim>;
  claimOrder: UUID[];
  documents: Record<UUID, Document>;
  /** streaming text per node; cleared when a node.version arrives */
  live: Record<UUID, string>;
  steers: Record<UUID, SteerEntry[]>;
  /** human_input prompts per node */
  prompts: Record<UUID, string>;
  /** reason attached to the latest node.status event */
  statusReason: Record<UUID, string | null>;
  /** whether the node's latest node.version was a reuse (input-hash hit / early cutoff, no LLM call) */
  reused: Record<UUID, boolean>;
  /** tool steps of the node's latest attempt; reset when it enters running, kept after it finishes */
  activity: Record<UUID, ActivityStep[]>;
  /** current attempt id per node (set when it enters running) */
  attempts: Record<UUID, string>;
  /** chunks returned by tools during the current attempt; cleared when a node.version lands */
  provisional: Record<UUID, RankedChunk[]>;
  errors: RunErrorEntry[];
  lastSeq: number;
  /** ts of the most recent applied event (persisted or not) */
  lastTs: string | null;
}

export interface BaseContext {
  run?: Run | null;
  documents?: Document[];
}

export function emptyState(base: BaseContext = {}): RunState {
  const documents: Record<UUID, Document> = {};
  for (const d of base.documents ?? []) documents[d.id] = d;
  return {
    run: base.run ?? null,
    runStatus: "planning",
    nodes: {},
    nodeOrder: [],
    edges: {},
    versions: {},
    retrievals: {},
    claims: {},
    claimOrder: [],
    documents,
    live: {},
    steers: {},
    prompts: {},
    statusReason: {},
    reused: {},
    activity: {},
    attempts: {},
    provisional: {},
    errors: [],
    lastSeq: 0,
    lastTs: null,
  };
}

/** Seed state from a REST snapshot (live mode). Subsequent events with seq > last_seq fold on top. */
export function stateFromSnapshot(snap: RunSnapshot): RunState {
  const s = emptyState({ run: snap.run, documents: snap.documents });
  s.runStatus = snap.run.status;
  for (const n of snap.nodes) {
    s.nodes[n.id] = n;
    s.nodeOrder.push(n.id);
  }
  for (const e of snap.edges) s.edges[e.id] = e;
  for (const [nodeId, v] of Object.entries(snap.versions)) {
    s.versions[nodeId] = [v];
    if (v.retrievals) s.retrievals[v.id] = v.retrievals;
  }
  for (const c of snap.claims) {
    s.claims[c.id] = c;
    s.claimOrder.push(c.id);
  }
  for (const [nodeId, steps] of Object.entries(snap.activity ?? {})) {
    s.activity[nodeId] = steps.map((st) => ({
      ...st,
      exchange: st.exchange ?? [],
      chunk_ids: st.chunk_ids ?? [],
      node_ids: st.node_ids ?? [],
      ref: st.ref ?? null,
      reads: st.reads ?? [],
    }));
  }
  for (const [nodeId, attempt] of Object.entries(snap.attempts ?? {})) s.attempts[nodeId] = attempt;
  s.lastSeq = snap.last_seq;
  return s;
}

function upsertVersion(list: NodeVersion[] | undefined, v: NodeVersion): NodeVersion[] {
  const next = (list ?? []).filter((x) => x.id !== v.id);
  next.push(v);
  next.sort((a, b) => a.version - b.version);
  return next;
}

function setNode(state: RunState, id: UUID, patch: Partial<Node>): RunState {
  const node = state.nodes[id];
  if (!node) return state;
  return { ...state, nodes: { ...state.nodes, [id]: { ...node, ...patch } } };
}

/** Is this tool event addressed to the node's current, running attempt? */
function acceptsTool(state: RunState, id: UUID | null, attemptId: string | undefined | null): id is UUID {
  if (!id || state.nodes[id]?.status !== "running") return false;
  const current = state.attempts[id];
  return !current || !attemptId || current === attemptId;
}

function updateStep(
  state: RunState,
  id: UUID,
  callId: string,
  fn: (step: ActivityStep) => ActivityStep,
): RunState {
  const list = state.activity[id];
  const idx = list?.findIndex((x) => x.call_id === callId) ?? -1;
  if (!list || idx < 0) return state;
  const next = list.slice();
  next[idx] = fn(list[idx]!);
  return { ...state, activity: { ...state.activity, [id]: next } };
}

function mergeProvisional(
  existing: RankedChunk[] | undefined,
  chunks: readonly ToolChunk[],
  documents: Record<UUID, Document>,
): RankedChunk[] {
  const out = existing ? existing.slice() : [];
  const seen = new Set(out.map((c) => c.chunk_id));
  for (const c of chunks) {
    if (seen.has(c.chunk_id)) continue;
    seen.add(c.chunk_id);
    out.push({
      chunk_id: c.chunk_id,
      document_id: c.document_id,
      document_title: c.title,
      source_url: documents[c.document_id]?.source_url ?? null,
      text: c.snippet,
      score: c.score,
      trust: c.trust,
      rank: out.length + 1,
    });
  }
  return out;
}

function withoutKey<T>(rec: Record<string, T>, key: string): Record<string, T> {
  if (!(key in rec)) return rec;
  const next = { ...rec };
  delete next[key];
  return next;
}

export function applyEvent(state: RunState, event: EventEnvelope): RunState {
  // Idempotency: skip already-applied persisted events.
  if (event.seq !== null && event.seq <= state.lastSeq) return state;

  let s = applyPayload(state, event);
  if (event.seq !== null) {
    s = s === state ? { ...s } : s;
    s.lastSeq = event.seq;
    s.lastTs = event.ts;
  } else if (s !== state) {
    s.lastTs = event.ts;
  }
  return s;
}

function applyPayload(state: RunState, event: EventEnvelope): RunState {
  switch (event.type) {
    case "run.status":
      return {
        ...state,
        runStatus: event.payload.status,
        run: state.run ? { ...state.run, status: event.payload.status, updated_at: event.ts } : state.run,
      };

    case "node.added": {
      const node = event.payload.node;
      const exists = node.id in state.nodes;
      return {
        ...state,
        nodes: { ...state.nodes, [node.id]: node },
        nodeOrder: exists ? state.nodeOrder : [...state.nodeOrder, node.id],
      };
    }

    case "edge.added": {
      const edge = event.payload.edge;
      return { ...state, edges: { ...state.edges, [edge.id]: edge } };
    }

    case "node.status": {
      const id = event.node_id;
      if (!id || !state.nodes[id]) return state;
      const prev = state.nodes[id].status;
      const next: NodeStatus = event.payload.status;
      const attemptId = event.payload.attempt_id ?? null;
      let s = setNode(state, id, { status: next });
      s = { ...s, statusReason: { ...s.statusReason, [id]: event.payload.reason ?? null } };
      // A fresh attempt starts with an empty buffer and activity (steer restarts, recomputes).
      const freshAttempt =
        next === "running" && (prev !== "running" || (attemptId !== null && attemptId !== state.attempts[id]));
      if (freshAttempt) {
        s = {
          ...s,
          live: withoutKey(s.live, id),
          activity: { ...s.activity, [id]: [] },
          provisional: withoutKey(s.provisional, id),
          attempts: attemptId ? { ...s.attempts, [id]: attemptId } : withoutKey(s.attempts, id),
        };
      }
      return s;
    }

    case "node.delta": {
      const id = event.node_id;
      // Only running nodes stream; late deltas (after node.version / cancel) are dropped.
      if (!id || state.nodes[id]?.status !== "running") return state;
      const text = event.payload.reset ? event.payload.text : (state.live[id] ?? "") + event.payload.text;
      return { ...state, live: { ...state.live, [id]: text } };
    }

    case "tool.call": {
      const p = event.payload;
      const id = event.node_id;
      if (!acceptsTool(state, id, p.attempt_id)) return state;
      const step: ActivityStep = {
        call_id: p.call_id,
        tool: p.tool,
        agent: p.agent ?? null,
        parent_call_id: p.parent_call_id ?? null,
        args_summary: p.args_summary,
        ok: null,
        summary: "",
        chunk_ids: [],
        node_ids: [],
        ms: null,
        ts: event.ts,
        exchange: [],
        ref: null,
        reads: [],
        step: p.step,
        budget: p.budget,
        args: p.args,
      };
      const list = (state.activity[id] ?? []).filter((x) => x.call_id !== p.call_id);
      list.push(step);
      return { ...state, activity: { ...state.activity, [id]: list } };
    }

    case "tool.result": {
      const p = event.payload;
      const id = event.node_id;
      if (!acceptsTool(state, id, p.attempt_id)) return state;
      const chunks = p.chunks ?? [];
      const s = updateStep(state, id, p.call_id, (st) => ({
        ...st,
        ok: p.ok,
        summary: p.summary,
        error: p.error,
        ms: p.ms,
        chunks,
        chunk_ids: chunks.map((c) => c.chunk_id),
        node_ids: p.node_ids ?? [],
        ref: p.ref ?? null,
        reads: p.reads ?? [],
      }));
      if (s === state || !chunks.length) return s;
      return { ...s, provisional: { ...s.provisional, [id]: mergeProvisional(s.provisional[id], chunks, s.documents) } };
    }

    case "agent.message": {
      const p = event.payload;
      const id = event.node_id;
      if (!acceptsTool(state, id, p.attempt_id)) return state;
      return updateStep(state, id, p.call_id, (st) => ({
        ...st,
        exchange: [
          ...st.exchange,
          { from_agent: p.from_agent, to_agent: p.to_agent, kind: p.kind, text: p.text, chunk_ids: p.chunk_ids ?? [] },
        ],
      }));
    }

    case "node.version": {
      const v = event.payload.version;
      const id = event.node_id ?? v.node_id;
      let s: RunState = {
        ...state,
        versions: { ...state.versions, [id]: upsertVersion(state.versions[id], v) },
        live: withoutKey(state.live, id),
        reused: { ...state.reused, [id]: !!event.payload.reused },
        // activity is kept (it explains the version); tool chunks now live in retrievals
        provisional: withoutKey(state.provisional, id),
      };
      if (v.retrievals) s = { ...s, retrievals: { ...s.retrievals, [v.id]: v.retrievals } };
      return setNode(s, id, { current_version_id: v.id });
    }

    case "node.restored": {
      const id = event.node_id;
      if (!id) return state;
      return setNode(state, id, { current_version_id: event.payload.version_id });
    }

    case "node.removed": {
      const id = event.node_id;
      if (!id) return state;
      return setNode(state, id, { removed: true });
    }

    case "node.updated": {
      // PATCH /nodes/{id}: title/config changed (status changes arrive as node.status events)
      const id = event.node_id ?? event.payload.node.id;
      if (!id || !state.nodes[id]) return state;
      const { title, config, position } = event.payload.node;
      return setNode(state, id, { title, config: config ?? {}, ...(position !== undefined ? { position } : {}) });
    }

    case "retrieval.done":
      return {
        ...state,
        retrievals: { ...state.retrievals, [event.payload.node_version_id]: event.payload.chunks },
      };

    case "claim.upserted": {
      const incoming = event.payload.claim;
      const existing = state.claims[incoming.id];
      const claim: Claim = {
        ...incoming,
        // keep locally accumulated history if the upsert omits it
        history: incoming.history?.length ? incoming.history : existing?.history ?? [],
      };
      return {
        ...state,
        claims: { ...state.claims, [claim.id]: claim },
        claimOrder: existing ? state.claimOrder : [...state.claimOrder, claim.id],
      };
    }

    case "claim.confidence": {
      const p = event.payload;
      const existing = state.claims[p.claim_id];
      if (!existing) return state;
      const entry = { confidence: p.confidence, rationale: p.rationale, ts: event.ts };
      return {
        ...state,
        claims: {
          ...state.claims,
          [p.claim_id]: { ...existing, confidence: p.confidence, history: [...existing.history, entry] },
        },
      };
    }

    case "steer.queued": {
      const id = event.node_id;
      if (!id) return state;
      const list = (state.steers[id] ?? []).filter((x) => x.steer_id !== event.payload.steer_id);
      list.push({
        steer_id: event.payload.steer_id,
        text: event.payload.text,
        consumed: false,
        source: event.payload.source ?? "user",
      });
      return { ...state, steers: { ...state.steers, [id]: list } };
    }

    case "steer.consumed": {
      const id = event.node_id;
      if (!id) return state;
      const list = state.steers[id] ?? [];
      const found = list.some((x) => x.steer_id === event.payload.steer_id);
      const next = found
        ? list.map((x) => (x.steer_id === event.payload.steer_id ? { ...x, consumed: true } : x))
        : [
            ...list,
            {
              steer_id: event.payload.steer_id,
              text: event.payload.text,
              consumed: true,
              source: event.payload.source ?? "user",
            },
          ];
      return { ...state, steers: { ...state.steers, [id]: next } };
    }

    case "input.requested": {
      const id = event.node_id;
      if (!id) return state;
      return { ...state, prompts: { ...state.prompts, [id]: event.payload.prompt } };
    }

    case "document.updated": {
      const d = event.payload.document;
      // Keep trust shown on already-folded retrieval chunks in sync.
      let retrievals = state.retrievals;
      for (const [vid, chunks] of Object.entries(state.retrievals)) {
        if (chunks.some((c) => c.document_id === d.id && c.trust !== d.trust)) {
          if (retrievals === state.retrievals) retrievals = { ...state.retrievals };
          retrievals[vid] = chunks.map((c) => (c.document_id === d.id ? { ...c, trust: d.trust } : c));
        }
      }
      return { ...state, documents: { ...state.documents, [d.id]: d }, retrievals };
    }

    case "error":
      return {
        ...state,
        errors: [
          ...state.errors,
          {
            seq: event.seq,
            message: event.payload.message,
            node_id: event.payload.node_id ?? event.node_id ?? null,
            ts: event.ts,
          },
        ],
      };

    default: {
      // Unknown event types (forward compatibility) are ignored.
      const _exhaustive: never = event;
      void _exhaustive;
      return state;
    }
  }
}

/** Fold persisted events with seq <= upTo, starting from an empty run. */
export function foldEvents(
  events: readonly EventEnvelope[],
  upTo: number = Number.POSITIVE_INFINITY,
  base: BaseContext = {},
): RunState {
  let s = emptyState(base);
  for (const e of events) {
    if (e.seq === null) continue;
    if (e.seq > upTo) break;
    s = applyEvent(s, e);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Selectors (pure helpers used by components)
// ---------------------------------------------------------------------------

export function currentVersion(state: RunState, nodeId: UUID): NodeVersion | null {
  const node = state.nodes[nodeId];
  if (!node?.current_version_id) return null;
  return state.versions[nodeId]?.find((v) => v.id === node.current_version_id) ?? null;
}

const NO_STEPS: readonly ActivityStep[] = Object.freeze([]);

/**
 * The tool steps to show for a node: the live attempt while running (or after it failed),
 * otherwise the current version's persisted trace, falling back to the last live activity.
 * Returns stable references (safe as a store selector).
 */
export function traceFor(state: RunState, nodeId: UUID): readonly ActivityStep[] {
  const status = state.nodes[nodeId]?.status;
  const activity = state.activity[nodeId];
  if (status === "running" || status === "failed") return activity ?? NO_STEPS;
  return currentVersion(state, nodeId)?.trace ?? activity ?? NO_STEPS;
}

export function effectiveStatus(node: Node): NodeStatus {
  return node.removed ? "skipped" : node.status;
}

export function nodeIdByKey(state: RunState, key: string): UUID | null {
  for (const id of state.nodeOrder) if (state.nodes[id]?.key === key) return id;
  return null;
}

export function parentsOf(state: RunState, nodeId: UUID): UUID[] {
  return Object.values(state.edges)
    .filter((e) => e.target_node_id === nodeId)
    .map((e) => e.source_node_id);
}

export type { EventOf };
