/**
 * The single pure fold that powers both live updates and the timeline scrubber.
 *
 *   state(N) = events.filter(e => e.seq <= N).reduce(applyEvent, emptyState(base))
 *
 * Rules:
 * - Never mutates its inputs; returns a new object when anything changes.
 * - Persisted events with seq <= state.lastSeq are ignored (idempotent replay/resume).
 * - Ephemeral events (seq === null, i.e. node.delta) only touch the live text buffer.
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
  UUID,
} from "./protocol";

export interface SteerEntry {
  steer_id: UUID;
  text: string;
  consumed: boolean;
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
      let s = setNode(state, id, { status: next });
      s = { ...s, statusReason: { ...s.statusReason, [id]: event.payload.reason ?? null } };
      // A fresh run starts with an empty buffer (steer restarts, recomputes).
      if (next === "running" && prev !== "running") s = { ...s, live: withoutKey(s.live, id) };
      return s;
    }

    case "node.delta": {
      const id = event.node_id;
      // Only running nodes stream; late deltas (after node.version / cancel) are dropped.
      if (!id || state.nodes[id]?.status !== "running") return state;
      return { ...state, live: { ...state.live, [id]: (state.live[id] ?? "") + event.payload.text } };
    }

    case "node.version": {
      const v = event.payload.version;
      const id = event.node_id ?? v.node_id;
      let s: RunState = {
        ...state,
        versions: { ...state.versions, [id]: upsertVersion(state.versions[id], v) },
        live: withoutKey(state.live, id),
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
      list.push({ steer_id: event.payload.steer_id, text: event.payload.text, consumed: false });
      return { ...state, steers: { ...state.steers, [id]: list } };
    }

    case "steer.consumed": {
      const id = event.node_id;
      if (!id) return state;
      const list = state.steers[id] ?? [];
      const found = list.some((x) => x.steer_id === event.payload.steer_id);
      const next = found
        ? list.map((x) => (x.steer_id === event.payload.steer_id ? { ...x, consumed: true } : x))
        : [...list, { steer_id: event.payload.steer_id, text: event.payload.text, consumed: true }];
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

export function effectiveStatus(node: Node): NodeStatus {
  return node.removed ? "skipped" : node.status;
}

export function parentsOf(state: RunState, nodeId: UUID): UUID[] {
  return Object.values(state.edges)
    .filter((e) => e.target_node_id === nodeId)
    .map((e) => e.source_node_id);
}

export type { EventOf };
