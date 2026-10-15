"use client";

import { create } from "zustand";
import { ApiError, BackendUnreachableError } from "./api";
import { getBackend, type Backend } from "./backend";
import type { ImpactPreview } from "./impact";
import type { EventEnvelope, PersistedEvent, UUID } from "./protocol";
import { applyEvent, emptyState, stateFromSnapshot, type RunState } from "./reducer";
import type { ConnectionState, Subscription } from "./socket";

export type LoadError = { kind: "unreachable" | "not_found" | "other"; message: string };

interface RunStore {
  runId: UUID | null;
  backend: Backend | null;
  live: RunState | null;
  /** persisted events, sorted by seq, deduped */
  log: PersistedEvent[];
  /** null = live; otherwise the seq being viewed */
  scrubSeq: number | null;
  /** fold of log up to scrubSeq (only when scrubbing) */
  historical: RunState | null;
  selectedNodeId: UUID | null;
  connection: ConnectionState;
  loadError: LoadError | null;
  loading: boolean;
  /** Boxes highlighted on the canvas while an action's impact preview is open (transient UI state). */
  impactPreview: ImpactPreview | null;

  open: (runId: UUID) => Promise<void>;
  close: () => void;
  receive: (e: EventEnvelope) => void;
  scrubTo: (seq: number | null) => void;
  select: (id: UUID | null) => void;
  setImpactPreview: (p: ImpactPreview) => void;
  /** Clears the preview, but only if `owner` still owns it (or always when omitted). */
  clearImpactPreview: (owner?: string) => void;
}

let sub: Subscription | null = null;
let openToken = 0;

function mergeLog(log: PersistedEvent[], incoming: readonly EventEnvelope[]): PersistedEvent[] {
  const fresh = incoming.filter((e): e is PersistedEvent => e.seq !== null);
  if (!fresh.length) return log;
  const last = log.at(-1)?.seq ?? 0;
  // fast path: strictly appending
  if (fresh.every((e, i) => e.seq > (i === 0 ? last : fresh[i - 1]!.seq))) return [...log, ...fresh];
  const bySeq = new Map<number, PersistedEvent>();
  for (const e of log) bySeq.set(e.seq, e);
  for (const e of fresh) bySeq.set(e.seq, e);
  return [...bySeq.values()].sort((a, b) => a.seq - b.seq);
}

function toLoadError(err: unknown): LoadError {
  if (err instanceof BackendUnreachableError) return { kind: "unreachable", message: err.message };
  if (err instanceof ApiError && err.status === 404) return { kind: "not_found", message: "This run doesn't exist." };
  return { kind: "other", message: err instanceof Error ? err.message : String(err) };
}

export const useRunStore = create<RunStore>((set, get) => ({
  runId: null,
  backend: null,
  live: null,
  log: [],
  scrubSeq: null,
  historical: null,
  selectedNodeId: null,
  connection: "closed",
  loadError: null,
  loading: false,
  impactPreview: null,

  open: async (runId) => {
    const token = ++openToken;
    sub?.close();
    sub = null;
    set({
      runId,
      live: null,
      log: [],
      scrubSeq: null,
      historical: null,
      selectedNodeId: null,
      loadError: null,
      loading: true,
      impactPreview: null,
      connection: "connecting",
    });
    try {
      const backend = await getBackend();
      const snap = await backend.getRun(runId);
      if (token !== openToken) return;
      set({ backend, live: stateFromSnapshot(snap), loading: false });
      sub = backend.subscribe(runId, snap.last_seq, {
        onEvent: (e) => get().receive(e),
        onState: (connection) => {
          if (token === openToken) set({ connection });
        },
      });
      // Full persisted history for the scrubber (snapshot covers state; this covers time).
      backend
        .getEvents(runId, 0)
        .then((events) => {
          if (token === openToken) set((s) => ({ log: mergeLog(s.log, events) }));
        })
        .catch(() => {
          /* scrubber degrades to events seen live */
        });
    } catch (err) {
      if (token !== openToken) return;
      set({ loading: false, loadError: toLoadError(err), connection: "closed" });
    }
  },

  close: () => {
    openToken++;
    sub?.close();
    sub = null;
    set({
      runId: null,
      live: null,
      log: [],
      scrubSeq: null,
      historical: null,
      selectedNodeId: null,
      impactPreview: null,
    });
  },

  receive: (e) =>
    set((s) => {
      if (!s.live) return s;
      const live = applyEvent(s.live, e);
      const log = e.seq !== null ? mergeLog(s.log, [e]) : s.log;
      if (live === s.live && log === s.log) return s;
      return { live, log };
    }),

  scrubTo: (seq) =>
    set((s) => {
      const max = s.log.at(-1)?.seq ?? 0;
      if (seq === null || seq >= max) return { scrubSeq: null, historical: null };
      const base = { run: s.live?.run ?? null, documents: Object.values(s.live?.documents ?? {}) };
      // Incremental: continue folding forward from the cached historical state when possible.
      let start = emptyState(base);
      if (s.historical && s.scrubSeq !== null && seq >= s.scrubSeq) start = s.historical;
      let state = start;
      for (const e of s.log) {
        if (e.seq <= state.lastSeq) continue;
        if (e.seq > seq) break;
        state = applyEvent(state, e);
      }
      return { scrubSeq: seq, historical: state };
    }),

  select: (id) => set({ selectedNodeId: id }),
  setImpactPreview: (p) => set({ impactPreview: p }),
  clearImpactPreview: (owner) =>
    set((s) => (s.impactPreview && (owner === undefined || s.impactPreview.owner === owner) ? { impactPreview: null } : s)),
}));

/** The state the UI should render: live, or the folded historical state while scrubbing. */
export function useView<T>(selector: (s: RunState) => T): T {
  return useRunStore((st) => {
    const view = st.scrubSeq === null ? st.live : st.historical;
    return selector(view ?? EMPTY);
  });
}

const EMPTY = emptyState();

export const useReadOnly = () => useRunStore((s) => s.scrubSeq !== null);
