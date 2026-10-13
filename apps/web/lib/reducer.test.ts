import { describe, expect, it } from "vitest";
import type { Claim, Edge, EventEnvelope, EventPayloads, EventType, Node, NodeVersion, RunSnapshot } from "./protocol";
import { applyEvent, currentVersion, emptyState, foldEvents, stateFromSnapshot } from "./reducer";
import { backoffDelay } from "./socket";

const RUN = "run-1";
let seq = 0;
const ts = (n: number) => new Date(Date.UTC(2026, 0, 1, 0, 0, n)).toISOString();

function ev<T extends EventType>(type: T, payload: EventPayloads[T], nodeId: string | null = null, persisted = true) {
  const s = persisted ? ++seq : null;
  return { seq: s, run_id: RUN, type, node_id: nodeId, payload, ts: ts(s ?? 0) } as EventEnvelope;
}

const node = (id: string, over: Partial<Node> = {}): Node => ({
  id,
  run_id: RUN,
  key: id,
  type: "research",
  title: id.toUpperCase(),
  status: "pending",
  config: { question: "q?" },
  position: null,
  created_by: "planner",
  current_version_id: null,
  removed: false,
  ...over,
});

const edge = (id: string, s: string, t: string): Edge => ({ id, run_id: RUN, source_node_id: s, target_node_id: t });

const version = (id: string, nodeId: string, n: number, text: string, over: Partial<NodeVersion> = {}): NodeVersion => ({
  id,
  node_id: nodeId,
  version: n,
  input_hash: `h${n}`,
  output: { text },
  output_text: text,
  reason: n === 1 ? "initial" : "recompute",
  steer_text: null,
  model: "m",
  tokens_in: 1,
  tokens_out: 1,
  created_at: ts(n),
  ...over,
});

const claim = (id: string, over: Partial<Claim> = {}): Claim => ({
  id,
  key: "C1",
  text: "Claim text",
  confidence: null,
  status: "active",
  history: [],
  ...over,
});

function scenario(): EventEnvelope[] {
  seq = 0;
  return [
    ev("run.status", { status: "planning" }),
    ev("node.added", { node: node("a", { type: "plan" }) }, "a"),
    ev("node.added", { node: node("b") }, "b"),
    ev("edge.added", { edge: edge("e1", "a", "b") }),
    ev("node.status", { status: "running" }, "a"),
    ev("node.delta", { text: "Hel" }, "a", false),
    ev("node.delta", { text: "lo" }, "a", false),
    ev("node.version", { version: version("va1", "a", 1, "Hello"), reused: false }, "a"),
    ev("node.status", { status: "done" }, "a"),
    ev("run.status", { status: "running" }),
    ev("claim.upserted", { claim: claim("c1") }, "b"),
    ev("claim.confidence", { claim_id: "c1", key: "C1", confidence: 0.4, previous: null, rationale: "weak" }, "b"),
    ev("claim.confidence", { claim_id: "c1", key: "C1", confidence: 0.8, previous: 0.4, rationale: "strong" }, "b"),
  ];
}

describe("applyEvent", () => {
  it("builds nodes, edges and statuses from an empty state", () => {
    const s = foldEvents(scenario());
    expect(s.nodeOrder).toEqual(["a", "b"]);
    expect(Object.keys(s.edges)).toEqual(["e1"]);
    expect(s.nodes.a?.status).toBe("done");
    expect(s.nodes.b?.status).toBe("pending");
    expect(s.runStatus).toBe("running");
    expect(s.lastSeq).toBe(11);
  });

  it("appends deltas to a live buffer and clears it when node.version arrives", () => {
    const events = scenario();
    let s = emptyState();
    for (const e of events.slice(0, 7)) s = applyEvent(s, e);
    expect(s.live.a).toBe("Hello");
    s = applyEvent(s, events[7]!);
    expect(s.live.a).toBeUndefined();
    expect(s.nodes.a?.current_version_id).toBe("va1");
    expect(currentVersion(s, "a")?.output_text).toBe("Hello");
  });

  it("does not advance lastSeq for ephemeral events", () => {
    let s = emptyState();
    s = applyEvent(s, ev("node.added", { node: node("x") }, "x"));
    const before = s.lastSeq;
    s = applyEvent(s, ev("node.delta", { text: "t" }, "x", false));
    expect(s.lastSeq).toBe(before);
  });

  it("clears the live buffer when a node starts running again", () => {
    seq = 0;
    let s = foldEvents([ev("node.added", { node: node("x") }, "x"), ev("node.status", { status: "running" }, "x")]);
    s = applyEvent(s, ev("node.delta", { text: "partial" }, "x", false));
    s = applyEvent(s, ev("node.status", { status: "stale" }, "x"));
    expect(s.live.x).toBe("partial");
    s = applyEvent(s, ev("node.status", { status: "running", reason: "recompute" }, "x"));
    expect(s.live.x).toBeUndefined();
    expect(s.statusReason.x).toBe("recompute");
  });

  it("is idempotent for persisted events at or below lastSeq", () => {
    const events = scenario();
    const s = foldEvents(events);
    const again = events.reduce(applyEvent, s);
    expect(again).toBe(s);
  });

  it("does not mutate the input state", () => {
    const events = scenario();
    const s = foldEvents(events.slice(0, 4));
    const frozen = JSON.stringify(s);
    applyEvent(s, events[4]!);
    applyEvent(s, events[5]!);
    expect(JSON.stringify(s)).toBe(frozen);
  });

  it("tracks claim confidence history and the latest value", () => {
    const s = foldEvents(scenario());
    const c = s.claims.c1!;
    expect(c.confidence).toBe(0.8);
    expect(c.history.map((h) => h.confidence)).toEqual([0.4, 0.8]);
    expect(c.history[1]?.rationale).toBe("strong");
  });

  it("keeps accumulated history when a claim is re-upserted without one (strike)", () => {
    const events = scenario();
    events.push(ev("claim.upserted", { claim: claim("c1", { status: "struck", confidence: 0.8 }) }, "b"));
    const s = foldEvents(events);
    expect(s.claims.c1?.status).toBe("struck");
    expect(s.claims.c1?.history).toHaveLength(2);
    expect(s.claimOrder).toEqual(["c1"]);
  });

  it("handles versions, restore, and dedupes reused versions", () => {
    const events = scenario();
    const v2 = version("va2", "a", 2, "Hello again", { reason: "steer", steer_text: "be loud" });
    events.push(ev("node.version", { version: v2, reused: false }, "a"));
    events.push(ev("node.restored", { version_id: "va1" }, "a"));
    events.push(ev("node.version", { version: v2, reused: true }, "a"));
    const s = foldEvents(events);
    expect(s.versions.a?.map((v) => v.version)).toEqual([1, 2]);
    expect(s.nodes.a?.current_version_id).toBe("va2");
    const atRestore = foldEvents(events, events.at(-2)!.seq!);
    expect(atRestore.nodes.a?.current_version_id).toBe("va1");
    expect(currentVersion(atRestore, "a")?.output_text).toBe("Hello");
  });

  it("marks removed nodes and stores retrievals, steers, prompts and errors", () => {
    const events = scenario();
    events.push(
      ev("node.removed", {}, "b"),
      ev("retrieval.done", {
        node_version_id: "va1",
        chunks: [
          { chunk_id: "k1", document_id: "d1", document_title: "Doc", source_url: null, text: "t", score: 0.9, trust: "normal", rank: 1 },
        ],
      }),
      ev("steer.queued", { steer_id: "s1", text: "focus on cost" }, "a"),
      ev("steer.consumed", { steer_id: "s1", text: "focus on cost" }, "a"),
      ev("input.requested", { prompt: "Who is the audience?" }, "b"),
      ev("error", { message: "boom", node_id: "a" }, "a"),
      ev("document.updated", {
        document: { id: "d1", run_id: RUN, title: "Doc", source_url: null, origin: "web", trust: "distrusted", created_at: ts(1) },
      }),
    );
    const s = foldEvents(events);
    expect(s.nodes.b?.removed).toBe(true);
    expect(s.retrievals.va1?.[0]?.trust).toBe("distrusted");
    expect(s.documents.d1?.trust).toBe("distrusted");
    expect(s.steers.a).toEqual([{ steer_id: "s1", text: "focus on cost", consumed: true }]);
    expect(s.prompts.b).toBe("Who is the audience?");
    expect(s.errors[0]?.message).toBe("boom");
  });
});

describe("foldEvents (scrubber)", () => {
  it("state at seq N only reflects events with seq <= N", () => {
    const events = scenario();
    const s3 = foldEvents(events, 3);
    expect(s3.nodeOrder).toEqual(["a", "b"]);
    expect(Object.keys(s3.edges)).toHaveLength(0);
    expect(s3.runStatus).toBe("planning");
    const s5 = foldEvents(events, 5);
    expect(s5.nodes.a?.status).toBe("running");
    // deltas are ephemeral and never replayed
    expect(s5.live.a).toBeUndefined();
  });

  it("live fold from a snapshot converges with a full fold", () => {
    const events = scenario();
    const full = foldEvents(events);
    const mid = foldEvents(events, 8);
    const snap: RunSnapshot = {
      run: { id: RUN, goal: "g", workflow: "living_brief", status: mid.runStatus, created_at: ts(0), updated_at: ts(0) },
      nodes: mid.nodeOrder.map((id) => mid.nodes[id]!),
      edges: Object.values(mid.edges),
      versions: { a: currentVersion(mid, "a")! },
      claims: [],
      documents: [],
      last_seq: mid.lastSeq,
    };
    // the socket may replay events the snapshot already covers; they must be skipped
    const live = events.reduce(applyEvent, stateFromSnapshot(snap));
    expect(live.nodes).toEqual(full.nodes);
    expect(live.claims).toEqual(full.claims);
    expect(live.runStatus).toBe(full.runStatus);
    expect(live.lastSeq).toBe(full.lastSeq);
  });
});

describe("backoffDelay", () => {
  it("grows exponentially and caps", () => {
    for (let i = 0; i < 20; i++) {
      expect(backoffDelay(0)).toBeLessThanOrEqual(500);
      expect(backoffDelay(0)).toBeGreaterThanOrEqual(250);
      expect(backoffDelay(10)).toBeLessThanOrEqual(10_000);
      expect(backoffDelay(10)).toBeGreaterThanOrEqual(5_000);
    }
  });
});
