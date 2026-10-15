import { describe, expect, it } from "vitest";
import type { Claim, Edge, EventEnvelope, EventPayloads, EventType, Node, NodeVersion, RunSnapshot } from "./protocol";
import { describeImpact, impactMark, toPreview } from "./impact";
import {
  applyEvent,
  currentVersion,
  emptyState,
  foldEvents,
  nodeIdByKey,
  splitSteerText,
  stateFromSnapshot,
  steerAgent,
  steerFromLabel,
  stepStatus,
  traceFor,
  upstreamRequest,
} from "./reducer";
import { useRunStore } from "./store";
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
    // events without `source` (older backends) default to the user
    expect(s.steers.a).toEqual([{ steer_id: "s1", text: "focus on cost", consumed: true, source: "user" }]);
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

describe("agent tool activity", () => {
  const chunk = (id: string) => ({ chunk_id: id, document_id: "d1", title: "Doc", score: 0.8, trust: "normal" as const, snippet: "snip" });
  const call = (callId: string, attempt: string, over: Partial<EventPayloads["tool.call"]> = {}) =>
    ev(
      "tool.call",
      {
        attempt_id: attempt,
        call_id: callId,
        tool: "web_search",
        args: { query: "q" },
        args_summary: "q",
        agent: "research",
        parent_call_id: null,
        step: 1,
        budget: 2,
        ...over,
      },
      "r",
    );
  const result = (callId: string, attempt: string, over: Partial<EventPayloads["tool.result"]> = {}) =>
    ev(
      "tool.result",
      { attempt_id: attempt, call_id: callId, ok: true, summary: "2 passages", error: null, ms: 420, chunks: [], node_ids: [], ...over },
      "r",
    );
  const start = (attempt: string) => ev("node.status", { status: "running", attempt_id: attempt }, "r");
  const base = () => {
    seq = 0;
    return [ev("node.added", { node: node("r") }, "r")];
  };

  it("starts a fresh activity list for each new running attempt", () => {
    const events = [...base(), start("A1"), call("c1", "A1"), result("c1", "A1"), ev("node.status", { status: "stale" }, "r")];
    let s = foldEvents(events);
    expect(s.activity.r).toHaveLength(1);
    s = applyEvent(s, start("A2"));
    expect(s.activity.r).toEqual([]);
    expect(s.attempts.r).toBe("A2");
  });

  it("resets activity and live text when a running node restarts with a new attempt (steer)", () => {
    let s = foldEvents([...base(), start("A1"), call("c1", "A1")]);
    s = applyEvent(s, ev("node.delta", { text: "partial" }, "r", false));
    s = applyEvent(s, start("A2"));
    expect(s.activity.r).toEqual([]);
    expect(s.live.r).toBeUndefined();
  });

  it("ignores tool events from a stale attempt or a node that is not running", () => {
    let s = foldEvents([...base(), start("A1"), call("c1", "A1"), start("A2")]);
    s = applyEvent(s, result("c1", "A1"));
    s = applyEvent(s, call("c2", "A1"));
    expect(s.activity.r).toEqual([]);
    s = applyEvent(s, ev("node.status", { status: "done" }, "r"));
    s = applyEvent(s, call("c3", "A2"));
    expect(s.activity.r).toEqual([]);
    expect(s.lastSeq).toBe(seq); // still advances the cursor
  });

  it("tool.result updates the matching step and collects provisional chunks", () => {
    const s = foldEvents([
      ...base(),
      start("A1"),
      call("c1", "A1"),
      call("c2", "A1", { tool: "search_documents", step: 1, budget: 5 }),
      result("c2", "A1", { chunks: [chunk("k1"), chunk("k2")], summary: "found 2" }),
      result("c1", "A1", { ok: false, error: "timeout", summary: "" }),
      ev("tool.result", { attempt_id: "A1", call_id: "c3", ok: true, summary: "orphan", error: null, ms: 1, chunks: [chunk("k1"), chunk("k3")], node_ids: [] }, "r"),
    ]);
    const [a, b] = s.activity.r!;
    expect(stepStatus(a!)).toBe("error");
    expect(a!.error).toBe("timeout");
    expect(stepStatus(b!)).toBe("ok");
    expect(b!.summary).toBe("found 2");
    expect(b!.chunk_ids).toEqual(["k1", "k2"]);
    expect(b!.ms).toBe(420);
    expect(s.activity.r).toHaveLength(2); // orphan result never creates a step
    expect(s.provisional.r?.map((c) => [c.chunk_id, c.rank, c.document_title])).toEqual([
      ["k1", 1, "Doc"],
      ["k2", 2, "Doc"],
    ]);
  });

  it("nests sub-agent calls and appends agent.message exchanges to the right step", () => {
    const s = foldEvents([
      ...base(),
      start("A1"),
      call("rb", "A1", { tool: "ask_rebuttal", agent: "judge", args: { side: "prosecutor" } }),
      ev("agent.message", { attempt_id: "A1", call_id: "rb", from_agent: "judge", to_agent: "prosecutor", kind: "question", text: "Why?", chunk_ids: [] }, "r"),
      call("sub", "A1", { tool: "search_documents", agent: "prosecutor", parent_call_id: "rb" }),
      result("sub", "A1", { chunks: [chunk("k9")] }),
      ev("agent.message", { attempt_id: "A1", call_id: "rb", from_agent: "prosecutor", to_agent: "judge", kind: "answer", text: "Because.", chunk_ids: ["k9"] }, "r"),
      ev("agent.message", { attempt_id: "OLD", call_id: "rb", from_agent: "x", to_agent: "y", kind: "answer", text: "stale", chunk_ids: [] }, "r"),
    ]);
    const steps = s.activity.r!;
    expect(steps.map((x) => [x.call_id, x.parent_call_id])).toEqual([
      ["rb", null],
      ["sub", "rb"],
    ]);
    expect(steps[0]!.exchange.map((m) => `${m.kind}:${m.text}`)).toEqual(["question:Why?", "answer:Because."]);
    expect(stepStatus(steps[0]!)).toBe("running");
  });

  it("replaces the live buffer on a delta reset", () => {
    let s = foldEvents([...base(), start("A1")]);
    s = applyEvent(s, ev("node.delta", { text: "draft one" }, "r", false));
    s = applyEvent(s, ev("node.delta", { text: "", reset: true }, "r", false));
    expect(s.live.r).toBe("");
    s = applyEvent(s, ev("node.delta", { text: "final" }, "r", false));
    s = applyEvent(s, ev("node.delta", { text: "Final!", reset: true }, "r", false));
    expect(s.live.r).toBe("Final!");
  });

  it("keeps activity after node.version, clears provisional, and traceFor switches to the version trace", () => {
    const trace = [
      { call_id: "c1", tool: "web_search", agent: "research", parent_call_id: null, args_summary: "q", ok: true, summary: "s", chunk_ids: ["k1"], node_ids: [], ms: 5, ts: ts(1), exchange: [] },
    ];
    const events = [
      ...base(),
      start("A1"),
      call("c1", "A1"),
      result("c1", "A1", { chunks: [chunk("k1")] }),
    ];
    const running = foldEvents(events);
    expect(traceFor(running, "r")).toBe(running.activity.r);
    events.push(ev("node.version", { version: version("v1", "r", 1, "out", { trace }), reused: false }, "r"));
    const versioned = foldEvents(events);
    expect(versioned.activity.r).toHaveLength(1);
    expect(versioned.provisional.r).toBeUndefined();
    events.push(ev("node.status", { status: "done" }, "r"));
    const done = foldEvents(events);
    expect(traceFor(done, "r")).toBe(trace);
    // a version without a trace (older backend) falls back to the last live activity
    seq = 0;
    const legacy = foldEvents([
      ev("node.added", { node: node("r") }, "r"),
      start("A1"),
      call("c1", "A1"),
      ev("node.version", { version: version("v1", "r", 1, "out"), reused: false }, "r"),
      ev("node.status", { status: "done" }, "r"),
    ]);
    expect(traceFor(legacy, "r").map((x) => x.call_id)).toEqual(["c1"]);
    expect(traceFor(emptyState(), "nope")).toHaveLength(0);
    expect(traceFor(emptyState(), "nope")).toBe(traceFor(emptyState(), "other")); // stable empty
  });

  it("failed nodes show the failed attempt's activity rather than the old version's trace", () => {
    const old = [
      { call_id: "old", tool: "web_search", agent: null, parent_call_id: null, args_summary: "", ok: true, summary: "", chunk_ids: [], node_ids: [], ms: 1, ts: ts(1), exchange: [] },
    ];
    const s = foldEvents([
      ...base(),
      ev("node.version", { version: version("v1", "r", 1, "out", { trace: old }), reused: false }, "r"),
      start("A2"),
      call("new", "A2"),
      ev("node.status", { status: "failed", reason: "boom" }, "r"),
    ]);
    expect(traceFor(s, "r").map((x) => x.call_id)).toEqual(["new"]);
  });

  it("scrubbing into the middle of a tool call shows a running step", () => {
    const events = [...base(), start("A1"), call("c1", "A1"), result("c1", "A1")];
    const mid = foldEvents(events, events[2]!.seq!);
    expect(mid.activity.r).toHaveLength(1);
    expect(stepStatus(mid.activity.r![0]!)).toBe("running");
    expect(traceFor(mid, "r")[0]?.call_id).toBe("c1");
    const after = foldEvents(events);
    expect(stepStatus(after.activity.r![0]!)).toBe("ok");
  });

  it("seeds activity and attempts from a snapshot, then folds later tool events on top", () => {
    seq = 0;
    const n = node("r", { status: "running" });
    const snap: RunSnapshot = {
      run: { id: RUN, goal: "g", workflow: "living_brief", status: "running", created_at: ts(0), updated_at: ts(0) },
      nodes: [n],
      edges: [],
      versions: {},
      claims: [],
      documents: [],
      last_seq: 10,
      activity: {
        r: [{ call_id: "c1", tool: "web_search", agent: "research", parent_call_id: null, args_summary: "q", ok: null, summary: "", chunk_ids: [], node_ids: [], ms: null, ts: ts(1), exchange: [] }],
      },
      attempts: { r: "A1" },
    };
    let s = stateFromSnapshot(snap);
    expect(s.attempts.r).toBe("A1");
    expect(traceFor(s, "r")).toHaveLength(1);
    seq = 10;
    s = applyEvent(s, result("c1", "A1"));
    s = applyEvent(s, result("c1", "STALE", { summary: "nope" }));
    expect(stepStatus(s.activity.r![0]!)).toBe("ok");
    expect(s.activity.r![0]!.summary).toBe("2 passages");
    // snapshots from older backends have no activity/attempts
    const legacy = stateFromSnapshot({ ...snap, activity: undefined, attempts: undefined });
    expect(legacy.activity).toEqual({});
    expect(legacy.attempts).toEqual({});
  });
});

describe("context traversal", () => {
  const base = () => {
    seq = 0;
    return [
      ev("node.added", { node: node("r", { key: "research-1", title: "Old title", config: { question: "old?" } }) }, "r"),
      ev("node.added", { node: node("j", { key: "judge", type: "judge" }) }, "j"),
      ev("edge.added", { edge: edge("e1", "r", "j") }),
    ];
  };

  it("node.updated replaces title and config but leaves status and versions alone", () => {
    const events = [
      ...base(),
      ev("node.version", { version: version("v1", "r", 1, "R"), reused: false }, "r"),
      ev("node.status", { status: "done" }, "r"),
      ev("node.updated", { node: node("r", { key: "research-1", title: "New title", config: { question: "new?" }, status: "pending" }) }, "r"),
    ];
    const s = foldEvents(events);
    expect(s.nodes.r?.title).toBe("New title");
    expect(s.nodes.r?.config.question).toBe("new?");
    expect(s.nodes.r?.status).toBe("done"); // statuses only change via node.status
    expect(s.nodes.r?.current_version_id).toBe("v1");
    // status changes follow as node.status events with a reason
    const t = applyEvent(s, ev("node.status", { status: "stale", reason: "config changed" }, "r"));
    expect(t.statusReason.r).toBe("config changed");
    // unknown node: ignored
    const u = applyEvent(t, ev("node.updated", { node: node("zz") }, "zz"));
    expect(u.nodes.zz).toBeUndefined();
  });

  it("tracks agent-sourced steers separately from the user's", () => {
    const events = [
      ...base(),
      ev("steer.queued", { steer_id: "s1", text: "be brief", source: "user" }, "r"),
      ev("steer.queued", { steer_id: "s2", text: "Request from judge: find cycle-life data", source: "agent:judge" }, "r"),
      ev("steer.consumed", { steer_id: "s2", text: "Request from judge: find cycle-life data", source: "agent:judge" }, "r"),
      // consumed without a prior queued event (e.g. scrubbed in late) keeps its source
      ev("steer.consumed", { steer_id: "s3", text: "Request from critic-1: x", source: "agent:critic-1" }, "r"),
    ];
    const steers = foldEvents(events).steers.r!;
    expect(steers.map((x) => [x.steer_id, x.source, x.consumed])).toEqual([
      ["s1", "user", false],
      ["s2", "agent:judge", true],
      ["s3", "agent:critic-1", true],
    ]);
    expect(steers.map((x) => steerFromLabel(x.source))).toEqual(["from you", "from judge", "from critic-1"]);
    expect(steerAgent("user")).toBeNull();
    expect(splitSteerText(steers[1]!.text)).toEqual({ agent: "judge", text: "find cycle-life data" });
    expect(splitSteerText("plain")).toEqual({ agent: null, text: "plain" });
  });

  it("keeps ref and reads on tool steps and parses upstream requests", () => {
    const start = () => ev("node.status", { status: "running", attempt_id: "A" }, "j");
    const call = (callId: string, tool: string, args: Record<string, unknown>, args_summary: string) =>
      ev("tool.call", { attempt_id: "A", call_id: callId, tool, args, args_summary, agent: "judge", parent_call_id: null, step: 1, budget: null }, "j");
    const res = (callId: string, over: Partial<EventPayloads["tool.result"]>) =>
      ev("tool.result", { attempt_id: "A", call_id: callId, ok: true, summary: "", error: null, ms: 5, chunks: [], node_ids: [], ...over }, "j");
    const events = [
      ...base(),
      start(),
      call("c1", "read_agent_output", { node_key: "research-1" }, "research-1"),
      res("c1", { summary: "research-1 v1", ref: "ctx_0011aabb", reads: [{ node_key: "research-1" }] }),
      call("c2", "request_upstream_work", { target_node_key: "research-1", instruction: "dig deeper" }, "target_node_key=research-1, instruction=dig deeper"),
      res("c2", { summary: "→ research-1: dig deeper", node_ids: ["r"] }),
      call("c3", "request_upstream_work", { target_node_key: "research-1", instruction: "again" }, "target_node_key=research-1, instruction=again"),
      res("c3", { ok: false, summary: "already requested", error: "you already asked research-1 once in this run" }),
    ];
    const s = foldEvents(events);
    const [read, accepted, rejected] = traceFor(s, "j");
    expect(read?.ref).toBe("ctx_0011aabb");
    expect(read?.reads).toEqual([{ node_key: "research-1" }]);
    expect(nodeIdByKey(s, "research-1")).toBe("r");
    expect(upstreamRequest(read!)).toBeNull();
    expect(upstreamRequest(accepted!)).toMatchObject({ target: "research-1", instruction: "dig deeper", state: "accepted", reason: null });
    expect(upstreamRequest(rejected!)).toMatchObject({ state: "rejected", reason: "you already asked research-1 once in this run" });
    // a persisted trace step has no args: parsed from args_summary / summary
    const persisted = { ...accepted!, args: undefined };
    expect(upstreamRequest(persisted)).toMatchObject({ target: "research-1", instruction: "dig deeper", state: "accepted" });
    const inFlight = foldEvents(events.slice(0, -1));
    expect(upstreamRequest(traceFor(inFlight, "j").at(-1)!)?.state).toBe("pending");
  });

  it("records reuse from node.version (early cutoff → 'upstream unchanged')", () => {
    const events = [
      ...base(),
      ev("node.version", { version: version("v1", "j", 1, "J"), reused: false }, "j"),
      ev("node.status", { status: "stale", reason: "upstream research-1 changed" }, "j"),
      ev("node.version", { version: version("v1", "j", 1, "J", { output_hash: "oh1" }), reused: true }, "j"),
      ev("node.status", { status: "done", reason: "upstream unchanged" }, "j"),
    ];
    const s = foldEvents(events);
    expect(s.reused.j).toBe(true);
    expect(s.versions.j).toHaveLength(1);
    expect(s.versions.j?.[0]?.output_hash).toBe("oh1");
    expect(s.statusReason.j).toBe("upstream unchanged");
    const t = applyEvent(s, ev("node.version", { version: version("v2", "j", 2, "J2"), reused: false }, "j"));
    expect(t.reused.j).toBe(false);
  });

  it("impact preview: summary text, canvas marks and owner-scoped clearing in the store", () => {
    const s = foldEvents(base());
    const impact = { action: "veto" as const, node_ids: ["j"], skipped_ids: ["r"], upper_bound: true };
    const d = describeImpact(s.nodes, impact);
    expect(d.headline).toBe("This will rerun 1 box: judge");
    expect(d.skippedLine).toBe("1 may be skipped: research-1");
    expect(describeImpact(s.nodes, { action: "edit", node_ids: [], skipped_ids: [], upper_bound: true }).headline).toBe(
      "Nothing downstream needs to rerun.",
    );
    expect(describeImpact(s.nodes, { action: "rerun", node_ids: ["r", "j"], skipped_ids: [], upper_bound: true }).headline).toBe(
      "This will rerun 2 boxes: research-1, judge",
    );

    const p = toPreview("owner-a", "r", impact);
    expect(impactMark(p, "j")).toBe("rerun");
    expect(impactMark(p, "r")).toBe("skipped");
    expect(impactMark(p, "other")).toBeNull();
    expect(impactMark(null, "j")).toBeNull();

    const store = useRunStore.getState();
    store.setImpactPreview(p);
    expect(useRunStore.getState().impactPreview?.owner).toBe("owner-a");
    store.clearImpactPreview("owner-b"); // a stale panel closing must not clear a newer preview
    expect(useRunStore.getState().impactPreview).toBe(p);
    store.clearImpactPreview("owner-a");
    expect(useRunStore.getState().impactPreview).toBeNull();
    store.setImpactPreview(p);
    store.close();
    expect(useRunStore.getState().impactPreview).toBeNull();
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
