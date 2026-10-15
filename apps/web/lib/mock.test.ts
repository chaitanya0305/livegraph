import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EventEnvelope, PersistedEvent } from "./protocol";
import { createMockBackend } from "./mock";
import { foldEvents, stepStatus, traceFor } from "./reducer";

describe("mock backend agent activity (smoke)", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("emits paired tool events, nested rebuttals, and version traces that fold cleanly", async () => {
    const backend = createMockBackend();
    const snap = await (async () => {
      const p = backend.createRun({ goal: "g" });
      await vi.advanceTimersByTimeAsync(500);
      return p;
    })();
    const seen: EventEnvelope[] = [];
    backend.subscribe(snap.run.id, 0, { onEvent: (e) => seen.push(e) });
    // Run long enough for research → critics → claims → debaters → judge to finish.
    for (let i = 0; i < 240 && !seen.some((e) => e.type === "input.requested"); i++) {
      await vi.advanceTimersByTimeAsync(500);
    }
    const log = seen.filter((e): e is PersistedEvent => e.seq !== null);
    const calls = log.filter((e) => e.type === "tool.call");
    const results = log.filter((e) => e.type === "tool.result");
    expect(calls.length).toBeGreaterThan(10);
    const tools = new Set(calls.map((e) => (e.type === "tool.call" ? e.payload.tool : "")));
    for (const t of ["add_node", "list_documents", "web_search", "search_documents", "read_source", "ask_rebuttal", "request_follow_up_research"]) {
      expect(tools).toContain(t);
    }
    const callIds = new Set(calls.map((e) => (e.type === "tool.call" ? e.payload.call_id : "")));
    for (const r of results) if (r.type === "tool.result") expect(callIds.has(r.payload.call_id)).toBe(true);
    expect(log.some((e) => e.type === "agent.message")).toBe(true);

    const s = foldEvents(log);
    const judge = s.nodeOrder.find((id) => s.nodes[id]!.type === "judge")!;
    expect(s.nodes[judge]!.status).toBe("done");
    const trace = traceFor(s, judge);
    const rebuttal = trace.find((x) => x.tool === "ask_rebuttal")!;
    expect(rebuttal.exchange.map((m) => m.kind)).toEqual(["question", "answer"]);
    expect(trace.some((x) => x.parent_call_id === rebuttal.call_id && x.agent === "prosecutor")).toBe(true);
    const research = s.nodeOrder.filter((id) => s.nodes[id]!.type === "research" && s.nodes[id]!.status === "done");
    for (const id of research) {
      const t = traceFor(s, id);
      expect(t.map((x) => x.tool)).toEqual(["web_search", "search_documents", "search_documents", "read_source"]);
      expect(t.every((x) => stepStatus(x) === "ok")).toBe(true);
    }
    const critic = s.nodeOrder.find((id) => traceFor(s, id).some((x) => x.tool === "request_follow_up_research"))!;
    const fu = traceFor(s, critic).find((x) => x.tool === "request_follow_up_research")!;
    expect(s.nodes[fu.node_ids[0]!]?.created_by).toBe("critic");

    // Context tools: critics read their research's output (ref + reads), served by getContext.
    const criticRead = traceFor(s, critic).find((x) => x.tool === "read_agent_output")!;
    expect(criticRead.reads?.[0]?.node_key).toMatch(/^research-/);
    expect(criticRead.ref).toMatch(/^ctx_[0-9a-f]{8}$/);
    const item = await (async () => {
      const p = backend.getContext(snap.run.id, criticRead.ref!);
      await vi.advanceTimersByTimeAsync(500);
      return p;
    })();
    expect(item.text).toContain("<source");
    // The judge asked upstream once (accepted → agent-sourced steer), and its rerun's repeat was rejected.
    const upstream = results.filter((e) => e.type === "tool.result" && e.payload.summary.startsWith("→ "));
    expect(upstream).toHaveLength(1);
    const steer = log.find((e) => e.type === "steer.queued" && e.payload.source === "agent:judge");
    expect(steer && steer.type === "steer.queued" && steer.payload.text).toMatch(/^Request from judge: /);
    const judgeTrace = traceFor(s, judge);
    expect(judgeTrace.find((x) => x.tool === "request_upstream_work")?.ok).toBe(false);
    expect(judgeTrace.some((x) => x.tool === "search_context" && x.reads?.length)).toBe(true);
  }, 30_000);

  it("previews impact, patches a research question, and cuts off an identical rerun", async () => {
    const backend = createMockBackend();
    const settle = async <T,>(p: Promise<T>) => {
      await vi.advanceTimersByTimeAsync(500);
      return p;
    };
    const snap = await settle(backend.createRun({ goal: "g" }));
    const seen: EventEnvelope[] = [];
    backend.subscribe(snap.run.id, 0, { onEvent: (e) => seen.push(e) });
    for (let i = 0; i < 240 && !seen.some((e) => e.type === "input.requested"); i++) await vi.advanceTimersByTimeAsync(500);
    const s0 = foldEvents(seen.filter((e): e is PersistedEvent => e.seq !== null));
    const byKey = (k: string) => s0.nodeOrder.find((id) => s0.nodes[id]!.key === k)!;
    const research = byKey("research-costs");

    const steerImpact = await settle(backend.impact(research, "steer"));
    expect(steerImpact.node_ids[0]).toBe(research);
    expect(steerImpact.node_ids).toContain(byKey("claims"));
    const editImpact = await settle(backend.impact(research, "edit"));
    expect(editImpact.node_ids).not.toContain(research);
    const veto = await settle(backend.impact(byKey("audience"), "veto"));
    expect(veto.node_ids).toContain(byKey("editor")); // the editor keeps the judge as an input
    expect(veto.skipped_ids).toEqual([]);

    const bad = expect(backend.patchNode(byKey("claims"), { question: "x" })).rejects.toMatchObject({ status: 422 });
    await vi.advanceTimersByTimeAsync(500);
    await bad;
    const n = await settle(backend.patchNode(research, { question: "What do cells cost in 2029?" }));
    expect(n.config.question).toBe("What do cells cost in 2029?");
    const updated = seen.filter((e) => e.type === "node.updated");
    expect(updated).toHaveLength(1);
    expect(seen.some((e) => e.type === "node.status" && e.node_id === research && e.payload.reason === "config changed")).toBe(true);
    await settle(backend.patchNode(research, { question: "What do cells cost in 2029?" })); // no-op emits nothing
    expect(seen.filter((e) => e.type === "node.updated")).toHaveLength(1);

    // Let it settle, then an identical Rerun: descendants come back "upstream unchanged".
    for (let i = 0; i < 240; i++) {
      await vi.advanceTimersByTimeAsync(500);
      const s = foldEvents(seen.filter((e): e is PersistedEvent => e.seq !== null));
      if (s.nodes[byKey("claims")]!.status === "done" && s.nodes[research]!.status === "done") break;
    }
    const mark = seen.length;
    await settle(backend.rerun(research));
    for (let i = 0; i < 40 && !seen.slice(mark).some((e) => e.type === "node.status" && e.payload.reason === "upstream unchanged"); i++) {
      await vi.advanceTimersByTimeAsync(500);
    }
    const cut = seen.slice(mark).filter((e) => e.type === "node.status" && e.payload.reason === "upstream unchanged");
    expect(cut.map((e) => e.node_id)).toContain(byKey("claims"));
  }, 30_000);
});
