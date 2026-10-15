/**
 * Dev-only mock backend (NEXT_PUBLIC_MOCK=1).
 *
 * Simulates the LiveGraph engine entirely in the browser: a planner that builds the
 * Living Brief graph node by node, nodes that stream token deltas, a critic that grows
 * the graph mid-run, a judge that moves claim confidences, a human-input gate, and the
 * steer / edit / rerun / restore / veto / trust actions with stale-and-recompute.
 * Agent nodes make scripted tool calls (tool.call → tool.result, nested sub-agent calls and
 * agent.message exchanges for the judge's rebuttals) so the Activity UI can be demoed offline.
 * Critics and the judge also use the cross-agent context tools (read_agent_output, list_agent_runs,
 * search_context, read_agent_trace; results get a `ctx_…` ref served by getContext), and the judge
 * makes one request_upstream_work per run (applied as an agent-sourced steer after it finishes; a
 * repeat on its rerun is rejected). A user Rerun reproduces the same output, so its stale descendants
 * come back with `reason: "upstream unchanged"` (early cutoff). Impact previews and PATCH are simulated.
 *
 * It emits contract-shaped events and keeps its own state with the same `applyEvent`
 * fold the UI uses, so the mock can never drift from the reducer's semantics.
 * Nothing in here is imported unless mock mode is on (see lib/backend.ts).
 */
import { ApiError } from "./api";
import type { Backend } from "./backend";
import type {
  Claim,
  ContextItem,
  CreatedBy,
  Document,
  Edge,
  EventEnvelope,
  EventPayloads,
  EventType,
  ImpactAction,
  ImpactOut,
  Node,
  NodePatch,
  NodeType,
  NodeVersion,
  RankedChunk,
  RunSnapshot,
  RunSummary,
  ToolChunk,
  TraceStep,
  TraceRead,
  Trust,
  UUID,
  VersionReason,
} from "./protocol";
import { applyEvent, currentVersion, emptyState, parentsOf, type RunState } from "./reducer";
import type { ConnectionState, SubscribeHandlers } from "./socket";

const SPEED = 1; // >1 = faster
const CONCURRENCY = 4;
const DEFAULT_GOAL = "Will solid-state batteries reach mass-market EVs before 2030?";

const uid = () => crypto.randomUUID();
const now = () => new Date().toISOString();
const wait = (ms: number) => new Promise<void>((r) => setTimeout(r, ms / SPEED));
const rand = (a: number, b: number) => a + Math.random() * (b - a);
const pick = <T,>(xs: readonly T[]): T => xs[Math.floor(Math.random() * xs.length)] as T;

class Cancelled extends Error {}

/** Soft per-tool budgets per agent, mirroring the backend's defaults (shown as "step/budget"). */
const BUDGETS: Partial<Record<string, Partial<Record<string, number>>>> = {
  plan: { list_documents: 1, search_documents: 2 },
  research: { web_search: 2, search_documents: 5, read_source: 6 },
  critic: { request_follow_up_research: 1, read_agent_output: 4 },
  prosecutor: { web_search: 1, search_documents: 4, read_source: 3 },
  defender: { web_search: 1, search_documents: 4, read_source: 3 },
  judge: { read_source: 6, ask_rebuttal: 2, list_agent_runs: 2, search_context: 3, read_agent_trace: 3 },
};

/** Tool results the backend doesn't put in the context store (no `ref`). */
const UNSTORED = new Set([
  "add_node",
  "add_edge",
  "finish_plan",
  "list_agent_runs",
  "recall",
  "request_follow_up_research",
  "request_upstream_work",
]);

const ctxRef = () => `ctx_${Math.floor(Math.random() * 0xffffffff).toString(16).padStart(8, "0")}`;

interface ToolOutcome {
  summary: string;
  ok?: boolean;
  error?: string | null;
  chunks?: RankedChunk[];
  node_ids?: UUID[];
  /** full result text stored in the context store (defaults to summary + chunk texts) */
  text?: string;
  /** upstream agents a context tool read */
  reads?: TraceRead[];
}

const toToolChunk = (c: RankedChunk): ToolChunk => ({
  chunk_id: c.chunk_id,
  document_id: c.document_id,
  title: c.document_title,
  score: c.score,
  trust: c.trust,
  snippet: c.text.slice(0, 240),
});

const short = (id: string) => id.slice(0, 8);

type Listener = (e: EventEnvelope) => void;

interface Task {
  cancelled: boolean;
}

const SOURCES: { title: string; url: string }[] = [
  { title: "IEA Global EV Outlook 2026", url: "https://www.iea.org/reports/global-ev-outlook-2026" },
  { title: "Toyota investor briefing: next-gen batteries", url: "https://global.toyota/en/ir/" },
  { title: "QuantumScape Q2 shareholder letter", url: "https://ir.quantumscape.com/" },
  { title: "Nature Energy: sulfide electrolyte stability review", url: "https://www.nature.com/nenergy/" },
  { title: "BloombergNEF battery price survey", url: "https://about.bnef.com/" },
  { title: "Samsung SDI pilot line announcement", url: "https://www.samsungsdi.com/" },
  { title: "Reuters: automakers hedge on solid-state timelines", url: "https://www.reuters.com/" },
  { title: "Argonne National Lab manufacturing cost model", url: "https://www.anl.gov/" },
];

const SNIPPETS = [
  "Pilot lines are producing A-sample cells at low volume, with yields reported well below the 90% needed for automotive cost targets.",
  "Sulfide electrolytes show high ionic conductivity but remain sensitive to moisture, raising dry-room manufacturing costs.",
  "Several OEMs publicly target limited solid-state production in 2027–2028, initially for premium or hybrid models.",
  "Cell-level energy density gains of 30–50% are demonstrated in lab settings; pack-level gains are smaller once stack pressure hardware is included.",
  "Lithium-ion pack prices fell again this year, raising the bar solid-state cells must clear to win mainstream platforms.",
  "Dendrite formation at high charge rates is reduced but not eliminated; cycle-life data beyond 1,000 cycles is still sparse.",
  "Analysts expect solid-state to remain under 5% of EV battery demand through 2030 in base-case scenarios.",
  "Semi-solid and hybrid electrolyte designs are shipping now and are often marketed as 'solid-state', muddying public timelines.",
];

const CLAIMS = [
  "At least one major automaker will sell a solid-state EV in volume (>10k units/yr) before 2030.",
  "Manufacturing yield, not chemistry, is the binding constraint on solid-state timelines.",
  "Falling lithium-ion costs will delay solid-state adoption in mass-market segments.",
  "Semi-solid cells will be marketed as solid-state and capture the early headlines.",
  "Solid-state will exceed 10% of EV battery demand by 2030.",
];

class MockRun {
  readonly id: UUID;
  state: RunState;
  log: EventEnvelope[] = [];
  private seq = 0;
  private listeners = new Set<Listener>();
  private tasks = new Map<UUID, Task>();
  private schedulerTimer: ReturnType<typeof setInterval> | null = null;
  private critiquedOnce = false;
  private claimIds: UUID[] = [];
  private inputTimers = new Map<UUID, ReturnType<typeof setTimeout>>();
  /** critic id → follow-up research node it requested (edge to claims added after the critic finishes) */
  private followUps = new Map<UUID, UUID>();
  private editorRetried = false;
  /** context store: ref → stored tool result */
  readonly contexts = new Map<string, ContextItem>();
  /** request_upstream_work: requester+target pairs already asked this run */
  private upstreamAsked = new Set<string>();
  /** requester node id → accepted requests, applied after the requester's version persists */
  private upstreamPending = new Map<UUID, { target: UUID; instruction: string }[]>();

  constructor(id: UUID, goal: string, private docs: Document[]) {
    this.id = id;
    const created = now();
    this.state = emptyState({
      run: { id, goal, workflow: "living_brief", status: "planning", created_at: created, updated_at: created },
      documents: docs,
    });
  }

  // -- event plumbing -------------------------------------------------------

  private emit<T extends EventType>(type: T, nodeId: UUID | null, payload: EventPayloads[T], persisted = true) {
    const e = {
      seq: persisted ? ++this.seq : null,
      run_id: this.id,
      type,
      node_id: nodeId,
      payload,
      ts: now(),
    } as EventEnvelope;
    this.state = applyEvent(this.state, e);
    if (persisted) this.log.push(e);
    for (const l of this.listeners) l(e);
  }

  subscribe(since: number, listener: Listener) {
    // Replay persisted events after `since` asynchronously, like the server does.
    queueMicrotask(() => {
      for (const e of this.log) if (e.seq !== null && e.seq > since) listener(e);
      this.listeners.add(listener);
    });
    return () => this.listeners.delete(listener);
  }

  snapshot(): RunSnapshot {
    const s = this.state;
    const versions: Record<UUID, NodeVersion> = {};
    const activity: Record<UUID, TraceStep[]> = {};
    const attempts: Record<UUID, string> = {};
    for (const id of s.nodeOrder) {
      const v = currentVersion(s, id);
      if (v) versions[id] = { ...v, retrievals: s.retrievals[v.id] };
      if (s.nodes[id]!.status === "running") {
        activity[id] = s.activity[id] ?? [];
        const a = s.attempts[id];
        if (a) attempts[id] = a;
      }
    }
    return {
      run: { ...s.run!, status: s.runStatus },
      nodes: s.nodeOrder.map((id) => s.nodes[id]!),
      edges: Object.values(s.edges),
      versions,
      claims: s.claimOrder.map((id) => s.claims[id]!),
      documents: Object.values(s.documents),
      last_seq: s.lastSeq,
      activity,
      attempts,
    };
  }

  summary(): RunSummary {
    return { ...this.state.run!, status: this.state.runStatus, node_count: this.state.nodeOrder.length };
  }

  // -- graph helpers --------------------------------------------------------

  private addNode(key: string, type: NodeType, title: string, createdBy: CreatedBy, config: Node["config"] = {}) {
    const node: Node = {
      id: uid(),
      run_id: this.id,
      key,
      type,
      title,
      status: "pending",
      config: { instructions: null, params: {}, ...config },
      position: null,
      created_by: createdBy,
      current_version_id: null,
      removed: false,
    };
    this.emit("node.added", node.id, { node });
    return node.id;
  }

  private addEdge(source: UUID, target: UUID) {
    const edge: Edge = { id: uid(), run_id: this.id, source_node_id: source, target_node_id: target };
    this.emit("edge.added", null, { edge });
  }

  private byKey(key: string): UUID | undefined {
    return this.state.nodeOrder.find((id) => this.state.nodes[id]!.key === key);
  }

  private children(id: UUID): UUID[] {
    return Object.values(this.state.edges)
      .filter((e) => e.source_node_id === id)
      .map((e) => e.target_node_id);
  }

  private descendants(id: UUID): UUID[] {
    const seen = new Set<UUID>();
    const stack = [...this.children(id)];
    while (stack.length) {
      const n = stack.pop()!;
      if (seen.has(n)) continue;
      seen.add(n);
      stack.push(...this.children(n));
    }
    return this.state.nodeOrder.filter((n) => seen.has(n));
  }

  private setStatus(id: UUID, status: Node["status"], reason?: string) {
    // Entering running always starts a new attempt (even a steer restart of a running node).
    if (status === "running") {
      this.emit("node.status", id, { status, reason: reason ?? null, attempt_id: uid() });
      return;
    }
    if (this.state.nodes[id]?.status === status) return;
    this.emit("node.status", id, { status, reason: reason ?? null });
  }

  private checkCancelled(task: Task) {
    if (task.cancelled || this.state.runStatus === "cancelled") throw new Cancelled();
  }

  /**
   * One scripted tool call: tool.call → (300–1200ms, cancellable) → run() → tool.result.
   * `run` receives the call id so it can nest sub-agent calls / agent.message under it.
   */
  private async tool(
    nodeId: UUID,
    task: Task,
    tool: string,
    args: Record<string, unknown>,
    argsSummary: string,
    run: (callId: string) => ToolOutcome | Promise<ToolOutcome>,
    opts: { agent?: string; parent?: string; delay?: readonly [number, number] } = {},
  ): Promise<ToolOutcome> {
    this.checkCancelled(task);
    const attempt = this.state.attempts[nodeId] ?? "";
    const agent = opts.agent ?? this.state.nodes[nodeId]!.type;
    const prior = (this.state.activity[nodeId] ?? []).filter((x) => x.tool === tool && (x.agent ?? "") === agent);
    const callId = uid();
    this.emit("tool.call", nodeId, {
      attempt_id: attempt,
      call_id: callId,
      tool,
      args,
      args_summary: argsSummary.slice(0, 300),
      agent,
      parent_call_id: opts.parent ?? null,
      step: prior.length + 1,
      budget: BUDGETS[agent]?.[tool] ?? null,
    });
    const t0 = performance.now();
    const [lo, hi] = opts.delay ?? [300, 1200];
    await wait(rand(lo, hi));
    this.checkCancelled(task);
    const out = await run(callId);
    this.checkCancelled(task);
    let ref: string | null = null;
    if (!UNSTORED.has(tool) && (out.ok ?? true)) {
      ref = ctxRef();
      const chunks = out.chunks ?? [];
      this.contexts.set(ref, {
        run_id: this.id,
        ref,
        node_id: nodeId,
        attempt_id: attempt,
        node_version_id: null,
        agent,
        parent_ref: null,
        kind: tool === "ask_rebuttal" ? "rebuttal" : "tool_result",
        tool,
        args,
        ok: out.ok ?? true,
        text:
          out.text ??
          [out.summary, ...chunks.map((c) => `<source chunk_id="${c.chunk_id}" title="${c.document_title}">\n${c.text}\n</source>`)].join("\n\n"),
        summary: `[${ref}] ${tool}(${argsSummary.slice(0, 60)}) -> ${out.summary}`,
        chunk_ids: chunks.map((c) => c.chunk_id),
        tokens_est: Math.round(((out.text ?? out.summary).length + chunks.reduce((n, c) => n + c.text.length, 0)) / 4),
        created_at: now(),
      });
    }
    this.emit("tool.result", nodeId, {
      attempt_id: attempt,
      call_id: callId,
      ok: out.ok ?? true,
      summary: out.summary.slice(0, 300),
      error: out.error ?? null,
      ms: Math.round(performance.now() - t0),
      chunks: (out.chunks ?? []).map(toToolChunk),
      node_ids: out.node_ids ?? [],
      ref,
      reads: out.reads ?? [],
    });
    return out;
  }

  private markStale(ids: UUID[], reason: string) {
    for (const id of ids) {
      const n = this.state.nodes[id];
      if (!n || n.removed || n.status === "skipped") continue;
      if (n.status === "pending") continue;
      this.cancelTask(id);
      this.setStatus(id, "stale", reason);
    }
    this.ensureRunning();
  }

  private cancelTask(id: UUID) {
    const t = this.tasks.get(id);
    if (t) {
      t.cancelled = true;
      this.tasks.delete(id);
    }
  }

  private ensureRunning() {
    const st = this.state.runStatus;
    if (st === "done" || st === "waiting_input") this.emit("run.status", null, { status: "running" });
  }

  // -- lifecycle ------------------------------------------------------------

  async start() {
    this.emit("run.status", null, { status: "planning" });
    await wait(500);
    const goal = this.state.run!.goal;
    const plan = this.addNode("plan", "plan", "Plan the investigation", "planner", { instructions: goal });
    this.setStatus(plan, "running");
    const task: Task = { cancelled: false };
    this.tasks.set(plan, task);
    try {
      await this.planGraph(plan, goal, task);
    } catch (err) {
      if (!(err instanceof Cancelled)) throw err;
    } finally {
      if (this.tasks.get(plan) === task) this.tasks.delete(plan);
    }
  }

  /** Planner agent: list_documents, then add_node / add_edge tool calls build the graph. */
  private async planGraph(plan: UUID, goal: string, task: Task) {
    const node = (key: string, type: NodeType, title: string, config: Node["config"] = {}) =>
      this.tool(
        plan,
        task,
        "add_node",
        { key, type, title },
        `${type} · ${title}`,
        () => ({ summary: `Added ${key}`, node_ids: [this.addNode(key, type, title, "planner", config)] }),
        { delay: [280, 560] },
      ).then((r) => r.node_ids![0]!);
    const edge = (a: UUID, b: UUID) => {
      const ka = this.state.nodes[a]!.key;
      const kb = this.state.nodes[b]!.key;
      return this.tool(
        plan,
        task,
        "add_edge",
        { source: ka, target: kb },
        `${ka} → ${kb}`,
        () => (this.addEdge(a, b), { summary: "Connected" }),
        { delay: [50, 120] },
      );
    };

    const subqs = [
      ["research-timelines", "OEM production timelines", "Which automakers have committed dates and volumes?"],
      ["research-manufacturing", "Manufacturing & yield", "What limits scaling solid-state cell production?"],
      ["research-costs", "Cost vs. lithium-ion", "How do projected $/kWh compare through 2030?"],
      ["research-performance", "Real-world performance", "What do independent tests show on cycle life and safety?"],
    ] as const;

    const planText = [
      `## Plan\n\nGoal: **${goal}**\n\nI'll split this into ${subqs.length} research tracks, then extract claims, `,
      "stress-test them with a prosecutor and a defender, let a judge score each claim, ",
      "ask you one framing question, and have the editor write the brief.\n\n",
      ...subqs.map(([, t, q]) => `- **${t}**: ${q}\n`),
    ];
    const stream = this.streamText(plan, planText.join(""), task, 30);
    stream.catch(() => {}); // awaited below

    await this.tool(plan, task, "list_documents", {}, "run scope", () => ({
      summary: `${this.docs.length} documents in scope (${this.docs.filter((d) => d.origin === "upload").length} uploaded)`,
    }), { delay: [250, 500] });

    const research: UUID[] = [];
    for (const [key, title, q] of subqs) {
      const id = await node(key, "research", title, { question: q });
      await edge(plan, id);
      research.push(id);
    }
    const claims = await node("claims", "claims", "Extract claims");
    for (const r of research) await edge(r, claims);
    const pros = await node("prosecutor", "prosecutor", "Prosecutor");
    const def = await node("defender", "defender", "Defender");
    await edge(claims, pros);
    await edge(claims, def);
    const judge = await node("judge", "judge", "Judge confidence");
    await edge(pros, judge);
    await edge(def, judge);
    await edge(claims, judge);
    const gate = await node("audience", "human_input", "Who is this brief for?", {
      prompt: "Who is the audience for this brief, and how long should it be?",
    });
    await edge(judge, gate);
    const editor = await node("editor", "editor", "Write the brief");
    await edge(judge, editor);
    await edge(gate, editor);
    for (const r of research) {
      const key = this.state.nodes[r]!.key.replace("research-", "critic-");
      const c = await node(key, "critic", `Critic: ${this.state.nodes[r]!.title}`);
      await edge(r, c);
    }
    await this.tool(plan, task, "finish_plan", {}, "", () => ({
      summary: `Plan ready: ${this.state.nodeOrder.length} steps`,
    }), { delay: [150, 300] });
    await stream;
    this.finishVersion(plan, "initial", { subquestions: subqs.map((s) => s[2]), rationale: "decompose" }, planText.join(""));
    this.setStatus(plan, "done");
    this.emit("run.status", null, { status: "running" });
    this.schedulerTimer = setInterval(() => this.tick(), 250);
  }

  private tick() {
    const s = this.state;
    if (s.runStatus !== "running" && s.runStatus !== "waiting_input") return;
    let running = this.tasks.size;
    let anyOpen = false;
    for (const id of s.nodeOrder) {
      const n = s.nodes[id]!;
      if (n.removed || n.status === "skipped") continue;
      if (n.status !== "done") anyOpen = true;
      if (n.status !== "pending" && n.status !== "stale") continue;
      if (this.tasks.has(id)) continue;
      const parents = parentsOf(s, id).filter((p) => !s.nodes[p]!.removed);
      if (!parents.every((p) => s.nodes[p]!.status === "done")) continue;
      if (n.type === "human_input") {
        const v = currentVersion(s, id);
        if (v) {
          // Inputs unchanged for an answered gate: reuse the answer (input_hash hit).
          this.emit("node.version", id, { version: v, reused: true });
          this.setStatus(id, "done");
        } else this.requestInput(id);
        continue;
      }
      if (running >= CONCURRENCY) break;
      running++;
      const reason: VersionReason = n.current_version_id ? "recompute" : "initial";
      void this.runNode(id, reason);
    }
    const waiting = s.nodeOrder.some((id) => s.nodes[id]!.status === "waiting_input");
    if (waiting && s.runStatus === "running" && this.tasks.size === 0) {
      this.emit("run.status", null, { status: "waiting_input" });
    }
    if (!anyOpen && this.tasks.size === 0 && s.runStatus === "running") {
      this.emit("run.status", null, { status: "done" });
    }
  }

  private requestInput(id: UUID) {
    const n = this.state.nodes[id]!;
    const prompt = String(n.config.prompt ?? "Your input is needed.");
    this.setStatus(id, "waiting_input");
    this.emit("input.requested", id, { prompt });
    // Keep unattended demos flowing.
    clearTimeout(this.inputTimers.get(id));
    this.inputTimers.set(
      id,
      setTimeout(() => {
        if (this.state.nodes[id]?.status === "waiting_input") {
          this.answer(id, "A product leadership team. Keep it under 400 words, with a clear bottom line.");
        }
      }, 20_000 / SPEED),
    );
  }

  answer(id: UUID, answer: string) {
    const n = this.state.nodes[id];
    if (!n || n.type !== "human_input") throw new ApiError("Not a human_input node", 400);
    clearTimeout(this.inputTimers.get(id));
    this.finishVersion(id, n.current_version_id ? "edit" : "initial", { answer }, answer);
    this.setStatus(id, "done");
    if (this.state.runStatus === "waiting_input") this.emit("run.status", null, { status: "running" });
  }

  // -- node execution -------------------------------------------------------

  private async streamText(id: UUID, text: string, task: Task, msPerChunk = 45) {
    const tokens = text.match(/\S+\s*|\s+/g) ?? [];
    let i = 0;
    while (i < tokens.length) {
      if (task.cancelled) throw new Cancelled();
      if (this.state.runStatus === "cancelled") throw new Cancelled();
      const n = Math.ceil(rand(1, 4));
      const chunk = tokens.slice(i, i + n).join("");
      i += n;
      this.emit("node.delta", id, { text: chunk }, false);
      await wait(rand(msPerChunk * 0.6, msPerChunk * 1.4));
    }
  }

  private finishVersion(id: UUID, reason: VersionReason, output: unknown, text: string, steer: string | null = null) {
    const prev = this.state.versions[id] ?? [];
    // The attempt's tool calls become the version's persisted trace.
    const trace: TraceStep[] =
      this.state.nodes[id]?.status === "running"
        ? (this.state.activity[id] ?? []).map((st) => ({
            call_id: st.call_id,
            tool: st.tool,
            agent: st.agent,
            parent_call_id: st.parent_call_id,
            args_summary: st.args_summary,
            ok: st.ok ?? false,
            summary: st.summary,
            chunk_ids: st.chunk_ids,
            node_ids: st.node_ids,
            ms: st.ms,
            ts: st.ts,
            exchange: st.exchange.map(({ from_agent, to_agent, kind, text }) => ({ from_agent, to_agent, kind, text })),
            ref: st.ref ?? null,
            reads: st.reads ?? [],
          }))
        : [];
    const v: NodeVersion = {
      id: uid(),
      node_id: id,
      version: (prev.at(-1)?.version ?? 0) + 1,
      input_hash: Math.random().toString(16).slice(2, 14),
      output,
      output_text: text,
      reason,
      steer_text: steer,
      model: ["plan", "judge", "editor"].includes(this.state.nodes[id]!.type) ? "openai/gpt-4.1" : "deepseek/deepseek-v4.1-flash",
      tokens_in: Math.round(rand(800, 4000)),
      tokens_out: Math.round(text.length / 4),
      created_at: now(),
      trace,
    };
    this.emit("node.version", id, { version: v, reused: false });
    return v;
  }

  private async runNode(id: UUID, reason: VersionReason, steerText: string | null = null) {
    this.cancelTask(id);
    const task: Task = { cancelled: false };
    this.tasks.set(id, task);
    const node = this.state.nodes[id]!;
    // consume pending steers
    const pending = (this.state.steers[id] ?? []).filter((x) => !x.consumed);
    for (const p of pending) this.emit("steer.consumed", id, { steer_id: p.steer_id, text: p.text, source: p.source });
    const steer = steerText ?? pending.map((p) => p.text).join("\n") ?? null;
    this.setStatus(id, "running", reason === "initial" ? undefined : reason);
    try {
      await wait(rand(200, 600));
      const before = currentVersion(this.state, id);
      const produced = await this.produce(node, task, steer || null);
      if (task.cancelled) throw new Cancelled();
      // A plain Rerun reproduces the same content (same output_hash): early cutoff downstream.
      const identical = reason === "retry" && !steer && !!before;
      const { text, output } = identical ? { text: before.output_text, output: before.output } : produced;
      const chunks = identical ? this.state.retrievals[before.id] ?? produced.chunks : produced.chunks;
      const v = this.finishVersion(id, steer && reason !== "retry" ? "steer" : reason, output, text, steer || null);
      if (chunks) this.emit("retrieval.done", id, { node_version_id: v.id, chunks });
      for (const item of this.contexts.values()) {
        if (item.node_id === id && item.attempt_id === this.state.attempts[id]) item.node_version_id = v.id;
      }
      this.setStatus(id, "done");
      if (identical) this.earlyCutoff(id);
      if (node.type === "critic") this.attachFollowUp(id);
      this.applyUpstreamRequests(id);
    } catch (err) {
      if (!(err instanceof Cancelled)) {
        this.emit("error", id, { message: String(err), node_id: id });
        this.setStatus(id, "failed", String(err));
      }
    } finally {
      if (this.tasks.get(id) === task) this.tasks.delete(id);
    }
  }

  /** Stale descendants whose inputs came back unchanged go back to done, reusing their version. */
  private earlyCutoff(id: UUID) {
    for (const d of this.activeDescendants(id)) {
      const n = this.state.nodes[d]!;
      if (n.removed || n.status !== "stale" || this.tasks.has(d)) continue;
      const v = currentVersion(this.state, d);
      const parentsDone = parentsOf(this.state, d).every((p) => {
        const pn = this.state.nodes[p]!;
        return pn.removed || pn.status === "done";
      });
      if (!v || !parentsDone) continue;
      this.emit("node.version", d, { version: v, reused: true });
      this.setStatus(d, "done", "upstream unchanged");
    }
  }

  /** request_upstream_work: after the requester's version persists, steer each target (source agent:<key>). */
  private applyUpstreamRequests(requester: UUID) {
    const reqs = this.upstreamPending.get(requester);
    if (!reqs?.length) return;
    this.upstreamPending.delete(requester);
    const from = this.state.nodes[requester]!.key;
    for (const { target, instruction } of reqs) {
      const t = this.state.nodes[target];
      if (!t || t.removed) continue;
      const text = `Request from ${from}: ${instruction}`;
      this.emit("steer.queued", target, { steer_id: uid(), text, source: `agent:${from}` });
      this.markStale(this.descendants(target), `upstream ${t.key} changed`);
      this.ensureRunning();
      void this.runNode(target, "steer");
    }
  }

  /** Body of the judge's request_upstream_work (mirrors the backend's acceptance rules). */
  private requestUpstream(requester: UUID, targetKey: string, instruction: string): ToolOutcome {
    const from = this.state.nodes[requester]!.key;
    const target = this.byKey(targetKey);
    const no = (error: string, summary: string): ToolOutcome => ({ ok: false, error, summary });
    if (!target) return no(`unknown node ${targetKey}; call list_agent_runs()`, "unknown node");
    const pair = `${from}->${targetKey}`;
    if (this.upstreamAsked.has(pair))
      return no(`you already asked ${targetKey} once in this run; use its current output`, "already requested");
    this.upstreamAsked.add(pair);
    const list = this.upstreamPending.get(requester) ?? [];
    list.push({ target, instruction });
    this.upstreamPending.set(requester, list);
    return { summary: `→ ${targetKey}: ${instruction}`, node_ids: [target] };
  }

  /** Transitive active ancestors (what a context tool may read). */
  private ancestors(id: UUID): UUID[] {
    const seen = new Set<UUID>();
    const stack = parentsOf(this.state, id);
    while (stack.length) {
      const n = stack.pop()!;
      if (seen.has(n) || this.state.nodes[n]?.removed) continue;
      seen.add(n);
      stack.push(...parentsOf(this.state, n));
    }
    return this.state.nodeOrder.filter((n) => seen.has(n));
  }

  private retrieve(): RankedChunk[] {
    const docs = [...this.docs].sort(() => Math.random() - 0.5).filter((d) => d.trust !== "distrusted").slice(0, 4);
    const chunks: RankedChunk[] = docs
      .map((d) => ({
        chunk_id: uid(),
        document_id: d.id,
        document_title: d.title,
        source_url: d.source_url,
        text: pick(SNIPPETS),
        score: Math.min(0.99, rand(0.62, 0.92) + (d.trust === "pinned" ? 0.08 : 0)),
        trust: d.trust,
        rank: 0,
      }))
      .sort((a, b) => b.score - a.score)
      .map((c, i) => ({ ...c, rank: i + 1 }));
    return chunks;
  }

  /** Body of the critic's request_follow_up_research tool: adds the node + critic→fu edge. */
  private growFollowUp(criticId: UUID): UUID | null {
    const pending = this.followUps.get(criticId);
    if (pending) return pending;
    if (this.critiquedOnce) return null;
    this.critiquedOnce = true;
    const id = this.addNode("research-manufacturing_fu", "research", "Electrolyte supply chain", "critic", {
      question: "Is there enough lithium-metal and sulfide electrolyte capacity to scale?",
    });
    this.addEdge(criticId, id);
    this.followUps.set(criticId, id);
    return id;
  }

  /** After the critic's version persists: wire the follow-up into claims (stale-and-recompute). */
  private attachFollowUp(criticId: UUID) {
    const id = this.followUps.get(criticId);
    if (!id) return;
    this.followUps.delete(criticId);
    const claims = this.byKey("claims");
    if (!claims) return;
    this.addEdge(id, claims);
    const st = this.state.nodes[claims]!.status;
    if (st === "running" || st === "done") this.markStale([claims, ...this.descendants(claims)], "new input from critic");
  }

  /** Scripted retrieval for a tool: `n` fresh passages from in-scope, trusted docs. */
  private take(found: RankedChunk[], n: number): RankedChunk[] {
    const cs = this.retrieve().slice(0, n);
    found.push(...cs);
    return cs;
  }

  /** Unique chunks, best first, re-ranked. */
  private ranked(found: RankedChunk[], max = 8): RankedChunk[] {
    const seen = new Set<string>();
    return found
      .filter((c) => (seen.has(c.chunk_id) ? false : (seen.add(c.chunk_id), true)))
      .sort((a, b) => b.score - a.score)
      .slice(0, max)
      .map((c, i) => ({ ...c, rank: i + 1 }));
  }

  private searchSummary(cs: RankedChunk[]): string {
    return cs.length ? `${cs.length} passage${cs.length > 1 ? "s" : ""}, top score ${cs[0]!.score.toFixed(2)}` : "No passages";
  }

  private async produce(
    node: Node,
    task: Task,
    steer: string | null,
  ): Promise<{ text: string; output: unknown; chunks?: RankedChunk[] }> {
    const goal = this.state.run!.goal;
    const steerLine = steer ? `\n\n> Steered: _${steer}_\n` : "";
    switch (node.type) {
      case "research": {
        const q = String(node.config.question ?? node.title);
        const found: RankedChunk[] = [];
        await this.tool(node.id, task, "web_search", { query: q }, q, () => {
          const cs = this.take(found, 2);
          return { summary: `Ingested ${cs.length} new page${cs.length === 1 ? "" : "s"} from ${cs.length + 3} results`, chunks: cs };
        });
        const queries = [
          `${node.title.toLowerCase()} 2027 2028 targets`,
          pick(["independent test data", "analyst forecasts", "pilot line yield", "cost per kWh projection"]),
        ];
        for (const query of queries) {
          await this.tool(node.id, task, "search_documents", { query, k: 6 }, query, () => {
            const cs = this.take(found, 3);
            return { summary: this.searchSummary(cs), chunks: cs };
          });
        }
        const best = [...found].sort((a, b) => b.score - a.score)[0];
        if (best) {
          await this.tool(
            node.id,
            task,
            "read_source",
            { chunk_id: best.chunk_id },
            `[${short(best.chunk_id)}] ${best.document_title}`,
            () => ({ summary: `Read the passage and 2 neighbours from “${best.document_title}”`, chunks: [best] }),
          );
        }
        const chunks = this.ranked(found, 6);
        const pts = chunks.slice(0, 3).map((c) => c.text);
        const text =
          `### ${node.title}\n\n_${node.config.question ?? ""}_${steerLine}\n\n` +
          chunks
            .slice(0, 3)
            .map((c) => `- ${c.text} [${c.chunk_id.slice(0, 8)}]`)
            .join("\n") +
          `\n\n**Gaps:** independent cycle-life data; ${pick(["2029 volume commitments", "pack-level cost curves", "recycling pathways"])}.`;
        await this.streamText(node.id, text, task, 40);
        return {
          text,
          chunks,
          output: { findings: pts.map((point, i) => ({ point, chunk_ids: [chunks[i]!.chunk_id] })), gaps: [] },
        };
      }
      case "critic": {
        const research = parentsOf(this.state, node.id).find((p) => this.state.nodes[p]?.type === "research");
        if (research) {
          const rk = this.state.nodes[research]!.key;
          const out = currentVersion(this.state, research)?.output_text ?? "";
          await this.tool(node.id, task, "read_agent_output", { node_key: rk, offset: 0 }, rk, () => ({
            summary: `${rk} v${currentVersion(this.state, research)?.version ?? 1}: ${out.length} chars (page 1/1)`,
            text: `<source agent="${rk}" kind="output">\n${out}\n</source>`,
            reads: [{ node_key: rk }],
          }), { delay: [250, 600] });
        }
        const thin = node.key.includes("manufacturing") && (!this.critiquedOnce || this.followUps.has(node.id));
        if (thin) {
          const question = "Is there enough lithium-metal and sulfide electrolyte capacity to scale?";
          await this.tool(node.id, task, "request_follow_up_research", { question }, question, () => {
            const id = this.growFollowUp(node.id);
            return id
              ? { summary: "Added a follow-up research step", node_ids: [id] }
              : { summary: "A follow-up was already requested this run", ok: false, error: "follow-up cap reached" };
          });
        }
        const text = thin
          ? "**Verdict: thin.** The manufacturing track ignores upstream material supply. Adding a follow-up research question on electrolyte and lithium-metal capacity."
          : `**Verdict: ok.** Coverage is adequate; sources are ${pick(["recent and diverse", "mostly primary", "consistent with each other"])}.`;
        await this.streamText(node.id, text, task, 35);
        return { text, output: { verdict: thin ? "thin" : "ok" } };
      }
      case "claims": {
        const list = CLAIMS.map((t, i) => `**C${i + 1}.** ${t}`).join("\n\n");
        const text = `### Claims under test${steerLine}\n\n${list}`;
        const streaming = this.streamText(node.id, text, task, 30);
        streaming.catch(() => {}); // awaited below; avoid an unhandled rejection if we bail early
        for (let i = 0; i < CLAIMS.length; i++) {
          await wait(450);
          if (task.cancelled) throw new Cancelled();
          const existing = this.claimIds[i] ? this.state.claims[this.claimIds[i]!] : undefined;
          const claim: Claim = existing
            ? { ...existing }
            : { id: uid(), key: `C${i + 1}`, text: CLAIMS[i]!, confidence: null, status: "active", history: [] };
          this.claimIds[i] = claim.id;
          this.emit("claim.upserted", node.id, { claim });
        }
        await streaming;
        return { text, output: { claims: CLAIMS.map((t, i) => ({ key: `C${i + 1}`, text: t, support_chunk_ids: [] })) } };
      }
      case "prosecutor":
      case "defender": {
        const attack = node.type === "prosecutor";
        const found: RankedChunk[] = [];
        // One search per claim (soft budget 4), run in parallel pairs like the real agent's read tools.
        const keys = CLAIMS.slice(0, 4).map((_, i) => `C${i + 1}`);
        for (let i = 0; i < keys.length; i += 2) {
          await Promise.all(
            keys.slice(i, i + 2).map((k) => {
              const query = `${k} ${attack ? "counter-evidence, delays, failures" : "supporting evidence, funded lines"}`;
              return this.tool(node.id, task, "search_documents", { query, k: 4 }, query, () => {
                const cs = this.take(found, Math.random() < 0.5 ? 1 : 2);
                return { summary: this.searchSummary(cs), chunks: cs };
              });
            }),
          );
        }
        const lines = CLAIMS.map(
          (_, i) =>
            `- **C${i + 1}**: ${
              attack
                ? pick(["Timelines have slipped every year since 2017.", "Pilot-line yields don't extrapolate.", "Announcements are not deliveries.", "Cost parity is assumed, not shown."])
                : pick(["Multiple OEMs have funded dedicated lines.", "Lab-to-pilot transition is already underway.", "Premium segments tolerate higher cost.", "Hybrid electrolytes de-risk the path."])
            }`,
        );
        const text = `### ${attack ? "Case against" : "Case for"}${steerLine}\n\n${lines.join("\n")}`;
        await this.streamText(node.id, text, task, 40);
        return { text, output: attack ? { attacks: [] } : { defenses: [] }, chunks: this.ranked(found) };
      }
      case "judge": {
        const found: RankedChunk[] = [];
        const upstream = this.ancestors(node.id).map((a) => this.state.nodes[a]!);
        await this.tool(node.id, task, "list_agent_runs", {}, "", () => ({
          summary: `${upstream.length} upstream agents: ${upstream.map((n) => n.key).join(", ")}`,
        }), { delay: [150, 300] });
        const researchKeys = upstream.filter((n) => n.type === "research").map((n) => n.key);
        const ctxQuery = "cycle life beyond 1,000 cycles";
        await this.tool(node.id, task, "search_context", { query: ctxQuery, k: 5 }, ctxQuery, () => {
          const hits = researchKeys.slice(0, 2);
          return {
            summary: `${hits.length} matches in ${hits.join(", ") || "upstream work"}`,
            text: hits
              .map((k) => `<source agent="${k}" kind="output">\n…independent cycle-life data beyond 1,000 cycles is still sparse…\n</source>`)
              .join("\n\n"),
            reads: hits.map((k) => ({ node_key: k })),
          };
        }, { delay: [250, 500] });
        if (upstream.some((n) => n.key === "prosecutor")) {
          await this.tool(node.id, task, "read_agent_trace", { node_key: "prosecutor" }, "prosecutor", () => ({
            summary: "prosecutor: 4 searches, 6 passages",
            text: '<source agent="prosecutor" kind="trace">\nsearch_documents ×4 → 6 passages; no web searches\n</source>',
            reads: [{ node_key: "prosecutor" }],
          }), { delay: [200, 400] });
        }
        const perf = researchKeys.find((k) => k.includes("performance")) ?? researchKeys[0];
        if (perf) {
          const instruction = "Find independent cycle-life data beyond 1,000 cycles, not manufacturer claims.";
          await this.tool(
            node.id,
            task,
            "request_upstream_work",
            { target_node_key: perf, instruction },
            `target_node_key=${perf}, instruction=${instruction}`,
            () => this.requestUpstream(node.id, perf, instruction),
            { delay: [200, 400] },
          );
        }
        const evidence = Object.values(this.state.retrievals).flat();
        const disputed = evidence.length ? pick(evidence) : null;
        if (disputed) {
          await this.tool(
            node.id,
            task,
            "read_source",
            { chunk_id: disputed.chunk_id },
            `[${short(disputed.chunk_id)}] ${disputed.document_title}`,
            () => (found.push(disputed), { summary: `Read “${disputed.document_title}” in context`, chunks: [disputed] }),
          );
        }
        const claimKey = this.state.claims[this.claimIds[0] ?? ""]?.key ?? "C1";
        const question = `On ${claimKey}: is there any delivered volume, or only announcements? Cite it.`;
        await this.tool(
          node.id,
          task,
          "ask_rebuttal",
          { claim_key: claimKey, side: "prosecutor", question },
          `${claimKey} → prosecutor: ${question}`,
          async (callId) => {
            const attempt = this.state.attempts[node.id] ?? "";
            this.emit("agent.message", node.id, {
              attempt_id: attempt,
              call_id: callId,
              from_agent: "judge",
              to_agent: "prosecutor",
              kind: "question",
              text: question,
              chunk_ids: [],
            });
            const query = `${claimKey} delivered volume vs announced`;
            const r = await this.tool(
              node.id,
              task,
              "search_documents",
              { query, k: 4 },
              query,
              () => {
                const cs = this.take(found, 2);
                return { summary: this.searchSummary(cs), chunks: cs };
              },
              { agent: "prosecutor", parent: callId },
            );
            await wait(rand(400, 800));
            this.checkCancelled(task);
            const cites = r.chunks ?? [];
            this.emit("agent.message", node.id, {
              attempt_id: attempt,
              call_id: callId,
              from_agent: "prosecutor",
              to_agent: "judge",
              kind: "answer",
              text: `Only announcements. The sources describe pilot lines and A-samples; none report deliveries above pilot scale${
                cites.length ? ` ${cites.map((c) => `[${short(c.chunk_id)}]`).join(" ")}` : ""
              }.`,
              chunk_ids: cites.map((c) => c.chunk_id),
            });
            return { summary: `Prosecutor answered citing ${cites.length} passage${cites.length === 1 ? "" : "s"}`, chunks: cites };
          },
          { delay: [200, 400] },
        );
        const header = `### Verdicts${steerLine}\n\n`;
        let text = header;
        this.emit("node.delta", node.id, { text: header }, false);
        const verdicts = [];
        for (let i = 0; i < this.claimIds.length; i++) {
          const cid = this.claimIds[i]!;
          const claim = this.state.claims[cid];
          if (!claim) continue;
          const base = [0.55, 0.78, 0.66, 0.84, 0.18][i] ?? 0.5;
          const conf = Math.max(0.03, Math.min(0.97, base + rand(-0.08, 0.08) + (steer ? -0.1 : 0)));
          const rationale = pick([
            "Defender cites concrete funded lines; prosecutor's slippage point still bites.",
            "Evidence is consistent across independent sources.",
            "Mostly forward-looking statements; little delivered volume.",
            "Cost data is strong; timeline data is weak.",
          ]);
          const line = `- **${claim.key}** → ${(conf * 100).toFixed(0)}%. ${rationale}\n`;
          await this.streamText(node.id, line, task, 35);
          text += line;
          // move the needle in two steps so the bar visibly animates
          const mid = claim.confidence ?? 0.5;
          this.emit("claim.confidence", node.id, {
            claim_id: cid,
            key: claim.key,
            confidence: (mid + conf) / 2,
            previous: claim.confidence,
            rationale: "preliminary",
          });
          await wait(500);
          if (task.cancelled) throw new Cancelled();
          this.emit("claim.confidence", node.id, {
            claim_id: cid,
            key: claim.key,
            confidence: conf,
            previous: (mid + conf) / 2,
            rationale,
          });
          const strike = conf < 0.25;
          if (strike !== (claim.status === "struck")) {
            this.emit("claim.upserted", node.id, {
              claim: { ...this.state.claims[cid]!, status: strike ? "struck" : "active" },
            });
          }
          verdicts.push({ claim_key: claim.key, confidence: conf, rationale, strike });
        }
        return { text, output: { verdicts }, chunks: found.length ? this.ranked(found) : undefined };
      }
      case "editor": {
        const claims = this.claimIds.map((id) => this.state.claims[id]).filter((c): c is Claim => !!c && c.status === "active");
        const audience = this.state.nodeOrder
          .map((id) => this.state.nodes[id]!)
          .find((n) => n.type === "human_input");
        const answer = audience ? currentVersion(this.state, audience.id)?.output_text : null;
        const hedge = (c: Claim) => ((c.confidence ?? 0) >= 0.7 ? "" : " _(moderate confidence)_");
        const text =
          `# Brief: ${goal}${steerLine}\n\n` +
          (answer ? `_For: ${answer}_\n\n` : "") +
          `**Bottom line:** expect solid-state cells in premium, low-volume EVs before 2030; mass-market adoption is unlikely until yields and costs close the gap with lithium-ion.\n\n` +
          `## What we're confident about\n\n` +
          claims.map((c) => `- ${c.text}${hedge(c)} (${Math.round((c.confidence ?? 0) * 100)}%)`).join("\n") +
          `\n\n## What would change our mind\n\n- A disclosed yield above 90% on a GWh-scale line.\n- A sub-$90/kWh cost commitment from a top-3 cell maker.\n`;
        if (!this.editorRetried) {
          // Demo a validator retry: a first draft streams, is rejected, and the buffer resets.
          this.editorRetried = true;
          await this.streamText(node.id, `# Brief: ${goal}\n\n**Bottom line:** solid-state EVs are coming (C9) `, task, 28);
          await wait(500);
          this.checkCancelled(task);
          this.emit("node.delta", node.id, { text: "", reset: true }, false);
        }
        await this.streamText(node.id, text, task, 28);
        return { text, output: { markdown: text } };
      }
      default: {
        const text = `Done.`;
        await this.streamText(node.id, text, task);
        return { text, output: {} };
      }
    }
  }

  // -- user actions ---------------------------------------------------------

  private node(id: UUID) {
    const n = this.state.nodes[id];
    if (!n) throw new ApiError("Node not found", 404);
    return n;
  }

  steer(id: UUID, text: string) {
    const n = this.node(id);
    const steerId = uid();
    this.emit("steer.queued", id, { steer_id: steerId, text, source: "user" });
    if (n.status === "running" || n.status === "done" || n.status === "failed") {
      if (n.status !== "running") this.markStale(this.descendants(id), `upstream ${n.key} changed`);
      this.ensureRunning();
      void this.runNode(id, "steer");
    }
  }

  edit(id: UUID, outputText: string) {
    const n = this.node(id);
    this.cancelTask(id);
    const v = this.finishVersion(id, "edit", { edited: true }, outputText);
    if (n.status !== "done") this.setStatus(id, "done");
    this.markStale(this.descendants(id), `upstream ${n.key} changed`);
    return v;
  }

  rerun(id: UUID) {
    const n = this.node(id);
    this.markStale(this.descendants(id), `upstream ${n.key} changed`);
    this.ensureRunning();
    void this.runNode(id, "retry");
  }

  restore(id: UUID, versionId: UUID) {
    const n = this.node(id);
    if (!this.state.versions[id]?.some((v) => v.id === versionId)) throw new ApiError("Unknown version", 404);
    this.cancelTask(id);
    this.emit("node.restored", id, { version_id: versionId });
    this.setStatus(id, "done");
    this.markStale(this.descendants(id), `upstream ${n.key} changed`);
  }

  veto(id: UUID) {
    this.node(id);
    this.cancelTask(id);
    this.emit("node.removed", id, {});
    this.setStatus(id, "skipped", "vetoed");
    for (const d of this.descendants(id)) {
      const parents = parentsOf(this.state, d).filter(
        (p) => !this.state.nodes[p]!.removed && this.state.nodes[p]!.status !== "skipped",
      );
      if (parents.length === 0) {
        this.cancelTask(d);
        this.setStatus(d, "skipped", "lost all inputs");
      } else this.markStale([d], "upstream vetoed");
    }
  }

  /** Active descendants in topological order (Kahn over the descendant subgraph, insertion order ties). */
  private activeDescendants(id: UUID): UUID[] {
    const desc = new Set(this.descendants(id).filter((d) => !this.state.nodes[d]!.removed));
    const order: UUID[] = [];
    const done = new Set<UUID>([id]);
    while (order.length < desc.size) {
      const next = this.state.nodeOrder.find(
        (d) => desc.has(d) && !done.has(d) && parentsOf(this.state, d).every((p) => !desc.has(p) || done.has(p)),
      );
      if (!next) break;
      done.add(next);
      order.push(next);
    }
    return order;
  }

  impact(id: UUID, action: ImpactAction): ImpactOut {
    const n = this.node(id);
    if (n.removed) return { action, node_ids: [], skipped_ids: [], upper_bound: true };
    const desc = this.activeDescendants(id);
    if (action === "veto") {
      const gone = new Set<UUID>([id]);
      const node_ids: UUID[] = [];
      const skipped_ids: UUID[] = [];
      for (const d of desc) {
        const live = parentsOf(this.state, d).filter((p) => !gone.has(p) && !this.state.nodes[p]!.removed);
        if (live.length) node_ids.push(d);
        else {
          gone.add(d);
          skipped_ids.push(d);
        }
      }
      return { action, node_ids, skipped_ids, upper_bound: true };
    }
    const self = action === "steer" || action === "rerun" || action === "config";
    return { action, node_ids: self ? [id, ...desc] : desc, skipped_ids: [], upper_bound: true };
  }

  patch(id: UUID, body: NodePatch): Node {
    const n = this.node(id);
    if (n.removed) throw new ApiError("node was removed", 409);
    if (body.title === undefined && body.question === undefined && body.instructions === undefined)
      throw new ApiError("nothing to change: give title, question or instructions", 422);
    if (body.question !== undefined && n.type !== "research")
      throw new ApiError("question is only valid on research nodes", 422);
    const config = { ...n.config };
    if (body.question !== undefined) config.question = body.question || null;
    if (body.instructions !== undefined) config.instructions = body.instructions || null;
    const title = body.title?.trim() || n.title;
    const same =
      title === n.title &&
      (config.question ?? null) === (n.config.question ?? null) &&
      (config.instructions ?? null) === (n.config.instructions ?? null);
    if (same) return n;
    const next: Node = { ...n, title, config };
    this.emit("node.updated", id, { node: next });
    this.cancelTask(id);
    if (n.status !== "pending") this.setStatus(id, "stale", "config changed");
    this.markStale(this.descendants(id), `upstream ${n.key} changed`);
    this.ensureRunning();
    return this.state.nodes[id]!;
  }

  setTrust(docId: UUID, trust: Trust): Document {
    const doc = this.state.documents[docId];
    if (!doc) throw new ApiError("Document not found", 404);
    const next = { ...doc, trust };
    this.docs = this.docs.map((d) => (d.id === docId ? next : d));
    this.emit("document.updated", null, { document: next });
    const touched = this.state.nodeOrder.filter((id) => {
      const v = currentVersion(this.state, id);
      return v && (this.state.retrievals[v.id] ?? []).some((c) => c.document_id === docId);
    });
    for (const t of touched) this.markStale([t, ...this.descendants(t)], `source ${trust}`);
    return next;
  }

  pause() {
    if (["running", "waiting_input", "planning"].includes(this.state.runStatus))
      this.emit("run.status", null, { status: "paused" });
  }
  resume() {
    if (this.state.runStatus === "paused") this.emit("run.status", null, { status: "running" });
  }
  cancel() {
    for (const id of [...this.tasks.keys()]) {
      this.cancelTask(id);
      this.setStatus(id, "failed", "cancelled");
    }
    this.emit("run.status", null, { status: "cancelled" });
    if (this.schedulerTimer) clearInterval(this.schedulerTimer);
  }
}

// ---------------------------------------------------------------------------

export function createMockBackend(): Backend {
  const runs = new Map<UUID, MockRun>();
  const library: Document[] = [];

  const makeDocs = (runId: UUID): Document[] =>
    SOURCES.map((s) => ({
      id: uid(),
      run_id: runId,
      title: s.title,
      source_url: s.url,
      origin: "web",
      trust: "normal",
      created_at: now(),
    }));

  const spawn = (id: UUID, goal: string) => {
    const docs = [...library.map((d) => ({ ...d })), ...makeDocs(id)];
    const r = new MockRun(id, goal, docs);
    runs.set(id, r);
    void r.start();
    return r;
  };

  // Unknown ids (e.g. after a hard reload) start a fresh demo under that id.
  const get = (id: UUID) => runs.get(id) ?? spawn(id, DEFAULT_GOAL);
  const later = <T,>(fn: () => T) => wait(rand(60, 160)).then(fn);

  return {
    mock: true,
    health: () => later(() => ({ ok: true })),
    listRuns: () => later(() => [...runs.values()].map((r) => r.summary()).reverse()),
    createRun: (body) => later(() => spawn(uid(), body.goal).snapshot()),
    getRun: (id) => later(() => get(id).snapshot()),
    getEvents: (id, since = 0) => later(() => get(id).log.filter((e) => (e.seq ?? 0) > since)),
    pauseRun: (id) => later(() => (get(id).pause(), get(id).summary())),
    resumeRun: (id) => later(() => (get(id).resume(), get(id).summary())),
    cancelRun: (id) => later(() => (get(id).cancel(), get(id).summary())),
    listVersions: (nodeId) =>
      later(() => {
        for (const r of runs.values()) {
          const vs = r.state.versions[nodeId];
          if (vs) return vs.map((v) => ({ ...v, retrievals: r.state.retrievals[v.id] }));
        }
        return [];
      }),
    steer: (nodeId, text) => later(() => withNode(nodeId, (r) => r.steer(nodeId, text))),
    edit: (nodeId, text) => later(() => withNode(nodeId, (r) => r.edit(nodeId, text))),
    rerun: (nodeId) => later(() => withNode(nodeId, (r) => r.rerun(nodeId))),
    restore: (nodeId, vid) => later(() => withNode(nodeId, (r) => r.restore(nodeId, vid))),
    answer: (nodeId, a) => later(() => withNode(nodeId, (r) => r.answer(nodeId, a))),
    veto: (nodeId) => later(() => withNode(nodeId, (r) => r.veto(nodeId))),
    impact: (nodeId, action) => later(() => withNode(nodeId, (r) => r.impact(nodeId, action))),
    patchNode: (nodeId, body) => later(() => withNode(nodeId, (r) => r.patch(nodeId, body))),
    getContext: (runId, ref) =>
      later(() => {
        const item = runs.get(runId)?.contexts.get(ref);
        if (!item) throw new ApiError("context item not found", 404);
        return item;
      }),
    uploadDocument: (file, runId) =>
      later(() => {
        const d: Document = {
          id: uid(),
          run_id: runId ?? null,
          title: file.name,
          source_url: null,
          origin: "upload",
          trust: "normal",
          created_at: now(),
        };
        library.push(d);
        return d;
      }),
    listDocuments: () => later(() => [...library]),
    setTrust: (docId, trust) =>
      later(() => {
        for (const r of runs.values()) if (r.state.documents[docId]) return r.setTrust(docId, trust);
        throw new ApiError("Document not found", 404);
      }),
    subscribe: (runId, since, handlers: SubscribeHandlers) => {
      const r = get(runId);
      const report = (s: ConnectionState) => handlers.onState?.(s, { attempt: 0 });
      report("connecting");
      const off = r.subscribe(since, handlers.onEvent);
      setTimeout(() => report("open"), 0);
      return {
        close: () => {
          off();
          report("closed");
        },
      };
    },
  };

  function withNode<T>(nodeId: UUID, fn: (r: MockRun) => T): T {
    for (const r of runs.values()) if (r.state.nodes[nodeId]) return fn(r);
    throw new ApiError("Node not found", 404);
  }
}
