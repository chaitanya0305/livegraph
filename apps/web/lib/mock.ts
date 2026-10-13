/**
 * Dev-only mock backend (NEXT_PUBLIC_MOCK=1).
 *
 * Simulates the LiveGraph engine entirely in the browser: a planner that builds the
 * Living Brief graph node by node, nodes that stream token deltas, a critic that grows
 * the graph mid-run, a judge that moves claim confidences, a human-input gate, and the
 * steer / edit / rerun / restore / veto / trust actions with stale-and-recompute.
 *
 * It emits contract-shaped events and keeps its own state with the same `applyEvent`
 * fold the UI uses, so the mock can never drift from the reducer's semantics.
 * Nothing in here is imported unless mock mode is on (see lib/backend.ts).
 */
import { ApiError } from "./api";
import type { Backend } from "./backend";
import type {
  Claim,
  CreatedBy,
  Document,
  Edge,
  EventEnvelope,
  EventPayloads,
  EventType,
  Node,
  NodeType,
  NodeVersion,
  RankedChunk,
  RunSnapshot,
  RunSummary,
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
    for (const id of s.nodeOrder) {
      const v = currentVersion(s, id);
      if (v) versions[id] = { ...v, retrievals: s.retrievals[v.id] };
    }
    return {
      run: { ...s.run!, status: s.runStatus },
      nodes: s.nodeOrder.map((id) => s.nodes[id]!),
      edges: Object.values(s.edges),
      versions,
      claims: s.claimOrder.map((id) => s.claims[id]!),
      documents: Object.values(s.documents),
      last_seq: s.lastSeq,
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
    if (this.state.nodes[id]?.status === status) return;
    this.emit("node.status", id, { status, reason: reason ?? null });
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
    const stream = this.streamText(plan, planText.join(""), { cancelled: false }, 30);

    const research: UUID[] = [];
    for (const [key, title, q] of subqs) {
      await wait(rand(450, 700));
      const id = this.addNode(key, "research", title, "planner", { question: q });
      this.addEdge(plan, id);
      research.push(id);
    }
    await wait(500);
    const claims = this.addNode("claims", "claims", "Extract claims", "planner");
    for (const r of research) this.addEdge(r, claims);
    await wait(450);
    const pros = this.addNode("prosecutor", "prosecutor", "Prosecutor", "planner");
    const def = this.addNode("defender", "defender", "Defender", "planner");
    this.addEdge(claims, pros);
    this.addEdge(claims, def);
    await wait(450);
    const judge = this.addNode("judge", "judge", "Judge confidence", "planner");
    this.addEdge(pros, judge);
    this.addEdge(def, judge);
    this.addEdge(claims, judge);
    await wait(350);
    const gate = this.addNode("audience", "human_input", "Who is this brief for?", "planner", {
      prompt: "Who is the audience for this brief, and how long should it be?",
    });
    this.addEdge(judge, gate);
    await wait(350);
    const editor = this.addNode("editor", "editor", "Write the brief", "planner");
    this.addEdge(judge, editor);
    this.addEdge(gate, editor);
    for (const r of research) {
      await wait(250);
      const key = this.state.nodes[r]!.key.replace("research-", "critic-");
      const c = this.addNode(key, "critic", `Critic: ${this.state.nodes[r]!.title}`, "planner");
      this.addEdge(r, c);
    }
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
    for (const p of pending) this.emit("steer.consumed", id, { steer_id: p.steer_id, text: p.text });
    const steer = steerText ?? pending.map((p) => p.text).join("\n") ?? null;
    this.setStatus(id, "running", reason === "initial" ? undefined : reason);
    try {
      await wait(rand(200, 600));
      const { text, output, chunks } = await this.produce(node, task, steer || null);
      if (task.cancelled) throw new Cancelled();
      const v = this.finishVersion(id, steer && reason !== "retry" ? "steer" : reason, output, text, steer || null);
      if (chunks) this.emit("retrieval.done", id, { node_version_id: v.id, chunks });
      this.setStatus(id, "done");
      if (node.type === "critic") this.maybeGrow(id);
    } catch (err) {
      if (!(err instanceof Cancelled)) {
        this.emit("error", id, { message: String(err), node_id: id });
        this.setStatus(id, "failed", String(err));
      }
    } finally {
      if (this.tasks.get(id) === task) this.tasks.delete(id);
    }
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

  private maybeGrow(criticId: UUID) {
    if (this.critiquedOnce) return;
    const critic = this.state.nodes[criticId]!;
    if (!critic.key.includes("manufacturing")) return;
    this.critiquedOnce = true;
    const claims = this.byKey("claims");
    if (!claims) return;
    const id = this.addNode("research-manufacturing_fu", "research", "Electrolyte supply chain", "critic", {
      question: "Is there enough lithium-metal and sulfide electrolyte capacity to scale?",
    });
    this.addEdge(criticId, id);
    this.addEdge(id, claims);
    const st = this.state.nodes[claims]!.status;
    if (st === "running" || st === "done") this.markStale([claims, ...this.descendants(claims)], "new input from critic");
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
        const chunks = this.retrieve();
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
        const thin = node.key.includes("manufacturing") && !this.critiquedOnce;
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
        return { text, output: attack ? { attacks: [] } : { defenses: [] } };
      }
      case "judge": {
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
        return { text, output: { verdicts } };
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
    this.emit("steer.queued", id, { steer_id: steerId, text });
    if (n.status === "running" || n.status === "done" || n.status === "failed") {
      if (n.status !== "running") this.markStale(this.descendants(id), "upstream steered");
      this.ensureRunning();
      void this.runNode(id, "steer");
    }
  }

  edit(id: UUID, outputText: string) {
    const n = this.node(id);
    this.cancelTask(id);
    const v = this.finishVersion(id, "edit", { edited: true }, outputText);
    if (n.status !== "done") this.setStatus(id, "done");
    this.markStale(this.descendants(id), "upstream edited");
    return v;
  }

  rerun(id: UUID) {
    this.node(id);
    this.markStale(this.descendants(id), "upstream rerun");
    this.ensureRunning();
    void this.runNode(id, "retry");
  }

  restore(id: UUID, versionId: UUID) {
    this.node(id);
    if (!this.state.versions[id]?.some((v) => v.id === versionId)) throw new ApiError("Unknown version", 404);
    this.cancelTask(id);
    this.emit("node.restored", id, { version_id: versionId });
    this.setStatus(id, "done");
    this.markStale(this.descendants(id), "upstream restored");
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
