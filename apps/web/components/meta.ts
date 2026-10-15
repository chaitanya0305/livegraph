import {
  BookOpen,
  CircleCheck,
  FileSearch,
  FileText,
  Files,
  GitBranchPlus,
  Globe,
  Link,
  MessagesSquare,
  Network,
  Plus,
  Route,
  Send,
  TextSearch,
  Undo2,
  Spline,
  Wrench,
  FileQuestion,
  Gavel,
  ListChecks,
  ListTree,
  type LucideIcon,
  MessageCircleQuestion,
  PenLine,
  ScanSearch,
  Search,
  Shield,
  Swords,
} from "lucide-react";
import type { ActivityStep } from "@/lib/reducer";
import type { EventType, NodeStatus, NodeType, RunStatus, ToolName, TraceStep, VersionReason } from "@/lib/protocol";

export const NODE_TYPE: Record<NodeType, { icon: LucideIcon; label: string }> = {
  plan: { icon: ListTree, label: "Planner" },
  research: { icon: Search, label: "Research" },
  claims: { icon: ListChecks, label: "Claims" },
  prosecutor: { icon: Swords, label: "Prosecutor" },
  defender: { icon: Shield, label: "Defender" },
  judge: { icon: Gavel, label: "Judge" },
  human_input: { icon: MessageCircleQuestion, label: "Your input" },
  editor: { icon: PenLine, label: "Editor" },
  critic: { icon: ScanSearch, label: "Critic" },
};

export const nodeTypeMeta = (t: string) => NODE_TYPE[t as NodeType] ?? { icon: FileQuestion, label: t };

export interface ToolMeta {
  icon: LucideIcon;
  /** present-tense verb phrase; `{agent}` is replaced with the target agent */
  verb: string;
  /** noun used in one-line trace summaries ("3 searches · 1 rebuttal"); null = not counted */
  noun: [singular: string, plural: string] | null;
}

export const TOOL_META: Record<ToolName, ToolMeta> = {
  web_search: { icon: Globe, verb: "Searching the web", noun: ["web search", "web searches"] },
  search_documents: { icon: FileSearch, verb: "Searching documents", noun: ["search", "searches"] },
  read_source: { icon: BookOpen, verb: "Reading source", noun: ["read", "reads"] },
  fetch_page: { icon: Link, verb: "Fetching page", noun: ["page fetch", "page fetches"] },
  ask_rebuttal: { icon: MessagesSquare, verb: "Asking {agent}", noun: ["rebuttal", "rebuttals"] },
  request_follow_up_research: { icon: GitBranchPlus, verb: "Requesting follow-up", noun: ["follow-up", "follow-ups"] },
  add_node: { icon: Plus, verb: "Adding step", noun: ["step added", "steps added"] },
  add_edge: { icon: Spline, verb: "Connecting steps", noun: null },
  finish_plan: { icon: CircleCheck, verb: "Finishing plan", noun: null },
  list_documents: { icon: Files, verb: "Listing documents", noun: ["listing", "listings"] },
  list_agent_runs: { icon: Network, verb: "Listing upstream agents", noun: null },
  read_agent_output: { icon: FileText, verb: "Reading upstream output", noun: ["upstream read", "upstream reads"] },
  read_agent_trace: { icon: Route, verb: "Reading upstream trace", noun: ["trace read", "trace reads"] },
  recall: { icon: Undo2, verb: "Recalling earlier result", noun: ["recall", "recalls"] },
  search_context: { icon: TextSearch, verb: "Searching shared context", noun: ["context search", "context searches"] },
  request_upstream_work: { icon: Send, verb: "Asking upstream to redo", noun: ["upstream request", "upstream requests"] },
};

/** Tools that read other agents' work (their steps carry `reads`). */
export const CONTEXT_TOOLS = new Set<string>([
  "list_agent_runs",
  "read_agent_output",
  "read_agent_trace",
  "recall",
  "search_context",
]);

export function toolMeta(tool: string): ToolMeta {
  return TOOL_META[tool as ToolName] ?? { icon: Wrench, verb: tool.replace(/_/g, " "), noun: ["tool call", "tool calls"] };
}

/** "Asking prosecutor" for ask_rebuttal; plain verb otherwise. `target` = the agent being asked. */
export function toolVerb(tool: string, target?: string | null): string {
  const v = toolMeta(tool).verb;
  return v.replace("{agent}", target ? target : "agent");
}

/** Who an ask_rebuttal call is addressed to (args.side, or the first exchange's recipient). */
export function stepTarget(step: ActivityStep): string | null {
  const side = step.args?.side;
  if (typeof side === "string") return side;
  return step.exchange[0]?.to_agent ?? null;
}

/** "Searching the web: 'solid-state yields' · 3/5" */
export function stepLine(step: ActivityStep): string {
  const verb = toolVerb(step.tool, stepTarget(step));
  const args = step.args_summary ? `: '${step.args_summary}'` : "";
  const count = step.step && step.budget ? ` · ${step.step}/${step.budget}` : "";
  return `${verb}${args}${count}`;
}

/** One-line summary of a trace, e.g. "3 searches · 1 rebuttal". Empty string for no steps. */
export function traceSummary(trace: readonly TraceStep[]): string {
  const counts = new Map<string, { n: number; noun: [string, string] }>();
  for (const st of trace) {
    const noun = toolMeta(st.tool).noun;
    if (!noun) continue;
    const key = noun[0];
    const c = counts.get(key) ?? { n: 0, noun };
    c.n++;
    counts.set(key, c);
  }
  return [...counts.values()].map((c) => `${c.n} ${c.n === 1 ? c.noun[0] : c.noun[1]}`).join(" · ");
}

export const NODE_STATUS_LABEL: Record<NodeStatus, string> = {
  pending: "Pending",
  running: "Running",
  waiting_input: "Needs you",
  done: "Done",
  stale: "Stale",
  failed: "Failed",
  skipped: "Skipped",
};

export const RUN_STATUS_LABEL: Record<RunStatus, string> = {
  planning: "Planning",
  running: "Running",
  paused: "Paused",
  waiting_input: "Waiting for you",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

/** Tailwind classes for a status chip (bg + text), keyed by any status string. */
export function statusTone(status: string): { chip: string; dot: string; live: boolean } {
  switch (status) {
    case "running":
    case "planning":
      return { chip: "bg-accent-soft text-accent-ink", dot: "bg-accent", live: true };
    case "waiting_input":
      return { chip: "bg-amber-soft text-amber-ink", dot: "bg-amber", live: true };
    case "paused":
      return { chip: "bg-amber-soft text-amber-ink", dot: "bg-amber", live: false };
    case "done":
      return { chip: "bg-green-soft text-green-ink", dot: "bg-green", live: false };
    case "failed":
      return { chip: "bg-red-soft text-red-ink", dot: "bg-red", live: false };
    case "stale":
      return { chip: "bg-sunken text-ink-2", dot: "bg-ink-4", live: false };
    default:
      return { chip: "bg-sunken text-ink-3", dot: "bg-ink-4", live: false };
  }
}

export const REASON_TONE: Record<VersionReason, string> = {
  initial: "bg-sunken text-ink-2",
  recompute: "bg-sunken text-ink-2",
  steer: "bg-accent-soft text-accent-ink",
  edit: "bg-violet-soft text-violet",
  retry: "bg-sunken text-ink-2",
  restore: "bg-green-soft text-green-ink",
};

/** Colors for event ticks on the timeline, grouped by meaning. */
export function eventColor(t: EventType): string {
  switch (t) {
    case "node.added":
    case "edge.added":
    case "node.removed":
      return "var(--ink-2)";
    case "node.status":
    case "run.status":
      return "var(--accent)";
    case "node.version":
    case "node.restored":
      return "var(--green)";
    case "claim.upserted":
    case "claim.confidence":
      return "var(--violet)";
    case "steer.queued":
    case "steer.consumed":
    case "input.requested":
      return "var(--amber)";
    case "retrieval.done":
    case "document.updated":
      return "var(--teal)";
    case "tool.call":
    case "tool.result":
    case "agent.message":
      return "var(--sky)";
    case "error":
      return "var(--red)";
    default:
      return "var(--ink-3)";
  }
}

export const EVENT_GROUPS: { label: string; color: string }[] = [
  { label: "graph", color: "var(--ink-2)" },
  { label: "status", color: "var(--accent)" },
  { label: "version", color: "var(--green)" },
  { label: "claims", color: "var(--violet)" },
  { label: "steer / input", color: "var(--amber)" },
  { label: "sources", color: "var(--teal)" },
  { label: "tools", color: "var(--sky)" },
  { label: "error", color: "var(--red)" },
];

export function confidenceColor(c: number | null): string {
  if (c === null) return "var(--ink-4)";
  if (c >= 0.7) return "var(--green)";
  if (c >= 0.4) return "var(--amber)";
  return "var(--red)";
}

export function relativeTime(iso: string, now = Date.now()): string {
  const d = (now - new Date(iso).getTime()) / 1000;
  if (!Number.isFinite(d)) return "";
  if (d < 45) return "just now";
  if (d < 3600) return `${Math.round(d / 60)}m ago`;
  if (d < 86400) return `${Math.round(d / 3600)}h ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function clockTime(iso: string | null): string {
  if (!iso) return "";
  return new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
