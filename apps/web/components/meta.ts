import {
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
import type { EventType, NodeStatus, NodeType, RunStatus, VersionReason } from "@/lib/protocol";

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
