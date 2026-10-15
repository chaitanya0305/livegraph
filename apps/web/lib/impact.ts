/** Impact preview helpers (GET /nodes/{id}/impact): pure, so they're unit-tested alongside the reducer. */
import type { ImpactAction, ImpactOut, UUID } from "./protocol";

/** Transient canvas highlight while an inline confirm is open (lives in the zustand store, never in RunState). */
export interface ImpactPreview {
  /** who opened it (so a closing panel never clears a newer one) */
  owner: string;
  nodeId: UUID;
  action: ImpactAction;
  /** boxes that would rerun */
  rerun: ReadonlySet<UUID>;
  /** boxes a veto would leave without inputs */
  skipped: ReadonlySet<UUID>;
}

export function toPreview(owner: string, nodeId: UUID, impact: ImpactOut): ImpactPreview {
  return {
    owner,
    nodeId,
    action: impact.action,
    rerun: new Set(impact.node_ids ?? []),
    skipped: new Set(impact.skipped_ids ?? []),
  };
}

export type ImpactMark = "rerun" | "skipped" | null;

export function impactMark(p: ImpactPreview | null, id: UUID): ImpactMark {
  if (!p) return null;
  if (p.skipped.has(id)) return "skipped";
  if (p.rerun.has(id)) return "rerun";
  return null;
}

type KeyLookup = Readonly<Record<UUID, { key: string } | undefined>>;

const keysOf = (nodes: KeyLookup, ids: readonly UUID[]) => ids.map((id) => nodes[id]?.key ?? id.slice(0, 8));

export interface ImpactSummary {
  rerunKeys: string[];
  skippedKeys: string[];
  /** "This will rerun 3 boxes: a, b, c" */
  headline: string;
  /** "1 may be skipped: x" (veto only) */
  skippedLine: string | null;
}

const plural = (n: number) => `${n} box${n === 1 ? "" : "es"}`;

/** `nodes` = RunState.nodes (only `key` is read). */
export function describeImpact(nodes: KeyLookup, impact: ImpactOut): ImpactSummary {
  const rerunKeys = keysOf(nodes, impact.node_ids ?? []);
  const skippedKeys = keysOf(nodes, impact.skipped_ids ?? []);
  const headline = rerunKeys.length
    ? `This will rerun ${plural(rerunKeys.length)}: ${rerunKeys.join(", ")}`
    : impact.action === "edit" || impact.action === "restore" || impact.action === "veto"
      ? "Nothing downstream needs to rerun."
      : "This will rerun nothing.";
  const skippedLine = skippedKeys.length ? `${skippedKeys.length} may be skipped: ${skippedKeys.join(", ")}` : null;
  return { rerunKeys, skippedKeys, headline, skippedLine };
}
