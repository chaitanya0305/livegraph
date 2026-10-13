import { Graph, layout } from "@dagrejs/dagre";
import type { Edge, Position, UUID } from "@/lib/protocol";

export const NODE_W = 248;
export const NODE_H = 116;

/**
 * Left-to-right dagre layout. Positions from the backend are null by contract, so the
 * frontend owns placement; nodes with an explicit position keep it.
 */
export function layoutGraph(
  nodeIds: readonly UUID[],
  edges: readonly Pick<Edge, "source_node_id" | "target_node_id">[],
  fixed: Record<UUID, Position | null> = {},
): Record<UUID, Position> {
  const g = new Graph();
  g.setGraph({ rankdir: "LR", nodesep: 26, ranksep: 64, marginx: 20, marginy: 20 });
  g.setDefaultEdgeLabel(() => ({}));
  const known = new Set(nodeIds);
  for (const id of nodeIds) g.setNode(id, { width: NODE_W, height: NODE_H });
  for (const e of edges) {
    if (known.has(e.source_node_id) && known.has(e.target_node_id)) g.setEdge(e.source_node_id, e.target_node_id);
  }
  layout(g);
  const out: Record<UUID, Position> = {};
  for (const id of nodeIds) {
    const f = fixed[id];
    if (f) {
      out[id] = f;
      continue;
    }
    const n = g.node(id);
    out[id] = { x: Math.round((n?.x ?? 0) - NODE_W / 2), y: Math.round((n?.y ?? 0) - NODE_H / 2) };
  }
  return out;
}
