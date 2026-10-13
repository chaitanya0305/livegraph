"use client";

import {
  applyNodeChanges,
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useStore,
  type Edge as RFEdge,
  type Node as RFNode,
  type NodeChange,
} from "@xyflow/react";
import { Crosshair } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { effectiveStatus } from "@/lib/reducer";
import { useRunStore, useView } from "@/lib/store";
import { Button } from "../ui";
import { GraphNode, type GraphNodeData } from "./GraphNode";
import { layoutGraph, NODE_H, NODE_W } from "./layout";

const MIN_READABLE_ZOOM = 0.62;

const nodeTypes = { lg: GraphNode };

type FlowNode = RFNode<GraphNodeData, "lg">;

function useStructure() {
  // A string key that only changes when the graph's shape changes (not on every delta).
  const key = useView(
    (s) =>
      s.nodeOrder.join(",") +
      "|" +
      Object.values(s.edges)
        .map((e) => `${e.source_node_id}>${e.target_node_id}`)
        .sort()
        .join(","),
  );
  const order = useView((s) => s.nodeOrder);
  const edges = useView((s) => s.edges);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => ({ key, order, edges: Object.values(edges) }), [key]);
}

function edgeVisualKey(s: ReturnType<typeof useRunStore.getState>["live"]) {
  if (!s) return "";
  return s.nodeOrder.map((id) => `${id}:${effectiveStatus(s.nodes[id]!)}`).join(",");
}

function CanvasInner() {
  const { key, order, edges } = useStructure();
  const statusKey = useView((s) => edgeVisualKey(s));
  const selectedId = useRunStore((s) => s.selectedNodeId);
  const select = useRunStore((s) => s.select);
  const scrubbing = useRunStore((s) => s.scrubSeq !== null);
  const { fitView } = useReactFlow();
  const [userMoved, setUserMoved] = useState(false);
  const [nodes, setNodes] = useState<FlowNode[]>([]);
  const fitTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const positions = useMemo(() => layoutGraph(order, edges), [key]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setNodes((prev) => {
      const byId = new Map(prev.map((n) => [n.id, n]));
      return order.map((id) => {
        const old = byId.get(id);
        const position = positions[id] ?? { x: 0, y: 0 };
        return {
          ...(old ?? {}),
          id,
          type: "lg" as const,
          position,
          data: old?.data ?? { nodeId: id },
          selected: id === selectedId,
          draggable: false,
          connectable: false,
        };
      });
    });
  }, [positions, order, selectedId]);

  const onNodesChange = useCallback(
    (changes: NodeChange<FlowNode>[]) =>
      setNodes((n) => applyNodeChanges(changes.filter((c) => c.type === "dimensions"), n)),
    [],
  );

  const rfEdges = useMemo<RFEdge[]>(() => {
    const st = useRunStore.getState();
    const view = (st.scrubSeq === null ? st.live : st.historical) ?? null;
    return edges.map((e) => {
      const src = view?.nodes[e.source_node_id];
      const tgt = view?.nodes[e.target_node_id];
      const ts = tgt ? effectiveStatus(tgt) : "pending";
      const ss = src ? effectiveStatus(src) : "pending";
      const cls =
        ss === "skipped" || ts === "skipped"
          ? "lg-edge-dead"
          : ts === "running"
            ? "lg-edge-live"
            : ts === "stale"
              ? "lg-edge-stale"
              : "";
      return {
        id: e.id,
        source: e.source_node_id,
        target: e.target_node_id,
        animated: ts === "running",
        className: cls,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 14,
          height: 14,
          color: ts === "running" ? "var(--accent)" : "var(--line-strong)",
        },
      };
    });
  }, [edges, statusKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // Follow the graph as it grows, until the user takes the camera. When the whole graph
  // would be too small to read, follow the active frontier (running / waiting / stale steps
  // and their neighbours) instead.
  const activeKey = useView((s) =>
    s.nodeOrder
      .filter((id) => {
        const st = effectiveStatus(s.nodes[id]!);
        return st === "running" || st === "waiting_input" || st === "stale";
      })
      .join(","),
  );
  const width = useStore((s) => s.width);
  const height = useStore((s) => s.height);
  useEffect(() => {
    if (userMoved || order.length === 0 || !width || !height) return;
    if (fitTimer.current) clearTimeout(fitTimer.current);
    fitTimer.current = setTimeout(() => {
      const xs = Object.values(positions);
      const minX = Math.min(...xs.map((p) => p.x));
      const maxX = Math.max(...xs.map((p) => p.x + NODE_W));
      const minY = Math.min(...xs.map((p) => p.y));
      const maxY = Math.max(...xs.map((p) => p.y + NODE_H));
      const zoomAll = Math.min(width / ((maxX - minX) * 1.2), height / ((maxY - minY) * 1.2));
      const active = activeKey ? activeKey.split(",") : [];
      if (zoomAll >= MIN_READABLE_ZOOM || active.length === 0) {
        void fitView({ padding: 0.16, duration: 650, maxZoom: 1.05 });
        return;
      }
      const focus = new Set(active);
      for (const e of edges) {
        if (focus.has(e.target_node_id) || focus.has(e.source_node_id)) {
          focus.add(e.source_node_id);
          focus.add(e.target_node_id);
        }
      }
      void fitView({
        nodes: [...focus].map((id) => ({ id })),
        padding: 0.2,
        duration: 750,
        maxZoom: 1,
        minZoom: MIN_READABLE_ZOOM,
      });
    }, 120);
    return () => {
      if (fitTimer.current) clearTimeout(fitTimer.current);
    };
  }, [key, activeKey, userMoved, fitView, order.length, positions, edges, width, height]);

  return (
    <ReactFlow<FlowNode>
      className="lg-canvas"
      nodes={nodes}
      edges={rfEdges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onNodeClick={(_, n) => select(n.id)}
      onPaneClick={() => select(null)}
      onMove={(event) => {
        // Only a real user pan/zoom (not a click, not our own fitView) takes the camera.
        const t = (event as Event | null)?.type;
        if (t === "wheel" || t === "mousemove" || t === "touchmove" || t === "pointermove") setUserMoved(true);
      }}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable
      minZoom={0.2}
      maxZoom={1.6}
      proOptions={{ hideAttribution: true }}
      defaultEdgeOptions={{ type: "default" }}
      aria-label="Workflow graph"
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1.3} color="var(--dot)" />
      <Controls showInteractive={false} position="bottom-right" />
      {userMoved && order.length > 0 ? (
        <Panel position="top-right">
          <Button size="sm" variant="secondary" onClick={() => setUserMoved(false)} className="shadow-sm">
            <Crosshair className="size-3.5" />
            Follow graph
          </Button>
        </Panel>
      ) : null}
      {order.length === 0 ? <EmptyCanvas scrubbing={scrubbing} /> : null}
    </ReactFlow>
  );
}

function EmptyCanvas({ scrubbing }: { scrubbing: boolean }) {
  const status = useView((s) => s.runStatus);
  const failed = status === "failed" || status === "cancelled";
  return (
    <div className="pointer-events-none absolute inset-0 z-10 grid place-items-center">
      <div className="flex flex-col items-center gap-4 text-center">
        {!failed ? (
          <div className="flex items-center gap-3" aria-hidden>
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="lg-shimmer h-16 w-36 rounded-[10px] border border-dashed border-line-strong bg-surface/60"
                style={{ animationDelay: `${i * 120}ms` }}
              />
            ))}
          </div>
        ) : null}
        <p className="max-w-sm text-[13px] text-ink-3">
          {scrubbing
            ? "Nothing had been planned yet at this point in the run."
            : failed
              ? "This run stopped before the planner added any steps."
              : "The planner is sketching the workflow. Steps appear here as it adds them."}
        </p>
      </div>
    </div>
  );
}

export function Canvas() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
