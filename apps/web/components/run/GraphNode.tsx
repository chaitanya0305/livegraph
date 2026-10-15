"use client";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Recycle, Wrench } from "lucide-react";
import { memo } from "react";
import { impactMark } from "@/lib/impact";
import { currentVersion, effectiveStatus, stepStatus, traceFor } from "@/lib/reducer";
import { useRunStore, useView } from "@/lib/store";
import type { CreatedBy, NodeStatus } from "@/lib/protocol";
import { NODE_STATUS_LABEL, nodeTypeMeta, stepLine, toolMeta } from "../meta";
import { cx } from "../ui";
import { NODE_H, NODE_W } from "./layout";

export type GraphNodeData = { nodeId: string };

function plain(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/^#+\s.*$/gm, " ")
    .replace(/^>\s?/gm, "")
    .replace(/[*_`#>|]/g, "")
    .replace(/\[(.*?)\]\(.*?\)/g, "$1")
    .replace(/^\s*[-+]\s+/gm, "· ")
    .replace(/\s+/g, " ")
    .trim();
}

const CARD: Record<NodeStatus, string> = {
  pending: "bg-surface/70 border-line text-ink-3",
  running: "bg-surface border-transparent lg-running",
  waiting_input: "bg-amber-soft border-transparent lg-attention",
  done: "bg-surface border-line-strong shadow-[0_1px_0_oklch(0.2_0.02_258/0.04),0_2px_8px_-4px_oklch(0.2_0.02_258/0.12)]",
  stale: "bg-sunken border-dashed border-ink-4",
  failed: "bg-red-soft border-red",
  skipped: "bg-sunken/60 border-line opacity-70",
};

const CREATOR: Record<CreatedBy, { label: string; cls: string }> = {
  planner: { label: "planner", cls: "text-ink-4" },
  critic: { label: "critic", cls: "text-violet bg-violet-soft px-1.5 rounded" },
  user: { label: "you", cls: "text-accent-ink bg-accent-soft px-1.5 rounded" },
};

function GraphNodeImpl({ data, selected }: NodeProps & { data: GraphNodeData }) {
  const id = data.nodeId;
  const node = useView((s) => s.nodes[id]);
  const live = useView((s) => s.live[id]);
  const version = useView((s) => currentVersion(s, id));
  const prompt = useView((s) => s.prompts[id]);
  const reason = useView((s) => s.statusReason[id]);
  const trace = useView((s) => traceFor(s, id));
  const pendingInputs = useView((s) => {
    let n = 0;
    for (const e of Object.values(s.edges)) {
      if (e.target_node_id !== id) continue;
      const p = s.nodes[e.source_node_id];
      if (p && !p.removed && p.status !== "done") n++;
    }
    return n;
  });
  const isNew = useRunStore((s) => s.scrubSeq === null);
  const reused = useView((s) => !!s.reused[id]);
  const mark = useRunStore((s) => impactMark(s.impactPreview, id));

  if (!node) return null;
  const status = effectiveStatus(node);
  const meta = nodeTypeMeta(node.type);
  const Icon = meta.icon;
  const creator = CREATOR[node.created_by] ?? CREATOR.planner;

  let body: React.ReactNode;
  switch (status) {
    case "running": {
      const tail = plain((live ?? "").slice(-220));
      const latest = trace.at(-1);
      if (!latest && !tail) {
        body = <p className="text-[11.5px] text-accent-ink">Thinking…</p>;
        break;
      }
      const StepIcon = latest ? toolMeta(latest.tool).icon : null;
      const inFlight = latest ? stepStatus(latest) === "running" : false;
      body = (
        <div className="space-y-0.5">
          {latest && StepIcon ? (
            <p
              className={cx(
                "flex items-center gap-1 text-[11px] leading-[1.45]",
                inFlight ? "text-sky-ink" : "text-ink-3",
              )}
              title={stepLine(latest)}
            >
              <StepIcon className={cx("size-3 shrink-0", inFlight && "animate-pulse")} strokeWidth={2} />
              <span className="min-w-0 truncate">{stepLine(latest)}</span>
            </p>
          ) : null}
          {tail ? (
            <p
              className={cx(
                "lg-stream-tail lg-caret text-[11.5px] leading-[1.45] text-ink-2",
                latest ? "line-clamp-2" : "line-clamp-3",
              )}
            >
              {tail}
            </p>
          ) : null}
        </div>
      );
      break;
    }
    case "waiting_input":
      body = (
        <p className="line-clamp-3 text-[11.5px] leading-[1.45] text-amber-ink">
          <span className="font-medium">Needs your answer.</span> {prompt ?? String(node.config.prompt ?? "")}
        </p>
      );
      break;
    case "stale":
      body = (
        <div className="space-y-1.5">
          <div className="lg-shimmer h-1.5 w-full rounded-full bg-line" />
          <p className="truncate text-[11.5px] text-ink-3">Recomputing{reason ? ` · ${reason}` : ""}</p>
        </div>
      );
      break;
    case "failed":
      body = <p className="line-clamp-3 text-[11.5px] text-red-ink">{reason || "This step failed."}</p>;
      break;
    case "skipped":
      body = <p className="text-[11.5px] text-ink-3">{node.removed ? "Vetoed" : reason || "Skipped"}</p>;
      break;
    case "done":
      body = (
        <p className="line-clamp-3 text-[11.5px] leading-[1.45] text-ink-2">
          {version ? plain(version.output_text).slice(0, 220) : "Done"}
        </p>
      );
      break;
    default:
      body = (
        <p className="text-[11.5px] text-ink-4">
          {pendingInputs > 0 ? `Waiting on ${pendingInputs} input${pendingInputs > 1 ? "s" : ""}` : "Queued"}
        </p>
      );
  }

  return (
    <div
      style={{ width: NODE_W, height: NODE_H }}
      className={cx(
        "group relative flex flex-col rounded-[10px] border px-3 pt-2.5 pb-2 transition-[background-color,border-color,opacity] duration-300",
        CARD[status],
        isNew && "lg-node-in",
        selected && "outline-2 outline-offset-2 outline-ink",
        mark === "rerun" && "lg-impact-rerun",
        mark === "skipped" && "lg-impact-skip",
      )}
      title={mark === "rerun" ? "Would rerun" : mark === "skipped" ? "Would be skipped" : undefined}
    >
      <Handle type="target" position={Position.Left} isConnectable={false} />
      <div className="flex items-center gap-2">
        <span
          className={cx(
            "grid size-6 shrink-0 place-items-center rounded-md",
            status === "running" ? "bg-accent text-white" : status === "waiting_input" ? "bg-amber text-ink" : "bg-sunken text-ink-2",
          )}
        >
          <Icon className="size-3.5" strokeWidth={2} />
        </span>
        <span
          className={cx(
            "min-w-0 flex-1 truncate text-[13px] font-medium",
            status === "pending" ? "text-ink-3" : "text-ink",
            status === "skipped" && "line-through decoration-ink-3",
          )}
          title={node.title}
        >
          {node.title}
        </span>
        {version ? (
          <span className="shrink-0 rounded bg-sunken px-1 font-mono text-[10.5px] text-ink-2 tabular-nums">
            v{version.version}
          </span>
        ) : null}
      </div>
      <div className="mt-2 min-h-0 flex-1 overflow-hidden">{body}</div>
      <div className="mt-1 flex items-center justify-between text-[10.5px]">
        <span className="flex min-w-0 items-center gap-1.5 overflow-hidden text-ink-3">
          {meta.label}
          <span className={creator.cls}>{creator.label}</span>
          {status === "done" && (reason === "upstream unchanged" || reused) ? (
            <span
              className="inline-flex min-w-0 items-center gap-0.5 rounded bg-green-soft px-1 whitespace-nowrap text-green-ink"
              title={
                reason === "upstream unchanged"
                  ? "Its inputs came back unchanged, so the previous result was reused (no model call)"
                  : "Inputs matched an earlier version, so that result was reused (no model call)"
              }
            >
              <Recycle className="size-2.5" strokeWidth={2.25} />
              {reason === "upstream unchanged" ? "unchanged · reused" : "reused"}
            </span>
          ) : null}
          {trace.length ? (
            <span
              className="inline-flex items-center gap-0.5 rounded bg-sky-soft px-1 font-mono text-sky-ink tabular-nums"
              title={`${trace.length} tool call${trace.length > 1 ? "s" : ""}`}
            >
              <Wrench className="size-2.5" strokeWidth={2.25} />
              {trace.length}
            </span>
          ) : null}
        </span>
        <span
          className={cx(
            "font-medium",
            status === "running" && "text-accent-ink",
            status === "waiting_input" && "text-amber-ink",
            status === "failed" && "text-red-ink",
            status === "done" && "text-green-ink",
            (status === "pending" || status === "stale" || status === "skipped") && "text-ink-3",
          )}
        >
          {NODE_STATUS_LABEL[status]}
        </span>
      </div>
      <Handle type="source" position={Position.Right} isConnectable={false} />
    </div>
  );
}

export const GraphNode = memo(GraphNodeImpl);
