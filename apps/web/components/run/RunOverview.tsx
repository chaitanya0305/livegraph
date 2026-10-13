"use client";

import { MousePointerClick } from "lucide-react";
import { effectiveStatus } from "@/lib/reducer";
import { useRunStore, useView } from "@/lib/store";
import { NODE_STATUS_LABEL, clockTime, nodeTypeMeta } from "../meta";
import { SectionLabel, StatusDot, cx } from "../ui";

/** Inspector content when no node is selected: an index of steps and any errors. */
export function RunOverview() {
  const order = useView((s) => s.nodeOrder);
  const nodes = useView((s) => s.nodes);
  const errors = useView((s) => s.errors);
  const select = useRunStore((s) => s.select);

  const counts = order.reduce<Record<string, number>>((acc, id) => {
    const st = effectiveStatus(nodes[id]!);
    acc[st] = (acc[st] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto">
      <div className="border-b border-line px-5 py-4">
        <div className="flex items-start gap-3">
          <MousePointerClick className="mt-0.5 size-4 shrink-0 text-ink-3" />
          <p className="text-[13px] leading-relaxed text-ink-2">
            Select a step to read its output, compare versions, check its sources, or steer it while it runs.
          </p>
        </div>
      </div>

      {errors.length ? (
        <div className="space-y-2 border-b border-line px-5 py-4">
          <SectionLabel>Errors</SectionLabel>
          <ul className="space-y-1.5">
            {errors.slice(-5).map((e, i) => (
              <li key={i} className="rounded-md bg-red-soft px-2.5 py-1.5 text-[12px] text-red-ink">
                <button
                  className="text-left"
                  onClick={() => e.node_id && select(e.node_id)}
                  disabled={!e.node_id}
                >
                  {e.node_id && nodes[e.node_id] ? <span className="font-medium">{nodes[e.node_id]!.title}: </span> : null}
                  {e.message}
                  <span className="ml-1.5 text-red-ink/70">{clockTime(e.ts)}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="space-y-2 px-5 py-4">
        <SectionLabel
          right={
            <span className="font-mono text-[11px] text-ink-4 tabular-nums">
              {counts.done ?? 0}/{order.length} done
            </span>
          }
        >
          Steps
        </SectionLabel>
        {order.length === 0 ? (
          <p className="text-[12.5px] text-ink-3">No steps yet.</p>
        ) : (
          <ul className="-mx-2">
            {order.map((id) => {
              const n = nodes[id]!;
              const st = effectiveStatus(n);
              const Icon = nodeTypeMeta(n.type).icon;
              return (
                <li key={id}>
                  <button
                    onClick={() => select(id)}
                    className="flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left transition-colors hover:bg-sunken"
                  >
                    <Icon className="size-3.5 shrink-0 text-ink-3" />
                    <span
                      className={cx(
                        "min-w-0 flex-1 truncate text-[13px]",
                        st === "pending" ? "text-ink-3" : "text-ink",
                        st === "skipped" && "line-through",
                      )}
                    >
                      {n.title}
                    </span>
                    <StatusDot status={st} />
                    <span className="w-16 text-right text-[11.5px] text-ink-3">{NODE_STATUS_LABEL[st]}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
