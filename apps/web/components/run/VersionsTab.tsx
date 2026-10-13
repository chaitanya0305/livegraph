"use client";

import { diffWordsWithSpace } from "diff";
import { History } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { splitRows } from "@/lib/diffRows";
import type { Node, NodeVersion } from "@/lib/protocol";
import { useReadOnly, useRunStore, useView } from "@/lib/store";
import { REASON_TONE, relativeTime } from "../meta";
import { Button, SectionLabel, cx } from "../ui";
import { useAction } from "./useAction";

export function VersionsTab({ node }: { node: Node }) {
  const readOnly = useReadOnly();
  const backend = useRunStore((s) => s.backend);
  const stateVersions = useView((s) => s.versions[node.id]);
  const [fetched, setFetched] = useState<NodeVersion[] | null>(null);
  const count = stateVersions?.length ?? 0;

  useEffect(() => {
    if (readOnly || !backend) return;
    let alive = true;
    backend
      .listVersions(node.id)
      .then((v) => alive && setFetched(v))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [backend, node.id, count, readOnly]);

  const versions = useMemo(() => {
    const byId = new Map<string, NodeVersion>();
    if (!readOnly) for (const v of fetched ?? []) byId.set(v.id, v);
    for (const v of stateVersions ?? []) byId.set(v.id, { ...byId.get(v.id), ...v });
    return [...byId.values()].sort((a, b) => b.version - a.version);
  }, [fetched, stateVersions, readOnly]);

  const currentId = node.current_version_id;
  const [aId, setA] = useState<string | null>(null);
  const [bId, setB] = useState<string | null>(null);

  // Default comparison: the version before current → current.
  const effB = bId ?? currentId ?? versions[0]?.id ?? null;
  const bVer = versions.find((v) => v.id === effB) ?? null;
  const effA = aId ?? versions.find((v) => bVer && v.version < bVer.version)?.id ?? null;
  const aVer = versions.find((v) => v.id === effA) ?? null;

  if (!versions.length) {
    return (
      <div className="px-5 py-6 text-[13px] text-ink-3">
        No versions yet. Every output this step produces is kept here, so you can compare and restore them.
      </div>
    );
  }

  return (
    <div className="space-y-5 px-5 py-4">
      <ul className="space-y-1">
        {versions.map((v) => (
          <VersionRow
            key={v.id}
            v={v}
            current={v.id === currentId}
            isA={v.id === effA}
            isB={v.id === effB}
            onA={() => setA(v.id)}
            onB={() => setB(v.id)}
            nodeId={node.id}
            readOnly={readOnly}
          />
        ))}
      </ul>
      {aVer && bVer && aVer.id !== bVer.id ? (
        <DiffView a={aVer} b={bVer} />
      ) : versions.length > 1 ? (
        <p className="text-[12.5px] text-ink-3">Pick two different versions as A and B to compare them.</p>
      ) : null}
    </div>
  );
}

function VersionRow({
  v,
  current,
  isA,
  isB,
  onA,
  onB,
  nodeId,
  readOnly,
}: {
  v: NodeVersion;
  current: boolean;
  isA: boolean;
  isB: boolean;
  onA: () => void;
  onB: () => void;
  nodeId: string;
  readOnly: boolean;
}) {
  const action = useAction();
  return (
    <li
      className={cx(
        "group flex items-start gap-3 rounded-lg px-2.5 py-2 transition-colors",
        current ? "bg-surface ring-1 ring-line-strong" : "hover:bg-sunken",
      )}
    >
      <span className="mt-0.5 w-7 shrink-0 font-mono text-[12.5px] font-medium text-ink tabular-nums">v{v.version}</span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className={cx("rounded px-1.5 py-px text-[11px] font-medium", REASON_TONE[v.reason] ?? "bg-sunken text-ink-2")}>
            {v.reason}
          </span>
          {current ? <span className="text-[11px] font-medium text-green-ink">current</span> : null}
          <span className="text-[11px] text-ink-4">{relativeTime(v.created_at)}</span>
        </div>
        {v.steer_text ? <p className="mt-1 line-clamp-2 text-[12px] text-ink-2">“{v.steer_text}”</p> : null}
        {action.error ? <p className="mt-1 text-[11.5px] text-red-ink">{action.error}</p> : null}
      </div>
      <div className="flex shrink-0 items-center gap-1">
        <CompareToggle label="A" active={isA} onClick={onA} />
        <CompareToggle label="B" active={isB} onClick={onB} />
        {!current && !readOnly ? (
          <Button
            size="sm"
            variant="ghost"
            title="Make this the current version; downstream steps recompute"
            loading={action.pending === "restore"}
            onClick={() => action.run("restore", (b) => b.restore(nodeId, v.id))}
          >
            {action.pending !== "restore" ? <History className="size-3.5" /> : null}
            Restore
          </Button>
        ) : null}
      </div>
    </li>
  );
}

function CompareToggle({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      title={`Compare as ${label}`}
      className={cx(
        "grid size-6 place-items-center rounded-md font-mono text-[11px] font-medium transition-colors",
        active ? "bg-ink text-surface" : "text-ink-4 hover:bg-line hover:text-ink",
      )}
    >
      {label}
    </button>
  );
}

function DiffView({ a, b }: { a: NodeVersion; b: NodeVersion }) {
  const [mode, setMode] = useState<"inline" | "split">("inline");
  const words = useMemo(() => diffWordsWithSpace(a.output_text, b.output_text), [a.output_text, b.output_text]);
  const rows = useMemo(() => (mode === "split" ? splitRows(a.output_text, b.output_text) : []), [mode, a, b]);
  const added = words.filter((p) => p.added).reduce((n, p) => n + p.value.split(/\s+/).filter(Boolean).length, 0);
  const removed = words.filter((p) => p.removed).reduce((n, p) => n + p.value.split(/\s+/).filter(Boolean).length, 0);

  return (
    <section className="space-y-2.5">
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-mono text-[12px] font-medium text-ink-2">
          v{a.version} → v{b.version}
        </h3>

          <div className="flex rounded-md bg-sunken p-0.5 text-[11.5px]">
            {(["inline", "split"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cx(
                  "rounded px-2 py-0.5 transition-colors",
                  mode === m ? "bg-surface font-medium text-ink shadow-sm" : "text-ink-3 hover:text-ink",
                )}
              >
                {m === "split" ? "Side by side" : "Inline"}
              </button>
            ))}
          </div>
      </div>
      <p className="font-mono text-[11px] text-ink-3">
        <span className="text-green-ink">+{added}</span> <span className="text-red-ink">−{removed}</span> words
      </p>
      {mode === "inline" ? (
        <div className="rounded-lg border border-line bg-surface p-3 text-[12.5px] leading-relaxed whitespace-pre-wrap">
          {words.map((p, i) => (
            <span
              key={i}
              className={cx(
                p.added && "rounded-sm bg-green-soft text-green-ink",
                p.removed && "rounded-sm bg-red-soft text-red-ink line-through decoration-red/50",
              )}
            >
              {p.value}
            </span>
          ))}
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-line bg-surface font-mono text-[11px] leading-[1.55]">
          <div className="grid grid-cols-2 border-b border-line bg-sunken text-ink-3">
            <div className="px-2 py-1">v{a.version}</div>
            <div className="border-l border-line px-2 py-1">v{b.version}</div>
          </div>
          {rows.map((r, i) => (
            <div key={i} className="grid grid-cols-2">
              {[r.left, r.right].map((c, j) => (
                <div
                  key={j}
                  className={cx(
                    "flex gap-2 px-2 py-px whitespace-pre-wrap break-words",
                    j === 1 && "border-l border-line",
                    c.kind === "del" && "bg-red-soft text-red-ink",
                    c.kind === "add" && "bg-green-soft text-green-ink",
                    c.kind === "empty" && "bg-sunken/60",
                  )}
                >
                  <span className="w-5 shrink-0 text-right text-ink-4 select-none">{c.no ?? ""}</span>
                  <span className="min-w-0">{c.text || " "}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
