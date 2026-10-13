"use client";

import { ExternalLink, Pin, ShieldOff } from "lucide-react";
import type { Node, RankedChunk, Trust } from "@/lib/protocol";
import { currentVersion } from "@/lib/reducer";
import { useReadOnly, useView } from "@/lib/store";
import { cx } from "../ui";
import { useAction } from "./useAction";

const NO_CHUNKS: RankedChunk[] = [];

export function SourcesTab({ node }: { node: Node }) {
  const version = useView((s) => currentVersion(s, node.id));
  const chunks = useView((s) => (version ? s.retrievals[version.id] ?? version.retrievals ?? NO_CHUNKS : NO_CHUNKS));

  if (!chunks.length) {
    return (
      <div className="px-5 py-6 text-[13px] text-ink-3">
        {version
          ? "This version didn't retrieve any passages."
          : "Passages this step retrieves appear here, ranked by relevance. Distrust a source to drop it and recompute; pin one to favor it."}
      </div>
    );
  }
  const sorted = [...chunks].sort((a, b) => a.rank - b.rank);
  return (
    <ol className="divide-y divide-line">
      {sorted.map((c) => (
        <ChunkRow key={c.chunk_id} chunk={c} />
      ))}
    </ol>
  );
}

function ChunkRow({ chunk }: { chunk: RankedChunk }) {
  const docTrust = useView((s) => s.documents[chunk.document_id]?.trust);
  const trust: Trust = docTrust ?? chunk.trust;
  const readOnly = useReadOnly();
  const action = useAction();
  const set = (t: Trust) => action.run(t, (b) => b.setTrust(chunk.document_id, t));

  return (
    <li className={cx("px-5 py-3.5", trust === "distrusted" && "opacity-60")}>
      <div className="flex items-center gap-2">
        <span className="font-mono text-[11px] text-ink-4 tabular-nums">#{chunk.rank}</span>
        <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium" title={chunk.document_title}>
          {chunk.document_title}
        </span>
        {chunk.source_url ? (
          <a
            href={chunk.source_url}
            target="_blank"
            rel="noreferrer"
            className="text-ink-4 hover:text-accent-ink"
            aria-label="Open source"
          >
            <ExternalLink className="size-3.5" />
          </a>
        ) : null}
      </div>
      <div className="mt-1.5 flex items-center gap-2">
        <div className="h-1 w-20 overflow-hidden rounded-full bg-line" aria-hidden>
          <div
            className="h-full origin-left rounded-full bg-teal"
            style={{ transform: `scaleX(${Math.max(0, Math.min(1, chunk.score))})` }}
          />
        </div>
        <span className="font-mono text-[11px] text-ink-3 tabular-nums">{chunk.score.toFixed(2)}</span>
        {trust !== "normal" ? (
          <span
            className={cx(
              "rounded px-1.5 text-[10.5px] font-medium",
              trust === "pinned" ? "bg-accent-soft text-accent-ink" : "bg-red-soft text-red-ink",
            )}
          >
            {trust}
          </span>
        ) : null}
      </div>
      <p className="mt-2 line-clamp-4 text-[12.5px] leading-relaxed text-ink-2">{chunk.text}</p>
      {!readOnly ? (
        <div className="mt-2 flex items-center gap-1">
          <TrustToggle
            active={trust === "pinned"}
            busy={action.pending === "pinned" || (action.pending === "normal" && trust === "pinned")}
            onClick={() => set(trust === "pinned" ? "normal" : "pinned")}
            icon={<Pin className="size-3" />}
            label={trust === "pinned" ? "Pinned" : "Pin"}
            tone="accent"
          />
          <TrustToggle
            active={trust === "distrusted"}
            busy={action.pending === "distrusted" || (action.pending === "normal" && trust === "distrusted")}
            onClick={() => set(trust === "distrusted" ? "normal" : "distrusted")}
            icon={<ShieldOff className="size-3" />}
            label={trust === "distrusted" ? "Distrusted" : "Distrust"}
            tone="red"
          />
          {action.error ? <span className="ml-2 text-[11.5px] text-red-ink">{action.error}</span> : null}
        </div>
      ) : null}
    </li>
  );
}

function TrustToggle({
  active,
  busy,
  onClick,
  icon,
  label,
  tone,
}: {
  active: boolean;
  busy: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  tone: "accent" | "red";
}) {
  return (
    <button
      onClick={onClick}
      disabled={busy}
      aria-pressed={active}
      className={cx(
        "inline-flex h-6 items-center gap-1 rounded-md border px-2 text-[11.5px] font-medium transition-colors disabled:opacity-60",
        active
          ? tone === "accent"
            ? "border-transparent bg-accent-soft text-accent-ink"
            : "border-transparent bg-red-soft text-red-ink"
          : "border-line text-ink-3 hover:border-line-strong hover:text-ink",
      )}
    >
      {icon}
      {label}
    </button>
  );
}
