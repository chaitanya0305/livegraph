"use client";

import { ChevronRight } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Claim } from "@/lib/protocol";
import { currentVersion } from "@/lib/reducer";
import { useRunStore, useView } from "@/lib/store";
import { useCitations } from "@/lib/useCitations";
import { confidenceColor } from "../meta";
import { Markdown, SectionLabel, cx } from "../ui";

/** Tween a number toward `value` with an exponential ease-out. */
function useTween(value: number | null, ms = 900): number | null {
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  useEffect(() => {
    if (value === null) {
      setShown(null);
      from.current = null;
      return;
    }
    const start = from.current ?? 0;
    const t0 = performance.now();
    let raf = 0;
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / ms);
      const eased = 1 - Math.pow(2, -10 * k);
      const v = start + (value - start) * (k === 1 ? 1 : eased);
      setShown(v);
      from.current = v;
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value, ms]);
  return shown;
}

export function BriefPanel() {
  const order = useView((s) => s.claimOrder);
  const claims = useView((s) => s.claims);
  const editorId = useView((s) => s.nodeOrder.find((id) => s.nodes[id]?.type === "editor" && !s.nodes[id]?.removed));
  const judged = order.filter((id) => claims[id]?.confidence !== null).length;

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto">
      <div className="sticky top-0 z-10 border-b border-line bg-panel/95 px-5 pt-4 pb-3 backdrop-blur-[2px]">
        <SectionLabel
          right={
            order.length ? (
              <span className="font-mono text-[11px] text-ink-4 tabular-nums">
                {judged}/{order.length} judged
              </span>
            ) : null
          }
        >
          Claims
        </SectionLabel>
      </div>
      {order.length === 0 ? (
        <p className="px-5 py-4 text-[12.5px] leading-relaxed text-ink-3">
          Claims show up here once research is distilled. The judge then scores each one, and you can watch confidence
          move as the prosecutor and defender argue.
        </p>
      ) : (
        <ul className="px-2 py-2">
          {order.map((id) => (claims[id] ? <ClaimRow key={id} claim={claims[id]!} /> : null))}
        </ul>
      )}
      <FinalBrief editorId={editorId ?? null} />
    </div>
  );
}

function ClaimRow({ claim }: { claim: Claim }) {
  const [open, setOpen] = useState(false);
  const shown = useTween(claim.confidence);
  const struck = claim.status === "struck";
  const color = confidenceColor(claim.confidence);
  const last = claim.history.at(-1);

  return (
    <li>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full rounded-lg px-3 py-2.5 text-left transition-colors hover:bg-sunken"
      >
        <div className="flex items-start gap-2.5">
          <span className="mt-px font-mono text-[11px] font-medium text-ink-3">{claim.key}</span>
          <p
            className={cx(
              "min-w-0 flex-1 text-[13px] leading-snug transition-colors duration-500",
              struck ? "text-ink-4 line-through decoration-red/60" : "text-ink",
            )}
          >
            {claim.text}
          </p>
          <ChevronRight
            className={cx("mt-0.5 size-3.5 shrink-0 text-ink-4 transition-transform duration-200", open && "rotate-90")}
          />
        </div>
        <div className="mt-2 flex items-center gap-2.5 pl-7">
          <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-line" aria-hidden>
            {claim.confidence === null ? (
              <div className="lg-shimmer absolute inset-0" />
            ) : (
              <div
                className="absolute inset-y-0 left-0 w-full origin-left rounded-full transition-colors duration-500"
                style={{ transform: `scaleX(${shown ?? 0})`, background: struck ? "var(--ink-4)" : color }}
              />
            )}
          </div>
          <span
            className={cx("w-9 text-right font-mono text-[11.5px] tabular-nums", struck ? "text-ink-4" : "text-ink-2")}
            aria-label={claim.confidence === null ? "not yet judged" : `confidence ${Math.round(claim.confidence * 100)}%`}
          >
            {shown === null ? "·" : `${Math.round(shown * 100)}%`}
          </span>
        </div>
      </button>
      {open ? (
        <div className="lg-fade-in space-y-2 px-3 pb-3 pl-10">
          <Sparkline claim={claim} />
          {last?.rationale ? <p className="text-[12px] leading-relaxed text-ink-2">{last.rationale}</p> : null}
          {struck ? <p className="text-[11.5px] text-red-ink">Struck by the judge; excluded from the brief.</p> : null}
        </div>
      ) : null}
    </li>
  );
}

function Sparkline({ claim }: { claim: Claim }) {
  const pts = claim.history.filter((h) => h.confidence !== null).map((h) => h.confidence!) as number[];
  if (pts.length === 0) return <p className="text-[11.5px] text-ink-4">No confidence history yet.</p>;
  const W = 220;
  const H = 40;
  const series = pts.length === 1 ? [pts[0]!, pts[0]!] : pts;
  const x = (i: number) => (i / (series.length - 1)) * (W - 8) + 4;
  const y = (v: number) => H - 4 - v * (H - 8);
  const d = series.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const lastV = series.at(-1)!;
  return (
    <figure className="space-y-1">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="overflow-visible" role="img" aria-label="Confidence over time">
        <line x1={4} x2={W - 4} y1={y(0.5)} y2={y(0.5)} stroke="var(--line)" strokeDasharray="2 3" />
        <path d={d} fill="none" stroke="var(--ink-3)" strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
        {series.map((v, i) => (
          <circle key={i} cx={x(i)} cy={y(v)} r={i === series.length - 1 ? 3 : 1.75} fill={i === series.length - 1 ? confidenceColor(lastV) : "var(--ink-3)"} />
        ))}
      </svg>
      <figcaption className="font-mono text-[10.5px] text-ink-4">
        {pts.map((p) => Math.round(p * 100)).join(" → ")}%
      </figcaption>
    </figure>
  );
}

function FinalBrief({ editorId }: { editorId: string | null }) {
  const version = useView((s) => (editorId ? currentVersion(s, editorId) : null));
  const live = useView((s) => (editorId ? s.live[editorId] : undefined));
  const status = useView((s) => (editorId ? s.nodes[editorId]?.status : undefined));
  const select = useRunStore((s) => s.select);
  const citations = useCitations();
  const streaming = status === "running" && !!live;
  const text = streaming ? live! : version?.output_text;

  return (
    <section className="mt-auto border-t border-line px-5 py-4">
      <SectionLabel
        right={
          editorId && text ? (
            <button className="text-[11.5px] text-ink-3 hover:text-ink" onClick={() => select(editorId)}>
              {streaming ? "Writing…" : status === "stale" ? "Rewriting soon" : `v${version?.version ?? ""}`}
            </button>
          ) : null
        }
      >
        Brief
      </SectionLabel>
      {text ? (
        <div className={cx("mt-3 transition-opacity", status === "stale" && "opacity-60")}>
          <Markdown className="md-compact" streaming={streaming} citations={citations}>
            {text}
          </Markdown>
        </div>
      ) : (
        <p className="mt-2 text-[12.5px] leading-relaxed text-ink-3">
          The editor writes the brief last, leaving out struck claims and hedging the shaky ones.
        </p>
      )}
    </section>
  );
}
