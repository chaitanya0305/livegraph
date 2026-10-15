"use client";

import { Pause, Play } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PersistedEvent } from "@/lib/protocol";
import { useRunStore } from "@/lib/store";
import { EVENT_GROUPS, clockTime, eventColor } from "../meta";
import { cx } from "../ui";

const TICK_H: Partial<Record<string, number>> = {
  "node.added": 0.85,
  "edge.added": 0.45,
  "node.version": 1,
  "claim.confidence": 0.7,
  "claim.upserted": 0.6,
  "run.status": 1,
  "tool.call": 0.4,
  "tool.result": 0.3,
  "agent.message": 0.65,
  error: 1,
};

/** Bottom timeline over the persisted event log; dragging folds history into view. */
export function Timeline() {
  const log = useRunStore((s) => s.log);
  const scrubSeq = useRunStore((s) => s.scrubSeq);
  const scrubTo = useRunStore((s) => s.scrubTo);
  const trackRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ x: number; e: PersistedEvent } | null>(null);
  const [playing, setPlaying] = useState(false);
  const dragging = useRef(false);

  const min = log[0]?.seq ?? 0;
  const max = log.at(-1)?.seq ?? 0;
  const span = Math.max(1, max - min);
  const at = scrubSeq ?? max;
  const frac = max > min ? (at - min) / span : 1;
  const live = scrubSeq === null;

  const nearest = useCallback(
    (seq: number): PersistedEvent | undefined => {
      // binary search on sorted log
      let lo = 0;
      let hi = log.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (log[mid]!.seq < seq) lo = mid + 1;
        else hi = mid;
      }
      return log[lo];
    },
    [log],
  );

  const seqAt = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el) return max;
      const r = el.getBoundingClientRect();
      const f = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
      return Math.round(min + f * span);
    },
    [min, span, max],
  );

  // Replay: walk forward through history, then rejoin live.
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      const s = useRunStore.getState();
      const cur = s.scrubSeq ?? min;
      const next = cur + 1;
      if (next >= max) {
        setPlaying(false);
        s.scrubTo(null);
      } else s.scrubTo(next);
    }, 70);
    return () => clearInterval(id);
  }, [playing, min, max]);

  const ticks = useMemo(
    () =>
      log.map((e) => ({
        x: ((e.seq - min) / span) * 1000,
        h: TICK_H[e.type] ?? 0.55,
        c: eventColor(e.type),
        k: e.seq,
      })),
    [log, min, span],
  );

  const viewed = live ? undefined : nearest(at);
  const empty = log.length === 0;

  return (
    <div className="flex h-[68px] shrink-0 items-center gap-4 border-t border-line bg-panel px-4">
      <div className="flex w-[148px] shrink-0 items-center gap-1.5">
        {live ? (
          <span className="inline-flex h-8 items-center gap-2 rounded-lg bg-accent-soft px-3 text-[12px] font-semibold tracking-wide text-accent-ink">
            <span className="relative inline-block size-1.5 rounded-full bg-accent">
              <span className="lg-ping absolute inset-0 rounded-full text-accent" />
            </span>
            LIVE
          </span>
        ) : (
          <button
            onClick={() => {
              setPlaying(false);
              scrubTo(null);
            }}
            className="inline-flex h-8 items-center gap-2 rounded-lg border border-line-strong bg-surface px-3 text-[12px] font-semibold tracking-wide text-ink-2 transition-colors hover:border-accent hover:text-accent-ink"
          >
            <span className="size-1.5 rounded-full bg-ink-4" />
            LIVE
          </button>
        )}
        <button
          onClick={() => {
            if (playing) return setPlaying(false);
            if (live) scrubTo(min);
            setPlaying(true);
          }}
          disabled={empty}
          className="grid size-8 place-items-center rounded-lg text-ink-3 transition-colors hover:bg-sunken hover:text-ink disabled:opacity-40"
          aria-label={playing ? "Pause replay" : "Replay run from the start"}
          title={playing ? "Pause replay" : "Replay from the start"}
        >
          {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
        </button>
      </div>

      <div className="relative min-w-0 flex-1">
        <div
          ref={trackRef}
          role="slider"
          tabIndex={0}
          aria-label="Run timeline"
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={at}
          aria-valuetext={live ? "Live" : `Event ${at} of ${max}`}
          className={cx(
            "relative h-9 cursor-ew-resize touch-none rounded-md bg-surface ring-1 ring-line outline-none select-none focus-visible:ring-2 focus-visible:ring-accent",
            empty && "cursor-default",
          )}
          onPointerDown={(e) => {
            if (empty) return;
            dragging.current = true;
            setPlaying(false);
            e.currentTarget.setPointerCapture(e.pointerId);
            scrubTo(seqAt(e.clientX));
          }}
          onPointerMove={(e) => {
            const seq = seqAt(e.clientX);
            const ev = nearest(seq);
            const r = trackRef.current!.getBoundingClientRect();
            if (ev) setHover({ x: e.clientX - r.left, e: ev });
            if (dragging.current) scrubTo(seq);
          }}
          onPointerUp={() => {
            dragging.current = false;
          }}
          onPointerLeave={() => setHover(null)}
          onKeyDown={(e) => {
            const step = e.shiftKey ? 10 : 1;
            if (e.key === "ArrowLeft") scrubTo(Math.max(min, at - step));
            else if (e.key === "ArrowRight") scrubTo(at + step);
            else if (e.key === "Home") scrubTo(min);
            else if (e.key === "End" || e.key === "Escape") scrubTo(null);
            else return;
            e.preventDefault();
          }}
        >
          <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1000 36" preserveAspectRatio="none" aria-hidden>
            {ticks.map((t) => (
              <line
                key={t.k}
                x1={t.x}
                x2={t.x}
                y1={34}
                y2={34 - t.h * 30}
                stroke={t.c}
                strokeWidth={1.25}
                vectorEffect="non-scaling-stroke"
                opacity={!live && t.k > at ? 0.22 : 0.85}
              />
            ))}
          </svg>
          {!empty ? (
            <div
              className="pointer-events-none absolute inset-y-[-4px] w-0"
              style={{ left: `${frac * 100}%` }}
              aria-hidden
            >
              <div className={cx("absolute inset-y-0 -left-px w-0.5 rounded-full", live ? "bg-accent" : "bg-ink")} />
              <div
                className={cx(
                  "absolute -top-1 -left-[5px] size-3 rounded-full border-2 border-surface shadow-sm",
                  live ? "bg-accent" : "bg-ink",
                )}
              />
            </div>
          ) : (
            <p className="absolute inset-0 grid place-items-center text-[11.5px] text-ink-4">
              Events will appear here as the run progresses
            </p>
          )}
        </div>
        {hover && !empty ? (
          <div
            className="pointer-events-none absolute bottom-full z-20 mb-2 -translate-x-1/2 rounded-md bg-ink px-2 py-1 font-mono text-[11px] whitespace-nowrap text-surface shadow-lg"
            style={{ left: hover.x }}
          >
            <span className="mr-1.5 inline-block size-1.5 rounded-full" style={{ background: eventColor(hover.e.type) }} />
            #{hover.e.seq} {hover.e.type} · {clockTime(hover.e.ts)}
          </div>
        ) : null}
        <div className="mt-1 flex gap-3 overflow-hidden text-[10.5px] whitespace-nowrap text-ink-4">
          {EVENT_GROUPS.map((g) => (
            <span key={g.label} className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-0.5 rounded-full" style={{ background: g.color }} />
              {g.label}
            </span>
          ))}
        </div>
      </div>

      <div className="w-[132px] shrink-0 text-right font-mono text-[11.5px] text-ink-3 tabular-nums">
        <div className={live ? "text-ink-2" : "text-ink"}>
          {at} <span className="text-ink-4">/ {max}</span>
        </div>
        <div className="text-[10.5px] text-ink-4">{viewed ? clockTime(viewed.ts) : live && max ? "now" : ""}</div>
      </div>
    </div>
  );
}
