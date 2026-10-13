"use client";

import { AlertTriangle, ArrowLeft, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useEffect } from "react";
import { API_URL } from "@/lib/api";
import { useRunStore, useView } from "@/lib/store";
import { clockTime } from "../meta";
import { Button } from "../ui";
import { BriefPanel } from "./BriefPanel";
import { Canvas } from "./Canvas";
import { Header } from "./Header";
import { Inspector } from "./Inspector";
import { Timeline } from "./Timeline";

export function RunView({ runId }: { runId: string }) {
  const open = useRunStore((s) => s.open);
  const close = useRunStore((s) => s.close);
  const loading = useRunStore((s) => s.loading);
  const loadError = useRunStore((s) => s.loadError);
  const hasState = useRunStore((s) => s.live !== null);

  useEffect(() => {
    void open(runId);
    return () => close();
  }, [runId, open, close]);

  if (loadError) return <LoadErrorView runId={runId} />;

  return (
    <div className="flex h-dvh min-h-0 flex-col overflow-hidden">
      <Header />
      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-[300px] shrink-0 border-r border-line bg-panel lg:block" aria-label="Brief">
          {hasState ? <BriefPanel /> : <PanelSkeleton />}
        </aside>
        <main className="relative min-w-0 flex-1">
          {hasState ? <Canvas /> : null}
          <ScrubBanner />
          <RunStatusBanner />
          {loading ? <p className="absolute inset-0 grid place-items-center text-[13px] text-ink-3">Loading run…</p> : null}
        </main>
        <aside className="w-[min(420px,36vw)] shrink-0 border-l border-line bg-panel" aria-label="Inspector">
          {hasState ? <Inspector /> : <PanelSkeleton />}
        </aside>
      </div>
      <Timeline />
    </div>
  );
}

function ScrubBanner() {
  const scrubSeq = useRunStore((s) => s.scrubSeq);
  const max = useRunStore((s) => s.log.at(-1)?.seq ?? 0);
  const ts = useRunStore((s) => s.historical?.lastTs ?? null);
  const scrubTo = useRunStore((s) => s.scrubTo);
  if (scrubSeq === null) return null;
  return (
    <div className="lg-fade-in pointer-events-none absolute inset-x-0 top-3 z-20 flex justify-center">
      <div className="pointer-events-auto flex items-center gap-3 rounded-full bg-ink py-1.5 pr-1.5 pl-4 text-[12.5px] text-surface shadow-lg">
        <span>
          Viewing <span className="font-mono tabular-nums">seq {scrubSeq} / {max}</span>
          {ts ? <span className="text-ink-4"> · {clockTime(ts)}</span> : null}
          <span className="text-ink-4"> · read-only</span>
        </span>
        <button
          onClick={() => scrubTo(null)}
          className="rounded-full bg-accent px-3 py-1 text-[12px] font-medium text-white transition-colors hover:bg-accent-ink"
        >
          Back to live
        </button>
      </div>
    </div>
  );
}

function RunStatusBanner() {
  const status = useView((s) => s.runStatus);
  const scrubbing = useRunStore((s) => s.scrubSeq !== null);
  const lastError = useView((s) => s.errors.at(-1));
  if (scrubbing || (status !== "failed" && status !== "cancelled")) return null;
  return (
    <div className="pointer-events-none absolute inset-x-0 top-3 z-20 flex justify-center px-4">
      <div
        className={
          "pointer-events-auto flex max-w-xl items-start gap-2.5 rounded-xl px-4 py-2.5 text-[12.5px] shadow-sm " +
          (status === "failed" ? "bg-red-soft text-red-ink ring-1 ring-red/30" : "bg-surface text-ink-2 ring-1 ring-line")
        }
      >
        <AlertTriangle className="mt-0.5 size-4 shrink-0" />
        <p>
          {status === "failed" ? (
            <>
              <span className="font-medium">This run failed.</span> {lastError?.message ?? "See the failed steps for details."}{" "}
              Rerun a failed step to try again from there.
            </>
          ) : (
            <>
              <span className="font-medium">This run was cancelled.</span> Everything produced so far is kept; use the
              timeline to review it.
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function PanelSkeleton() {
  return (
    <div className="space-y-3 p-5" aria-hidden>
      <div className="lg-shimmer h-3 w-24 rounded bg-line" />
      <div className="lg-shimmer h-3 w-full rounded bg-line" />
      <div className="lg-shimmer h-3 w-4/5 rounded bg-line" />
    </div>
  );
}

function LoadErrorView({ runId }: { runId: string }) {
  const err = useRunStore((s) => s.loadError)!;
  const open = useRunStore((s) => s.open);
  const loading = useRunStore((s) => s.loading);
  return (
    <div className="grid min-h-dvh place-items-center px-6">
      <div className="max-w-md space-y-4">
        <p className="font-mono text-[12px] text-ink-4">{runId}</p>
        <h1 className="text-[22px] leading-tight font-semibold">
          {err.kind === "unreachable"
            ? "The LiveGraph API isn't answering"
            : err.kind === "not_found"
              ? "This run doesn't exist"
              : "Couldn't load this run"}
        </h1>
        <p className="text-[14px] leading-relaxed text-ink-2">
          {err.kind === "unreachable" ? (
            <>
              Tried <code className="rounded bg-sunken px-1 font-mono text-[12.5px]">{API_URL}</code>. Start it with{" "}
              <code className="rounded bg-sunken px-1 font-mono text-[12.5px]">make dev</code>, or run the UI against simulated
              events with <code className="rounded bg-sunken px-1 font-mono text-[12.5px]">NEXT_PUBLIC_MOCK=1</code>.
            </>
          ) : err.kind === "not_found" ? (
            "The link may be from a database that was reset. Start a new run from the home page."
          ) : (
            err.message
          )}
        </p>
        <div className="flex gap-2">
          {err.kind !== "not_found" ? (
            <Button variant="primary" onClick={() => open(runId)} loading={loading}>
              {!loading ? <RefreshCw className="size-3.5" /> : null}
              Try again
            </Button>
          ) : null}
          <Link
            href="/"
            className="inline-flex h-9 items-center gap-2 rounded-lg px-3.5 text-[13.5px] font-medium text-ink-2 hover:bg-sunken hover:text-ink"
          >
            <ArrowLeft className="size-3.5" />
            New run
          </Link>
        </div>
      </div>
    </div>
  );
}
