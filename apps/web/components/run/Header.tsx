"use client";

import { Check, Link2, Pause, Play, Square } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { useReadOnly, useRunStore, useView } from "@/lib/store";
import { RUN_STATUS_LABEL } from "../meta";
import { Button, StatusPill, cx } from "../ui";
import { Logo } from "../Logo";
import { useAction } from "./useAction";

export function Header() {
  const runId = useRunStore((s) => s.runId);
  const goal = useRunStore((s) => s.live?.run?.goal ?? "");
  const status = useView((s) => s.runStatus);
  const liveStatus = useRunStore((s) => s.live?.runStatus);
  const connection = useRunStore((s) => s.connection);
  const mock = useRunStore((s) => s.backend?.mock ?? false);
  const readOnly = useReadOnly();
  const action = useAction();
  const [copied, setCopied] = useState(false);

  const active = liveStatus === "running" || liveStatus === "planning" || liveStatus === "waiting_input";
  const finished = liveStatus === "done" || liveStatus === "failed" || liveStatus === "cancelled";
  const [confirmCancel, setConfirmCancel] = useState(false);

  return (
    <header className="flex h-14 shrink-0 items-center gap-4 border-b border-line bg-surface px-4">
      <Link href="/" className="flex shrink-0 items-center gap-2 text-ink" aria-label="LiveGraph home">
        <Logo />
      </Link>
      <span className="h-5 w-px shrink-0 bg-line" />
      <h1 className="min-w-0 flex-1 truncate text-[14px] font-medium" title={goal}>
        {goal || <span className="text-ink-4">Loading run…</span>}
      </h1>
      <StatusPill status={status} label={RUN_STATUS_LABEL[status] ?? status} />
      <ConnectionBadge state={connection} mock={mock} />
      <div className="flex shrink-0 items-center gap-1">
        {!readOnly && runId && !finished ? (
          <>
            {liveStatus === "paused" ? (
              <Button
                size="sm"
                variant="secondary"
                loading={action.pending === "resume"}
                onClick={() => action.run("resume", (b) => b.resumeRun(runId))}
              >
                {action.pending !== "resume" ? <Play className="size-3.5" /> : null}
                Resume
              </Button>
            ) : (
              <Button
                size="sm"
                variant="ghost"
                disabled={!active}
                loading={action.pending === "pause"}
                onClick={() => action.run("pause", (b) => b.pauseRun(runId))}
                title="Finish running steps, start no new ones"
              >
                {action.pending !== "pause" ? <Pause className="size-3.5" /> : null}
                Pause
              </Button>
            )}
            {confirmCancel ? (
              <>
                <Button
                  size="sm"
                  variant="danger"
                  loading={action.pending === "cancel"}
                  onClick={async () => {
                    await action.run("cancel", (b) => b.cancelRun(runId));
                    setConfirmCancel(false);
                  }}
                >
                  Stop run
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setConfirmCancel(false)}>
                  Keep going
                </Button>
              </>
            ) : (
              <Button size="sm" variant="ghost" onClick={() => setConfirmCancel(true)} title="Stop every running step">
                <Square className="size-3" />
                Cancel
              </Button>
            )}
          </>
        ) : null}
        <Button
          size="sm"
          variant="secondary"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(window.location.href);
              setCopied(true);
              setTimeout(() => setCopied(false), 1600);
            } catch {
              /* clipboard blocked */
            }
          }}
          title="Anyone with the link sees this run live"
        >
          {copied ? <Check className="size-3.5 text-green-ink" /> : <Link2 className="size-3.5" />}
          {copied ? "Copied" : "Copy link"}
        </Button>
      </div>
      {action.error ? <span className="max-w-48 truncate text-[12px] text-red-ink">{action.error}</span> : null}
    </header>
  );
}

function ConnectionBadge({ state, mock }: { state: string; mock: boolean }) {
  if (mock)
    return (
      <span className="shrink-0 rounded bg-violet-soft px-1.5 py-0.5 font-mono text-[10.5px] text-violet" title="NEXT_PUBLIC_MOCK=1: events are simulated in your browser">
        mock
      </span>
    );
  if (state === "open") return null;
  return (
    <span
      className={cx(
        "flex shrink-0 items-center gap-1.5 text-[12px]",
        state === "reconnecting" ? "text-amber-ink" : "text-ink-3",
      )}
    >
      <span className={cx("size-1.5 rounded-full", state === "reconnecting" ? "bg-amber" : "bg-ink-4")} />
      {state === "reconnecting" ? "Reconnecting…" : state === "connecting" ? "Connecting…" : "Offline"}
    </span>
  );
}
