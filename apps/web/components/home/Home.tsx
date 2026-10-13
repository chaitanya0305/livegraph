"use client";

import { ArrowRight, FileText, Paperclip, RefreshCw, X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { API_URL, BackendUnreachableError } from "@/lib/api";
import { getBackend, MOCK } from "@/lib/backend";
import type { Document, RunSummary } from "@/lib/protocol";
import { Logo } from "../Logo";
import { RUN_STATUS_LABEL, relativeTime } from "../meta";
import { Button, Kbd, SectionLabel, StatusDot, cx } from "../ui";

const EXAMPLES = [
  "Will solid-state batteries reach mass-market EVs before 2030?",
  "Is a four-day work week net positive for software teams?",
  "How credible are claims that small modular reactors will be cost-competitive by 2035?",
];

type Upload = { name: string; status: "uploading" | "done" | "error"; doc?: Document; error?: string; key: string };

export function Home() {
  const router = useRouter();
  const [goal, setGoal] = useState("");
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const uploading = uploads.some((u) => u.status === "uploading");

  const submit = async () => {
    const g = goal.trim();
    if (!g || submitting || uploading) return;
    setSubmitting(true);
    setError(null);
    try {
      const b = await getBackend();
      const snap = await b.createRun({
        goal: g,
        workflow: "living_brief",
        document_ids: uploads.flatMap((u) => (u.doc ? [u.doc.id] : [])),
      });
      router.push(`/runs/${snap.run.id}`);
    } catch (err) {
      setSubmitting(false);
      setError(
        err instanceof BackendUnreachableError
          ? `Can't reach the API at ${API_URL}. Is it running?`
          : err instanceof Error
            ? err.message
            : String(err),
      );
    }
  };

  const addFiles = async (files: FileList | File[]) => {
    const b = await getBackend();
    for (const file of Array.from(files)) {
      const key = `${file.name}-${file.size}-${Math.random()}`;
      setUploads((u) => [...u, { name: file.name, status: "uploading", key }]);
      b.uploadDocument(file)
        .then((doc) => setUploads((u) => u.map((x) => (x.key === key ? { ...x, status: "done", doc } : x))))
        .catch((err: unknown) =>
          setUploads((u) =>
            u.map((x) =>
              x.key === key ? { ...x, status: "error", error: err instanceof Error ? err.message : "Upload failed" } : x,
            ),
          ),
        );
    }
  };

  return (
    <div className="min-h-dvh">
      <header className="mx-auto flex h-14 max-w-5xl items-center justify-between px-6">
        <Logo />
        {MOCK ? (
          <span className="rounded bg-violet-soft px-1.5 py-0.5 font-mono text-[11px] text-violet">mock mode</span>
        ) : null}
      </header>

      <main className="mx-auto max-w-5xl px-6 pt-[12vh] pb-24">
        <div className="max-w-2xl">
          <h1 className="text-[34px] leading-[1.15] font-semibold tracking-[-0.02em] text-balance">
            What should the agents figure out?
          </h1>
          <p className="mt-3 max-w-[60ch] text-[15px] leading-relaxed text-ink-2">
            A planner builds the research workflow in front of you, step by step. Every step streams its work, and you can
            steer, edit, rerun or veto any of it while it runs.
          </p>

          <form
            className="mt-8"
            onSubmit={(e) => {
              e.preventDefault();
              void submit();
            }}
          >
            <Dropzone onFiles={addFiles}>
              <div className="rounded-2xl border border-line-strong bg-surface shadow-[0_1px_2px_oklch(0.2_0.02_258/0.05),0_12px_32px_-16px_oklch(0.2_0.02_258/0.18)] transition-colors focus-within:border-accent">
                <label htmlFor="goal" className="sr-only">
                  Research goal
                </label>
                <textarea
                  id="goal"
                  ref={inputRef}
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void submit();
                    }
                  }}
                  rows={3}
                  autoFocus
                  placeholder="e.g. Is it worth migrating our analytics stack to DuckDB this year?"
                  className="block w-full resize-none rounded-t-2xl bg-transparent focus-visible:outline-none px-5 pt-4 pb-2 text-[16px] leading-relaxed outline-none placeholder:text-ink-4"
                />
                {uploads.length ? (
                  <ul className="flex flex-wrap gap-1.5 px-4 pb-2">
                    {uploads.map((u) => (
                      <li
                        key={u.key}
                        className={cx(
                          "inline-flex h-7 max-w-64 items-center gap-1.5 rounded-md px-2 text-[12px]",
                          u.status === "error" ? "bg-red-soft text-red-ink" : "bg-sunken text-ink-2",
                        )}
                        title={u.error}
                      >
                        <FileText className={cx("size-3.5 shrink-0", u.status === "uploading" && "animate-pulse")} />
                        <span className="truncate">{u.name}</span>
                        {u.status === "uploading" ? <span className="text-ink-4">uploading</span> : null}
                        {u.status === "error" ? <span>failed</span> : null}
                        <button
                          type="button"
                          onClick={() => setUploads((xs) => xs.filter((x) => x.key !== u.key))}
                          className="text-ink-4 hover:text-ink"
                          aria-label={`Remove ${u.name}`}
                        >
                          <X className="size-3" />
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : null}
                <div className="flex items-center gap-2 border-t border-line px-3 py-2.5">
                  <FilePicker onFiles={addFiles} />
                  <span className="hidden text-[12px] text-ink-4 sm:inline">Drop .pdf, .md or .txt files to ground the research</span>
                  <span className="ml-auto hidden items-center gap-1 text-[11.5px] text-ink-4 sm:flex">
                    <Kbd>Enter</Kbd> to start
                  </span>
                  <Button type="submit" variant="primary" loading={submitting} disabled={!goal.trim() || uploading}>
                    Start run
                    {!submitting ? <ArrowRight className="size-3.5" /> : null}
                  </Button>
                </div>
              </div>
            </Dropzone>
            {error ? <p className="mt-3 text-[13px] text-red-ink">{error}</p> : null}
          </form>

          <div className="mt-5 flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => {
                  setGoal(ex);
                  inputRef.current?.focus();
                }}
                className="rounded-full border border-line px-3 py-1 text-left text-[12.5px] text-ink-2 transition-colors hover:border-line-strong hover:bg-surface hover:text-ink"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>

        <RecentRuns />
      </main>
    </div>
  );
}

function Dropzone({ onFiles, children }: { onFiles: (f: FileList) => void; children: React.ReactNode }) {
  const [over, setOver] = useState(false);
  const depth = useRef(0);
  return (
    <div
      className="relative"
      onDragEnter={(e) => {
        if (!e.dataTransfer.types.includes("Files")) return;
        depth.current++;
        setOver(true);
      }}
      onDragLeave={() => {
        depth.current = Math.max(0, depth.current - 1);
        if (depth.current === 0) setOver(false);
      }}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes("Files")) e.preventDefault();
      }}
      onDrop={(e) => {
        e.preventDefault();
        depth.current = 0;
        setOver(false);
        if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files);
      }}
    >
      {children}
      {over ? (
        <div className="lg-fade-in pointer-events-none absolute inset-0 grid place-items-center rounded-2xl border-2 border-dashed border-accent bg-accent-soft/90 text-[14px] font-medium text-accent-ink">
          Drop to attach as sources
        </div>
      ) : null}
    </div>
  );
}

function FilePicker({ onFiles }: { onFiles: (f: FileList) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={ref}
        type="file"
        multiple
        accept=".pdf,.md,.txt,text/plain,text/markdown,application/pdf"
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) onFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <Button type="button" size="sm" variant="ghost" onClick={() => ref.current?.click()}>
        <Paperclip className="size-3.5" />
        Attach
      </Button>
    </>
  );
}

function RecentRuns() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const b = await getBackend();
      const r = await b.listRuns();
      setRuns([...r].sort((a, b2) => b2.created_at.localeCompare(a.created_at)));
    } catch (e) {
      setErr(e instanceof BackendUnreachableError ? `The API at ${API_URL} isn't reachable.` : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <section className="mt-20 max-w-3xl">
      <SectionLabel>Recent runs</SectionLabel>
      {err ? (
        <div className="mt-3 flex items-center gap-3 text-[13px] text-ink-2">
          <span className="size-1.5 rounded-full bg-red" />
          {err}
          <Button size="sm" variant="ghost" onClick={load}>
            <RefreshCw className="size-3.5" />
            Retry
          </Button>
        </div>
      ) : loading && !runs ? (
        <div className="mt-3 space-y-2" aria-hidden>
          {[0, 1, 2].map((i) => (
            <div key={i} className="lg-shimmer h-10 rounded-lg bg-sunken" />
          ))}
        </div>
      ) : !runs?.length ? (
        <p className="mt-3 text-[13px] text-ink-3">No runs yet. Your first one will show up here, and its link can be shared.</p>
      ) : (
        <ul className="mt-2 divide-y divide-line">
          {runs.slice(0, 12).map((r) => (
            <li key={r.id}>
              <Link
                href={`/runs/${r.id}`}
                className="group -mx-3 flex items-center gap-4 rounded-lg px-3 py-3 transition-colors hover:bg-surface"
              >
                <StatusDot status={r.status} />
                <span className="min-w-0 flex-1 truncate text-[14px] text-ink">{r.goal}</span>
                <span className="w-28 shrink-0 text-[12px] text-ink-3">{RUN_STATUS_LABEL[r.status] ?? r.status}</span>
                <span className="w-20 shrink-0 text-right text-[12px] text-ink-4">{relativeTime(r.created_at)}</span>
                <ArrowRight className="size-3.5 shrink-0 text-ink-4 opacity-0 transition-opacity group-hover:opacity-100" />
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
