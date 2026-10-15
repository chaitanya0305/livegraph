"use client";

import { ChevronDown, CircleAlert, CornerUpLeft } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ContextItem, ExchangeMessage, Node, RankedChunk, TraceRead } from "@/lib/protocol";
import { effectiveStatus, nodeIdByKey, stepStatus, traceFor, upstreamRequest, type ActivityStep } from "@/lib/reducer";
import { useRunStore, useView } from "@/lib/store";
import { useCitations } from "@/lib/useCitations";
import { nodeTypeMeta, stepTarget, toolMeta, toolVerb, traceSummary } from "../meta";
import { Citation, Spinner, cx } from "../ui";

/** Node types backed by a tool-using agent. */
export const TOOL_NODE_TYPES = new Set(["plan", "research", "critic", "prosecutor", "defender", "judge"]);

const MAX_CHIPS = 8;

export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "";
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

export function ActivityTab({ node }: { node: Node }) {
  const trace = useView((s) => traceFor(s, node.id));
  const status = effectiveStatus(node);
  const ref = useRef<HTMLDivElement>(null);

  // Follow new steps while the agent works (unless the reader scrolled away).
  useEffect(() => {
    if (status !== "running") return;
    const el = ref.current?.closest(".overflow-y-auto");
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, [trace, status]);

  const { roots, children } = useMemo(() => {
    const ids = new Set(trace.map((s) => s.call_id));
    const kids = new Map<string, ActivityStep[]>();
    const top: ActivityStep[] = [];
    for (const st of trace) {
      if (st.parent_call_id && ids.has(st.parent_call_id)) {
        const list = kids.get(st.parent_call_id) ?? [];
        list.push(st);
        kids.set(st.parent_call_id, list);
      } else top.push(st);
    }
    return { roots: top, children: kids };
  }, [trace]);

  if (!trace.length) {
    return (
      <div ref={ref} className="px-5 py-6 text-[13px] text-ink-3">
        {status === "running"
          ? "Working… tool calls will appear here as the agent makes them."
          : TOOL_NODE_TYPES.has(node.type)
            ? "Every search, read and rebuttal this agent makes shows up here, live, with what it found."
            : "This step answers in one pass and doesn't call tools."}
      </div>
    );
  }

  const totalMs = trace.reduce((n, s) => n + (s.parent_call_id ? 0 : s.ms ?? 0), 0);
  return (
    <div ref={ref} className="space-y-3 px-5 py-4">
      <p className="text-[12px] text-ink-3">
        {traceSummary(trace) || `${trace.length} tool calls`}
        {totalMs ? <span className="text-ink-4"> · {formatMs(totalMs)} in tools</span> : null}
      </p>
      <ol className="space-y-3">
        {roots.map((st) => (
          <StepRow key={st.call_id} step={st} childrenOf={children} nodeType={node.type} />
        ))}
      </ol>
    </div>
  );
}

function StepRow({
  step,
  childrenOf,
  nodeType,
}: {
  step: ActivityStep;
  childrenOf: Map<string, ActivityStep[]>;
  nodeType: string;
}) {
  const meta = toolMeta(step.tool);
  const Icon = meta.icon;
  const st = stepStatus(step);
  const kids = childrenOf.get(step.call_id) ?? [];
  const questions = step.exchange.filter((m) => m.kind === "question");
  const answers = step.exchange.filter((m) => m.kind !== "question");
  const foreignAgent = step.agent && step.agent !== nodeType ? step.agent : null;
  const [refOpen, setRefOpen] = useState(false);
  const upstream = upstreamRequest(step);
  if (upstream) return <UpstreamRow step={step} req={upstream} />;

  return (
    <li className="flex gap-2.5">
      <span
        className={cx(
          "mt-px grid size-6 shrink-0 place-items-center rounded-md",
          st === "error" ? "bg-red-soft text-red-ink" : "bg-sky-soft text-sky-ink",
        )}
      >
        <Icon className={cx("size-3.5", st === "running" && "animate-pulse")} strokeWidth={2} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-[12.5px] font-medium text-ink">{toolVerb(step.tool, stepTarget(step))}</span>
          {foreignAgent ? (
            <span className="shrink-0 rounded bg-sunken px-1.5 text-[10.5px] font-medium text-ink-2">
              {nodeTypeMeta(foreignAgent).label}
            </span>
          ) : null}
          <span className="ml-auto flex shrink-0 items-center gap-1.5 font-mono text-[11px] text-ink-4 tabular-nums">
            {step.step && step.budget ? (
              <span title="call number / soft budget for this tool">
                {step.step}/{step.budget}
              </span>
            ) : null}
            {st === "running" ? <Spinner className="size-3 text-sky-ink" /> : <span>{formatMs(step.ms)}</span>}
          </span>
        </div>
        {step.args_summary ? (
          <p className="mt-1 rounded bg-sunken px-1.5 py-0.5 font-mono text-[11.5px] leading-relaxed break-words text-ink-2">
            {step.args_summary}
          </p>
        ) : null}
        {st === "error" ? (
          <p className="mt-1 flex items-start gap-1 text-[12px] text-red-ink">
            <CircleAlert className="mt-0.5 size-3 shrink-0" />
            <span>{step.error || step.summary || "Tool call failed"}</span>
          </p>
        ) : step.summary ? (
          <p className="mt-1 text-[12px] leading-relaxed text-ink-2">{step.summary}</p>
        ) : null}
        <ChunkChips step={step} />
        <NodeChips ids={step.node_ids} />
        {step.ref || step.reads?.length ? (
          <div className="mt-1.5 flex flex-wrap items-center gap-1">
            {step.reads?.length ? <ReadChips reads={step.reads} /> : null}
            {step.ref ? <RefChip refId={step.ref} open={refOpen} onToggle={() => setRefOpen((o) => !o)} /> : null}
          </div>
        ) : null}
        {step.ref && refOpen ? <ContextPanel refId={step.ref} /> : null}
        {questions.length || answers.length ? (
          <div className="mt-2 space-y-1.5">
            {questions.map((m, i) => (
              <Bubble key={`q${i}`} msg={m} />
            ))}
            {kids.length ? <Nested kids={kids} childrenOf={childrenOf} nodeType={nodeType} /> : null}
            {answers.map((m, i) => (
              <Bubble key={`a${i}`} msg={m} />
            ))}
            {st === "running" && !answers.length ? (
              <p className="pl-1 text-[11.5px] text-ink-4">Waiting for {stepTarget(step) ?? "the reply"}…</p>
            ) : null}
          </div>
        ) : kids.length ? (
          <Nested kids={kids} childrenOf={childrenOf} nodeType={nodeType} />
        ) : null}
      </div>
    </li>
  );
}

function Nested({
  kids,
  childrenOf,
  nodeType,
}: {
  kids: ActivityStep[];
  childrenOf: Map<string, ActivityStep[]>;
  nodeType: string;
}) {
  return (
    <ol className="mt-2 space-y-2.5 border-l border-line pl-3">
      {kids.map((k) => (
        <StepRow key={k.call_id} step={k} childrenOf={childrenOf} nodeType={nodeType} />
      ))}
    </ol>
  );
}

function Bubble({ msg }: { msg: ExchangeMessage }) {
  const from = nodeTypeMeta(msg.from_agent);
  const FromIcon = from.icon;
  const isQ = msg.kind === "question";
  return (
    <div className={cx("rounded-lg px-2.5 py-1.5", isQ ? "mr-6 bg-sunken" : "ml-6 bg-sky-soft")}>
      <p className={cx("flex items-center gap-1 text-[10.5px] font-medium", isQ ? "text-ink-3" : "text-sky-ink")}>
        <FromIcon className="size-3" />
        {from.label} → {nodeTypeMeta(msg.to_agent).label}
      </p>
      <p className="mt-0.5 text-[12px] leading-relaxed whitespace-pre-wrap text-ink">{msg.text}</p>
    </div>
  );
}

function ChunkChips({ step }: { step: ActivityStep }) {
  const citations = useCitations();
  const ids = step.chunk_ids;
  if (!ids.length) return null;
  const byId = new Map((step.chunks ?? []).map((c, i) => [c.chunk_id, { c, i }]));
  const shown = ids.slice(0, MAX_CHIPS);
  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-1 text-[12px]">
      {shown.map((id, i) => {
        const short = String(id).slice(0, 8).toLowerCase();
        let chunk: RankedChunk | undefined = citations[short];
        const tc = byId.get(id);
        if (!chunk && tc) {
          chunk = {
            chunk_id: tc.c.chunk_id,
            document_id: tc.c.document_id,
            document_title: tc.c.title,
            source_url: null,
            text: tc.c.snippet,
            score: tc.c.score,
            trust: tc.c.trust,
            rank: tc.i + 1,
          };
        }
        return <Citation key={`${id}-${i}`} id={short} chunk={chunk} />;
      })}
      {ids.length > MAX_CHIPS ? <span className="text-[11px] text-ink-4">+{ids.length - MAX_CHIPS}</span> : null}
    </div>
  );
}

function NodeChips({ ids }: { ids: readonly string[] }) {
  const nodes = useView((s) => s.nodes);
  const select = useRunStore((s) => s.select);
  if (!ids.length) return null;
  return (
    <div className="mt-1.5 flex flex-wrap gap-1">
      {ids.map((id) => {
        const n = nodes[id];
        return (
          <button
            key={id}
            onClick={() => n && select(id)}
            disabled={!n}
            className="inline-flex h-6 max-w-full items-center gap-1 rounded-md border border-line px-2 text-[11.5px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink disabled:opacity-60"
          >
            <span className="truncate">{n?.title ?? id.slice(0, 8)}</span>
          </button>
        );
      })}
    </div>
  );
}

/** "read research-1" chips: the upstream agents a context tool read (click selects that box). */
function ReadChips({ reads }: { reads: readonly TraceRead[] }) {
  const ids = useView((s) => reads.map((r) => nodeIdByKey(s, r.node_key)).join(","));
  const select = useRunStore((s) => s.select);
  const idList = ids.split(",");
  const seen = new Set<string>();
  return (
    <>
      {reads.map((r, i) => {
        if (seen.has(r.node_key)) return null;
        seen.add(r.node_key);
        const id = idList[i] || null;
        return (
          <button
            key={r.node_key}
            onClick={() => id && select(id)}
            disabled={!id}
            title={id ? `Open ${r.node_key}` : r.node_key}
            className="inline-flex h-6 max-w-full items-center gap-1 rounded-md bg-teal/10 px-2 text-[11.5px] text-ink-2 transition-colors hover:bg-teal/20 hover:text-ink disabled:opacity-60"
          >
            <span className="text-ink-3">read</span>
            <span className="truncate font-mono text-[11px]">{r.node_key}</span>
          </button>
        );
      })}
    </>
  );
}

function RefChip({ refId, open, onToggle }: { refId: string; open: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      aria-expanded={open}
      title="Show the full stored result this agent saw"
      className={cx(
        "inline-flex h-6 items-center gap-0.5 rounded-md border px-1.5 font-mono text-[11px] transition-colors",
        open ? "border-line-strong bg-surface text-ink" : "border-line text-ink-3 hover:border-line-strong hover:text-ink",
      )}
    >
      {refId}
      <ChevronDown className={cx("size-3 transition-transform", open && "rotate-180")} />
    </button>
  );
}

const contextCache = new Map<string, Promise<ContextItem>>();

function ContextPanel({ refId }: { refId: string }) {
  const backend = useRunStore((s) => s.backend);
  const runId = useRunStore((s) => s.runId);
  const [item, setItem] = useState<ContextItem | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!backend || !runId) return;
    let alive = true;
    const key = `${runId}/${refId}`;
    let p = contextCache.get(key);
    if (!p) {
      p = backend.getContext(runId, refId);
      contextCache.set(key, p);
      p.catch(() => contextCache.delete(key));
    }
    p.then((it) => alive && setItem(it)).catch(
      (err: unknown) => alive && setError(err instanceof Error ? err.message : String(err)),
    );
    return () => {
      alive = false;
    };
  }, [backend, runId, refId]);

  return (
    <div className="mt-1.5 overflow-hidden rounded-lg border border-line bg-surface">
      {item ? (
        <>
          <p className="flex flex-wrap items-center gap-x-2 border-b border-line bg-sunken px-2.5 py-1 font-mono text-[10.5px] text-ink-3">
            <span>{item.tool}</span>
            <span>{item.agent}</span>
            {item.kind === "rebuttal" ? <span>rebuttal</span> : null}
            <span className="ml-auto">~{item.tokens_est} tok</span>
          </p>
          <pre className="max-h-72 overflow-auto px-2.5 py-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap break-words text-ink-2">
            {item.text}
          </pre>
        </>
      ) : error ? (
        <p className="px-2.5 py-2 text-[12px] text-red-ink">Couldn&apos;t load {refId}: {error}</p>
      ) : (
        <p className="flex items-center gap-2 px-2.5 py-2 text-[12px] text-ink-3">
          <Spinner className="size-3" /> Loading {refId}…
        </p>
      )}
    </div>
  );
}

/** request_upstream_work: "asked research-2 to redo: …" with its accepted / rejected state. */
function UpstreamRow({ step, req }: { step: ActivityStep; req: NonNullable<ReturnType<typeof upstreamRequest>> }) {
  const targetId = useView((s) => (req.target ? nodeIdByKey(s, req.target) : null));
  const select = useRunStore((s) => s.select);
  const tone =
    req.state === "accepted"
      ? "border-violet/30 bg-violet-soft"
      : req.state === "rejected"
        ? "border-line bg-sunken"
        : "border-violet/20 bg-violet-soft/50";
  return (
    <li className="flex gap-2.5">
      <span className="mt-px grid size-6 shrink-0 place-items-center rounded-md bg-violet-soft text-violet">
        <CornerUpLeft className={cx("size-3.5", req.state === "pending" && "animate-pulse")} strokeWidth={2} />
      </span>
      <div className={cx("min-w-0 flex-1 rounded-lg border px-2.5 py-2", tone)}>
        <div className="flex items-center gap-1.5">
          <span className={cx("truncate text-[12.5px] font-medium", req.state === "rejected" ? "text-ink-3" : "text-ink")}>
            Asked{" "}
            <button
              onClick={() => targetId && select(targetId)}
              disabled={!targetId}
              className="font-mono text-[12px] text-violet underline decoration-violet/30 underline-offset-2 hover:decoration-violet disabled:no-underline"
            >
              {req.target ?? "upstream"}
            </button>{" "}
            to redo
          </span>
          <span
            className={cx(
              "ml-auto shrink-0 rounded px-1.5 text-[10.5px] font-medium",
              req.state === "accepted" && "bg-surface text-violet",
              req.state === "rejected" && "bg-red-soft text-red-ink",
              req.state === "pending" && "text-ink-3",
            )}
          >
            {req.state === "pending" ? <Spinner className="size-3" /> : req.state}
          </span>
        </div>
        {req.instruction ? (
          <p className={cx("mt-1 text-[12px] leading-relaxed", req.state === "rejected" ? "text-ink-3" : "text-ink-2")}>
            “{req.instruction}”
          </p>
        ) : null}
        {req.state === "rejected" && req.reason ? (
          <p className="mt-1 flex items-start gap-1 text-[11.5px] text-red-ink">
            <CircleAlert className="mt-0.5 size-3 shrink-0" />
            <span>{req.reason}</span>
          </p>
        ) : req.state === "accepted" ? (
          <p className="mt-1 text-[11.5px] text-ink-3">
            {req.target ?? "It"} reruns with this after {step.agent ?? "this agent"} finishes; this box reruns only if that changes its inputs.
          </p>
        ) : null}
      </div>
    </li>
  );
}
