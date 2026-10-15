"use client";

import { Ban, CornerDownLeft, Pencil, RotateCw, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { currentVersion, effectiveStatus, splitSteerText, steerAgent, traceFor } from "@/lib/reducer";
import { useReadOnly, useRunStore, useView } from "@/lib/store";
import { useCitations } from "@/lib/useCitations";
import type { Node } from "@/lib/protocol";
import { NODE_STATUS_LABEL, nodeTypeMeta } from "../meta";
import { Button, Kbd, Markdown, SectionLabel, StatusPill, cx } from "../ui";
import { ActivityTab, TOOL_NODE_TYPES } from "./ActivityTab";
import { ConfigCard } from "./ConfigCard";
import { ImpactConfirm } from "./ImpactConfirm";
import { RunOverview } from "./RunOverview";
import { SourcesTab } from "./SourcesTab";
import { useAction } from "./useAction";
import { VersionsTab } from "./VersionsTab";

type Tab = "output" | "activity" | "versions" | "sources";

export function Inspector() {
  const selectedId = useRunStore((s) => s.selectedNodeId);
  const node = useView((s) => (selectedId ? s.nodes[selectedId] : undefined));
  if (!selectedId || !node) return <RunOverview />;
  return <NodeInspector key={node.id} node={node} />;
}

function NodeInspector({ node }: { node: Node }) {
  const [tab, setTab] = useState<Tab>("output");
  const select = useRunStore((s) => s.select);
  const readOnly = useReadOnly();
  const versionsCount = useView((s) => s.versions[node.id]?.length ?? 0);
  const version = useView((s) => currentVersion(s, node.id));
  const sourcesCount = useView((s) => (version ? s.retrievals[version.id]?.length ?? 0 : 0));
  const provisionalCount = useView((s) => s.provisional[node.id]?.length ?? 0);
  const toolCount = useView((s) => traceFor(s, node.id).length);
  const status = effectiveStatus(node);
  const meta = nodeTypeMeta(node.type);
  const Icon = meta.icon;
  const action = useAction();
  const [confirming, setConfirming] = useState<"rerun" | "veto" | null>(null);

  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: "output", label: node.type === "human_input" ? "Answer" : "Output" },
  ];
  if (TOOL_NODE_TYPES.has(node.type) || toolCount) tabs.push({ id: "activity", label: "Activity", count: toolCount });
  tabs.push({ id: "versions", label: "Versions", count: versionsCount });
  const liveSources = status === "running" && provisionalCount > 0;
  if (node.type === "research" || sourcesCount || provisionalCount)
    tabs.push({ id: "sources", label: "Sources", count: liveSources ? provisionalCount : sourcesCount });

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-line px-5 pt-4 pb-3">
        <div className="flex items-center gap-2 text-[12px] text-ink-3">
          <Icon className="size-3.5" />
          <span>{meta.label}</span>
          <span className="font-mono text-ink-4">{node.key}</span>
          <button
            onClick={() => select(null)}
            className="ml-auto grid size-7 place-items-center rounded-md text-ink-3 hover:bg-sunken hover:text-ink"
            aria-label="Close inspector"
          >
            <X className="size-4" />
          </button>
        </div>
        <h2 className={cx("mt-1 text-[16px] leading-snug font-semibold", node.removed && "line-through decoration-ink-3")}>
          {node.title}
        </h2>
        <div className="mt-2.5 flex flex-wrap items-center gap-2">
          <StatusPill status={status} label={NODE_STATUS_LABEL[status]} />
          {node.created_by !== "planner" ? (
            <span className="text-[12px] text-ink-3">added by {node.created_by === "user" ? "you" : "the critic"}</span>
          ) : null}
          {version?.model ? (
            <span className="truncate font-mono text-[11px] text-ink-4" title={version.model}>
              {version.model.split("/").at(-1)}
              {version.tokens_out ? ` · ${version.tokens_out} tok` : ""}
            </span>
          ) : null}
          {!readOnly && !node.removed ? (
            <div className="ml-auto flex items-center gap-1">
              {node.type !== "human_input" ? (
                <Button
                  size="sm"
                  variant="ghost"
                  title="Run this step again, even if its inputs haven't changed"
                  loading={action.pending === "rerun"}
                  disabled={status === "running" || status === "pending"}
                  aria-pressed={confirming === "rerun"}
                  onClick={() => setConfirming((c) => (c === "rerun" ? null : "rerun"))}
                >
                  {action.pending !== "rerun" ? <RotateCw className="size-3.5" /> : null}
                  Rerun
                </Button>
              ) : null}
              <Button
                size="sm"
                variant="danger"
                aria-pressed={confirming === "veto"}
                onClick={() => setConfirming((c) => (c === "veto" ? null : "veto"))}
                title="Remove this step"
              >
                <Ban className="size-3.5" />
                Veto
              </Button>
            </div>
          ) : null}
        </div>
        {confirming && !readOnly && !node.removed ? (
          <ImpactConfirm
            key={confirming}
            className="mt-2.5"
            nodeId={node.id}
            action={confirming}
            danger={confirming === "veto"}
            confirmLabel={confirming === "veto" ? "Veto step" : "Rerun"}
            pending={action.pending === confirming}
            onCancel={() => setConfirming(null)}
            onConfirm={async () => {
              const which = confirming;
              const ok = await action.run(which, (b) => (which === "veto" ? b.veto(node.id) : b.rerun(node.id)));
              if (ok) setConfirming(null);
            }}
          />
        ) : null}
        {action.error ? <p className="mt-2 text-[12px] text-red-ink">{action.error}</p> : null}
      </div>

      <div role="tablist" className="flex gap-1 border-b border-line px-4">
        {tabs.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={cx(
              "relative -mb-px flex h-9 items-center gap-1.5 border-b-2 px-2 text-[13px] transition-colors",
              tab === t.id ? "border-ink font-medium text-ink" : "border-transparent text-ink-3 hover:text-ink",
            )}
          >
            {t.label}
            {t.count ? <span className="font-mono text-[11px] text-ink-4 tabular-nums">{t.count}</span> : null}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {tab === "output" ? <OutputTab node={node} /> : null}
        {tab === "activity" ? <ActivityTab node={node} /> : null}
        {tab === "versions" ? <VersionsTab node={node} /> : null}
        {tab === "sources" ? <SourcesTab node={node} /> : null}
      </div>

      {!readOnly && !node.removed && node.type !== "human_input" ? <SteerBox node={node} /> : null}
    </div>
  );
}

function OutputTab({ node }: { node: Node }) {
  const readOnly = useReadOnly();
  const status = effectiveStatus(node);
  const live = useView((s) => s.live[node.id]);
  const version = useView((s) => currentVersion(s, node.id));
  const reason = useView((s) => s.statusReason[node.id]);
  const allErrors = useView((s) => s.errors);
  const errors = allErrors.filter((e) => e.node_id === node.id);
  const [editing, setEditing] = useState(false);
  const [confirmEdit, setConfirmEdit] = useState(false);
  const [draft, setDraft] = useState("");
  const action = useAction();
  const scrollRef = useRef<HTMLDivElement>(null);
  const citations = useCitations();

  // Keep the newest streamed text in view.
  useEffect(() => {
    if (status !== "running") return;
    const el = scrollRef.current?.closest(".overflow-y-auto");
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 120) el.scrollTop = el.scrollHeight;
  }, [live, status]);

  if (node.type === "human_input") return <HumanInput node={node} />;

  const streaming = status === "running";
  const text = streaming ? live ?? "" : version?.output_text ?? "";
  const saveEdit = async () => {
    const ok = await action.run("edit", (b) => b.edit(node.id, draft));
    if (ok) {
      setConfirmEdit(false);
      setEditing(false);
    }
  };
  const canSave = !!draft.trim() && draft !== version?.output_text;

  return (
    <div ref={scrollRef} className="space-y-4 px-5 py-4">
      <ConfigCard node={node} />
      {status === "stale" ? (
        <Notice tone="neutral">
          Inputs changed{reason ? ` (${reason})` : ""}. This step will recompute; the output below is the previous version.
        </Notice>
      ) : null}
      {status === "failed" ? (
        <Notice tone="red">{errors.at(-1)?.message ?? reason ?? "This step failed."}</Notice>
      ) : null}
      {version?.steer_text && !streaming ? <SteeredWith text={version.steer_text} /> : null}

      {editing ? (
        <div className="space-y-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            className="min-h-[320px] w-full resize-y rounded-lg border border-line-strong bg-surface p-3 font-mono text-[12.5px] leading-relaxed outline-none focus:border-accent"
            autoFocus
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canSave) setConfirmEdit(true);
              if (e.key === "Escape") {
                if (confirmEdit) setConfirmEdit(false);
                else setEditing(false);
              }
            }}
          />
          {confirmEdit ? (
            <ImpactConfirm
              nodeId={node.id}
              action="edit"
              confirmLabel="Save as new version"
              pending={action.pending === "edit"}
              onCancel={() => setConfirmEdit(false)}
              onConfirm={() => void saveEdit()}
            />
          ) : (
            <div className="flex items-center gap-2">
              <Button variant="primary" size="sm" disabled={!canSave} onClick={() => setConfirmEdit(true)}>
                Save as new version
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setEditing(false)}>
                Cancel
              </Button>
              <span className="ml-auto text-[11.5px] text-ink-4">Downstream steps will recompute</span>
            </div>
          )}
          {action.error ? <p className="text-[12px] text-red-ink">{action.error}</p> : null}
        </div>
      ) : text ? (
        <div className="group relative">
          {!readOnly && !streaming && version ? (
            <Button
              size="sm"
              variant="secondary"
              className="float-right ml-3"
              onClick={() => {
                setDraft(version.output_text);
                setConfirmEdit(false);
                setEditing(true);
              }}
            >
              <Pencil className="size-3" />
              Edit
            </Button>
          ) : null}
          <Markdown streaming={streaming} citations={citations}>
            {text}
          </Markdown>
        </div>
      ) : (
        <EmptyOutput status={status} />
      )}
      <QueuedSteers node={node} />
    </div>
  );
}

function EmptyOutput({ status }: { status: string }) {
  if (status === "running")
    return <p className="lg-caret text-[13px] text-accent-ink">Starting</p>;
  return (
    <div className="space-y-2" aria-hidden={status === "pending"}>
      <p className="text-[13px] text-ink-3">
        {status === "pending"
          ? "This step hasn't run yet. It starts as soon as every input is done."
          : status === "skipped"
            ? "This step was skipped."
            : "No output yet."}
      </p>
    </div>
  );
}

function Notice({ tone, children }: { tone: "neutral" | "red" | "amber"; children: React.ReactNode }) {
  return (
    <div
      className={cx(
        "rounded-lg px-3 py-2 text-[12.5px] leading-relaxed",
        tone === "red" && "bg-red-soft text-red-ink",
        tone === "amber" && "bg-amber-soft text-amber-ink",
        tone === "neutral" && "bg-sunken text-ink-2",
      )}
    >
      {children}
    </div>
  );
}

function QueuedSteers({ node }: { node: Node }) {
  const steers = useView((s) => s.steers[node.id]);
  if (!steers?.length) return null;
  return (
    <div className="space-y-2 border-t border-line pt-3">
      <SectionLabel>Steering</SectionLabel>
      <ul className="space-y-1.5">
        {steers.map((s) => {
          const agent = steerAgent(s.source);
          const text = agent ? splitSteerText(s.text).text : s.text;
          return (
            <li key={s.steer_id} className="flex items-start gap-2 text-[12.5px]">
              <span
                className={cx(
                  "mt-1.5 size-1.5 shrink-0 rounded-full",
                  s.consumed ? "bg-ink-4" : agent ? "bg-violet" : "bg-accent",
                )}
              />
              <span className="min-w-0">
                <SteerSourceTag agent={agent} />{" "}
                <span className={s.consumed ? "text-ink-3" : "text-ink"}>{text}</span>
              </span>
              <span className="ml-auto shrink-0 text-[11px] text-ink-4">{s.consumed ? "applied" : "queued"}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** "from you" / "from judge" chip for a steer. */
export function SteerSourceTag({ agent }: { agent: string | null }) {
  return agent ? (
    <span className="rounded bg-violet-soft px-1 text-[10.5px] font-medium whitespace-nowrap text-violet">from {agent}</span>
  ) : (
    <span className="rounded bg-accent-soft px-1 text-[10.5px] font-medium whitespace-nowrap text-accent-ink">from you</span>
  );
}

function SteeredWith({ text }: { text: string }) {
  // A version's steer_text joins every steer it consumed; agent requests read "Request from <key>: …".
  const parts = text.split("\n").filter((t) => t.trim());
  return (
    <div className="space-y-0.5 text-[12px] text-ink-3">
      {parts.map((p, i) => {
        const { agent, text: body } = splitSteerText(p);
        return (
          <p key={i}>
            {agent ? `Asked by ${agent}: ` : "Steered by you with "}
            <span className={agent ? "text-violet" : "text-accent-ink"}>“{body}”</span>
          </p>
        );
      })}
    </div>
  );
}

function SteerBox({ node }: { node: Node }) {
  const [text, setText] = useState("");
  const [confirming, setConfirming] = useState(false);
  const action = useAction();
  const status = effectiveStatus(node);
  const submit = () => {
    if (text.trim()) setConfirming(true);
  };
  const send = async () => {
    const t = text.trim();
    if (!t) return;
    const res = await action.run("steer", (b) => b.steer(node.id, t));
    if (res) {
      setText("");
      setConfirming(false);
    }
  };
  const hint =
    status === "running"
      ? "Restarts this step with your instruction"
      : status === "done"
        ? "Reruns this step and recomputes what depends on it"
        : "Applied when this step runs";
  return (
    <form
      className="border-t border-line bg-surface px-4 py-3"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      {confirming && text.trim() ? (
        <ImpactConfirm
          className="mb-2"
          nodeId={node.id}
          action="steer"
          confirmLabel="Send steer"
          pending={action.pending === "steer"}
          onCancel={() => setConfirming(false)}
          onConfirm={() => void send()}
        />
      ) : null}
      <label htmlFor="steer" className="sr-only">
        Steer this step
      </label>
      <div className="flex items-end gap-2 rounded-lg border border-line-strong bg-surface px-3 py-2 transition-colors focus-within:border-accent">
        <textarea
          id="steer"
          value={text}
          rows={1}
          onChange={(e) => {
            setText(e.target.value);
            if (confirming) setConfirming(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={`Steer ${node.title.toLowerCase()}…`}
          className="max-h-32 min-h-[22px] flex-1 resize-none bg-transparent focus-visible:outline-none text-[13px] leading-[22px] outline-none placeholder:text-ink-4"
        />
        <Button type="submit" size="sm" variant="primary" loading={action.pending === "steer"} disabled={!text.trim() || confirming}>
          Steer
          {action.pending !== "steer" ? <CornerDownLeft className="size-3" /> : null}
        </Button>
      </div>
      <p className="mt-1.5 flex items-center gap-1.5 text-[11.5px] text-ink-4">
        {action.error ? <span className="text-red-ink">{action.error}</span> : hint}
        <span className="ml-auto">
          <Kbd>Enter</Kbd> to send
        </span>
      </p>
    </form>
  );
}

function HumanInput({ node }: { node: Node }) {
  const readOnly = useReadOnly();
  const status = effectiveStatus(node);
  const prompt = useView((s) => s.prompts[node.id]) ?? String(node.config.prompt ?? node.config.question ?? "");
  const version = useView((s) => currentVersion(s, node.id));
  const [answer, setAnswer] = useState("");
  const action = useAction();
  const answered = version?.output_text;
  const waiting = status === "waiting_input";

  return (
    <div className="space-y-4 px-5 py-4">
      <div className={cx("rounded-xl p-4", waiting ? "bg-amber-soft" : "bg-sunken")}>
        <p className={cx("text-[11.5px] font-medium", waiting ? "text-amber-ink" : "text-ink-3")}>
          {waiting ? "The agents are waiting on you" : "Question"}
        </p>
        <p className="mt-1 text-[15px] leading-snug text-ink">{prompt || "Your input is needed."}</p>
      </div>
      {waiting && !readOnly ? (
        <form
          className="space-y-2"
          onSubmit={async (e) => {
            e.preventDefault();
            if (!answer.trim()) return;
            const r = await action.run("answer", (b) => b.answer(node.id, answer.trim()));
            if (r) setAnswer("");
          }}
        >
          <textarea
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            autoFocus
            rows={4}
            placeholder="Type your answer"
            className="w-full resize-y rounded-lg border border-line-strong bg-surface p-3 text-[13.5px] outline-none focus:border-accent"
          />
          <Button type="submit" variant="primary" loading={action.pending === "answer"} disabled={!answer.trim()}>
            Send answer
          </Button>
          {action.error ? <p className="text-[12px] text-red-ink">{action.error}</p> : null}
        </form>
      ) : answered ? (
        <div>
          <SectionLabel>Your answer</SectionLabel>
          <p className="mt-1.5 text-[13.5px] leading-relaxed">{answered}</p>
        </div>
      ) : (
        <p className="text-[13px] text-ink-3">This question will be asked once the steps before it finish.</p>
      )}
    </div>
  );
}
