"use client";

import { Pencil } from "lucide-react";
import { useState } from "react";
import type { Node, NodePatch } from "@/lib/protocol";
import { useReadOnly } from "@/lib/store";
import { Button, SectionLabel, cx } from "../ui";
import { ImpactConfirm } from "./ImpactConfirm";
import { useAction } from "./useAction";

const str = (v: unknown) => (typeof v === "string" ? v : "");

/** The diff between the form and the node, as a PATCH body (only changed fields; "" clears). */
export function patchFor(node: Node, draft: { title: string; question: string; instructions: string }): NodePatch {
  const body: NodePatch = {};
  const title = draft.title.trim();
  if (title && title !== node.title) body.title = title;
  if (node.type === "research" && draft.question.trim() !== str(node.config.question).trim()) body.question = draft.question.trim();
  if (draft.instructions.trim() !== str(node.config.instructions).trim()) body.instructions = draft.instructions.trim();
  return body;
}

/**
 * A box's setup (research question, instructions) with an inline editor saved via PATCH /nodes/{id}.
 * Saving first previews which boxes rerun (action "config").
 */
export function ConfigCard({ node }: { node: Node }) {
  const readOnly = useReadOnly();
  const action = useAction();
  const [editing, setEditing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [draft, setDraft] = useState({ title: "", question: "", instructions: "" });
  const isResearch = node.type === "research";
  const question = str(node.config.question);
  const instructions = str(node.config.instructions);
  // The planner's instructions hold the run goal; not worth repeating here.
  const showInstructions = !!instructions && node.type !== "plan";
  const editable = !readOnly && !node.removed;

  if (!editing && !isResearch && !showInstructions) {
    return editable ? (
      <button
        onClick={() => start()}
        className="inline-flex items-center gap-1 text-[11.5px] text-ink-4 transition-colors hover:text-ink"
      >
        <Pencil className="size-3" /> Edit title or instructions
      </button>
    ) : null;
  }

  function start() {
    setDraft({ title: node.title, question, instructions });
    setConfirming(false);
    setEditing(true);
  }

  const body = patchFor(node, draft);
  const changed = Object.keys(body).length > 0;
  const save = async () => {
    const ok = await action.run("save", (b) => b.patchNode(node.id, body));
    if (ok) {
      setConfirming(false);
      setEditing(false);
    }
  };

  if (!editing) {
    return (
      <div className="group rounded-lg bg-sunken px-3 py-2">
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1 space-y-1.5">
            {isResearch ? (
              <div>
                <SectionLabel>Question</SectionLabel>
                <p className={cx("mt-0.5 text-[13px] leading-snug", question ? "text-ink" : "text-ink-3")}>
                  {question || node.title}
                </p>
              </div>
            ) : null}
            {showInstructions ? (
              <div>
                <SectionLabel>Instructions</SectionLabel>
                <p className="mt-0.5 line-clamp-3 text-[12.5px] leading-snug text-ink-2">{instructions}</p>
              </div>
            ) : null}
          </div>
          {editable ? (
            <Button size="sm" variant="ghost" onClick={start} title="Change this box's setup; it and what depends on it recompute">
              <Pencil className="size-3" />
              Edit
            </Button>
          ) : null}
        </div>
      </div>
    );
  }

  const field =
    "w-full rounded-md border border-line-strong bg-surface px-2.5 py-1.5 text-[13px] leading-snug outline-none focus:border-accent";
  return (
    <form
      className="space-y-2.5 rounded-lg border border-line bg-panel p-3"
      onSubmit={(e) => {
        e.preventDefault();
        if (changed) setConfirming(true);
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape" && !confirming) setEditing(false);
      }}
    >
      <label className="block space-y-1">
        <FieldLabel>Title</FieldLabel>
        <input
          value={draft.title}
          onChange={(e) => (setDraft({ ...draft, title: e.target.value }), setConfirming(false))}
          className={field}
          autoFocus={!isResearch}
        />
      </label>
      {isResearch ? (
        <label className="block space-y-1">
          <FieldLabel>Question</FieldLabel>
          <textarea
            value={draft.question}
            rows={2}
            onChange={(e) => (setDraft({ ...draft, question: e.target.value }), setConfirming(false))}
            placeholder="Empty: the title is used as the question"
            className={cx(field, "resize-y")}
            autoFocus
          />
        </label>
      ) : null}
      <label className="block space-y-1">
        <FieldLabel>Instructions</FieldLabel>
        <textarea
          value={draft.instructions}
          rows={2}
          onChange={(e) => (setDraft({ ...draft, instructions: e.target.value }), setConfirming(false))}
          placeholder="Optional extra guidance for this agent"
          className={cx(field, "resize-y")}
        />
      </label>
      {confirming && changed ? (
        <ImpactConfirm
          nodeId={node.id}
          action="config"
          confirmLabel="Save and rerun"
          pending={action.pending === "save"}
          onCancel={() => setConfirming(false)}
          onConfirm={() => void save()}
        />
      ) : (
        <div className="flex items-center gap-2">
          <Button type="submit" size="sm" variant="primary" disabled={!changed}>
            Save
          </Button>
          <Button type="button" size="sm" variant="ghost" onClick={() => setEditing(false)}>
            Cancel
          </Button>
          <span className="ml-auto text-[11.5px] text-ink-4">Changing a field back reuses the old result</span>
        </div>
      )}
      {action.error ? <p className="text-[12px] text-red-ink">{action.error}</p> : null}
    </form>
  );
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <span className="block text-[11.5px] font-medium tracking-[0.06em] text-ink-3 uppercase">{children}</span>;
}
