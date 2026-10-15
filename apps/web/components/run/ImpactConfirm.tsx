"use client";

import { Info } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { describeImpact, toPreview } from "@/lib/impact";
import type { ImpactAction, ImpactOut, UUID } from "@/lib/protocol";
import { useRunStore, useView } from "@/lib/store";
import { Button, Spinner, cx } from "../ui";

/**
 * Inline confirm for an action that reruns part of the graph. On mount it fetches
 * GET /nodes/{id}/impact and highlights the affected boxes on the canvas (store.impactPreview)
 * until it closes. Never uses window.confirm.
 */
export function ImpactConfirm({
  nodeId,
  action,
  confirmLabel,
  danger,
  pending,
  onConfirm,
  onCancel,
  className,
}: {
  nodeId: UUID;
  action: ImpactAction;
  confirmLabel: string;
  danger?: boolean;
  pending?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  className?: string;
}) {
  const owner = useId();
  const backend = useRunStore((s) => s.backend);
  const setPreview = useRunStore((s) => s.setImpactPreview);
  const clearPreview = useRunStore((s) => s.clearImpactPreview);
  const nodes = useView((s) => s.nodes);
  const [impact, setImpact] = useState<ImpactOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!backend) return;
    let alive = true;
    setImpact(null);
    setError(null);
    backend
      .impact(nodeId, action)
      .then((res) => {
        if (!alive) return;
        setImpact(res);
        setPreview(toPreview(owner, nodeId, res));
      })
      .catch((err: unknown) => alive && setError(err instanceof Error ? err.message : String(err)));
    return () => {
      alive = false;
      clearPreview(owner);
    };
  }, [backend, nodeId, action, owner, setPreview, clearPreview]);

  useEffect(() => {
    confirmRef.current?.focus();
  }, [impact, error]);

  const summary = impact ? describeImpact(nodes, impact) : null;

  return (
    <div
      role="group"
      aria-label="Confirm action"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          onCancel();
        }
      }}
      className={cx(
        "rounded-lg border px-3 py-2.5 text-[12.5px] leading-relaxed",
        danger ? "border-red/40 bg-red-soft/60" : "border-accent/30 bg-accent-soft/70",
        className,
      )}
    >
      {!impact && !error ? (
        <p className="flex items-center gap-2 text-ink-3">
          <Spinner className="size-3" /> Checking what this affects…
        </p>
      ) : error ? (
        <p className="text-ink-2">Couldn&apos;t preview the impact ({error}). You can still go ahead.</p>
      ) : summary ? (
        <div className="space-y-1">
          <p className="text-ink">
            {summary.rerunKeys.length ? (
              <>
                This will rerun{" "}
                <span className="font-medium">
                  {summary.rerunKeys.length} box{summary.rerunKeys.length === 1 ? "" : "es"}
                </span>
                :{" "}
                <KeyList keys={summary.rerunKeys} />
              </>
            ) : (
              summary.headline
            )}
          </p>
          {summary.skippedLine ? (
            <p className="text-red-ink">
              <span className="font-medium">{summary.skippedKeys.length} may be skipped</span>:{" "}
              <KeyList keys={summary.skippedKeys} />
            </p>
          ) : null}
          {summary.rerunKeys.length ? (
            <p className="flex items-start gap-1 text-[11.5px] text-ink-3">
              <Info className="mt-[3px] size-3 shrink-0" />
              Boxes whose input ends up unchanged are skipped automatically, so fewer may actually run.
            </p>
          ) : null}
        </div>
      ) : null}
      <div className="mt-2 flex items-center gap-1.5">
        <Button
          ref={confirmRef}
          size="sm"
          variant={danger ? "danger" : "primary"}
          className={danger ? "border border-red/40 bg-surface" : undefined}
          loading={pending}
          disabled={!impact && !error}
          onClick={onConfirm}
        >
          {confirmLabel}
        </Button>
        <Button size="sm" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

function KeyList({ keys }: { keys: string[] }) {
  return (
    <span className="font-mono text-[11.5px] text-ink-2">
      {keys.map((k, i) => (
        <span key={k + i}>
          {i ? ", " : ""}
          {k}
        </span>
      ))}
    </span>
  );
}
