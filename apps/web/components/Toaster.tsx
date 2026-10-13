"use client";

import { AlertCircle, X } from "lucide-react";
import { useToasts } from "@/lib/toast";
import { cx } from "./ui";

export function Toaster() {
  const toasts = useToasts((s) => s.toasts);
  const dismiss = useToasts((s) => s.dismiss);
  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed inset-x-0 bottom-[84px] z-50 flex flex-col items-center gap-2 px-4"
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.tone === "error" ? "alert" : "status"}
          className={cx(
            "lg-fade-in pointer-events-auto flex max-w-md items-start gap-2.5 rounded-xl px-3.5 py-2.5 text-[13px] shadow-lg",
            t.tone === "error" ? "bg-ink text-surface" : "bg-surface text-ink ring-1 ring-line",
          )}
        >
          {t.tone === "error" ? <AlertCircle className="mt-0.5 size-4 shrink-0 text-red" /> : null}
          <span className="min-w-0 flex-1">{t.message}</span>
          <button onClick={() => dismiss(t.id)} className="text-ink-4 hover:text-surface" aria-label="Dismiss">
            <X className="size-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}
