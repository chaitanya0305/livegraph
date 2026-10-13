"use client";

import { forwardRef, memo, useMemo, type ButtonHTMLAttributes, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import type { RankedChunk } from "@/lib/protocol";
import remarkGfm from "remark-gfm";
import { statusTone } from "./meta";

export function cx(...xs: (string | false | null | undefined)[]) {
  return xs.filter(Boolean).join(" ");
}

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

const VARIANT: Record<Variant, string> = {
  primary: "bg-accent text-white hover:bg-accent-ink active:translate-y-px disabled:bg-ink-4",
  secondary:
    "bg-surface text-ink border border-line-strong hover:bg-sunken active:translate-y-px disabled:text-ink-4 disabled:hover:bg-surface",
  ghost: "text-ink-2 hover:bg-sunken hover:text-ink disabled:text-ink-4 disabled:hover:bg-transparent",
  danger: "text-red-ink hover:bg-red-soft disabled:text-ink-4 disabled:hover:bg-transparent",
};
const SIZE: Record<Size, string> = {
  sm: "h-7 px-2 text-[12.5px] gap-1.5 rounded-md",
  md: "h-9 px-3.5 text-[13.5px] gap-2 rounded-lg",
};

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size; loading?: boolean }
>(function Button({ variant = "secondary", size = "md", loading, className, children, disabled, ...rest }, ref) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cx(
        "inline-flex shrink-0 items-center justify-center font-medium whitespace-nowrap transition-colors duration-150 select-none disabled:cursor-not-allowed",
        VARIANT[variant],
        SIZE[size],
        className,
      )}
      {...rest}
    >
      {loading ? <Spinner /> : null}
      {children}
    </button>
  );
});

export function Spinner({ className }: { className?: string }) {
  return (
    <svg className={cx("size-3.5 animate-spin", className)} viewBox="0 0 16 16" fill="none" aria-hidden>
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" />
      <path d="M14 8a6 6 0 0 0-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function StatusPill({ status, label, className }: { status: string; label: string; className?: string }) {
  const tone = statusTone(status);
  return (
    <span
      className={cx(
        "inline-flex h-6 items-center gap-1.5 rounded-full px-2.5 text-[12px] font-medium whitespace-nowrap",
        tone.chip,
        className,
      )}
    >
      <StatusDot status={status} />
      {label}
    </span>
  );
}

export function StatusDot({ status, className }: { status: string; className?: string }) {
  const tone = statusTone(status);
  return (
    <span className={cx("relative inline-block size-1.5 rounded-full", tone.dot, className)}>
      {tone.live ? <span className={cx("lg-ping absolute inset-0 rounded-full", tone.dot.replace("bg-", "text-"))} /> : null}
    </span>
  );
}

/** `[1a2b3c4d]` (first 8 chars of a chunk id) becomes a hoverable citation chip. */
const CITE_RE = /\[([0-9a-f]{8})\](?!\()/gi;

export const Markdown = memo(function Markdown({
  children,
  className,
  streaming,
  citations,
}: {
  children: string;
  className?: string;
  streaming?: boolean;
  citations?: Record<string, RankedChunk>;
}) {
  const text = useMemo(() => children.replace(CITE_RE, "[$1](#cite-$1)"), [children]);
  const components = useMemo<Components>(
    () => ({
      a({ href, children: kids, ...rest }) {
        if (href?.startsWith("#cite-")) {
          const id = href.slice(6).toLowerCase();
          return <Citation id={id} chunk={citations?.[id]} />;
        }
        return (
          <a href={href} target="_blank" rel="noreferrer" {...rest}>
            {kids}
          </a>
        );
      },
    }),
    [citations],
  );
  return (
    <div className={cx("md", streaming && "md-streaming", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
});

function Citation({ id, chunk }: { id: string; chunk?: RankedChunk }) {
  return (
    <span className="group/cite relative inline-block align-baseline">
      <span
        tabIndex={0}
        className={cx(
          "cursor-help rounded px-1 font-mono text-[0.78em] no-underline",
          chunk ? "bg-teal/12 text-teal ring-1 ring-teal/25" : "bg-sunken text-ink-3",
        )}
      >
        {chunk ? `#${chunk.rank}` : id.slice(0, 4)}
      </span>
      <span
        role="tooltip"
        className="pointer-events-none invisible absolute bottom-full left-1/2 z-30 mb-1.5 w-72 -translate-x-1/2 rounded-lg bg-ink p-3 text-left text-[12px] leading-relaxed text-surface opacity-0 shadow-xl transition-opacity duration-150 group-focus-within/cite:visible group-focus-within/cite:opacity-100 group-hover/cite:visible group-hover/cite:opacity-100"
      >
        {chunk ? (
          <>
            <span className="block font-medium">{chunk.document_title}</span>
            <span className="mt-1 line-clamp-5 block text-ink-4">{chunk.text}</span>
            <span className="mt-1.5 block font-mono text-[10.5px] text-ink-4">
              score {chunk.score.toFixed(2)}
              {chunk.trust !== "normal" ? ` · ${chunk.trust}` : ""}
            </span>
          </>
        ) : (
          <span className="font-mono">source {id}</span>
        )}
      </span>
    </span>
  );
}

export function SectionLabel({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <h3 className="text-[11.5px] font-medium tracking-[0.06em] text-ink-3 uppercase">{children}</h3>
      {right}
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-line bg-surface px-1 font-mono text-[10.5px] text-ink-3">{children}</kbd>
  );
}
