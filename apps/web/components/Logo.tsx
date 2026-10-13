export function Logo({ size = 22 }: { size?: number }) {
  return (
    <span className="flex items-center gap-2">
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
        <rect x="1.5" y="3" width="7" height="6" rx="1.6" fill="var(--ink)" />
        <rect x="15.5" y="3" width="7" height="6" rx="1.6" fill="var(--ink)" />
        <rect x="8.5" y="15" width="7" height="6" rx="1.6" fill="var(--accent)" />
        <path d="M5 9v2.5a3 3 0 0 0 3 3h0M19 9v2.5a3 3 0 0 1-3 3h0" stroke="var(--ink-3)" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      <span className="text-[14.5px] font-semibold tracking-[-0.01em]">LiveGraph</span>
    </span>
  );
}
