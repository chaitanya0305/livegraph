"use client";

import { useMemo } from "react";
import type { RankedChunk } from "./protocol";
import { useView } from "./store";

/** Map of chunk-id prefix (8 chars) → chunk, across every retrieval (and live tool result) in the viewed state. */
export function useCitations(): Record<string, RankedChunk> {
  const retrievals = useView((s) => s.retrievals);
  const provisional = useView((s) => s.provisional);
  return useMemo(() => {
    const out: Record<string, RankedChunk> = {};
    // provisional first so finished retrievals (with final ranks) win
    for (const chunks of [...Object.values(provisional), ...Object.values(retrievals)]) {
      for (const c of chunks) out[String(c.chunk_id).slice(0, 8).toLowerCase()] = c;
    }
    return out;
  }, [retrievals, provisional]);
}
