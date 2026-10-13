"use client";

import { useMemo } from "react";
import type { RankedChunk } from "./protocol";
import { useView } from "./store";

/** Map of chunk-id prefix (8 chars) → chunk, across every retrieval in the viewed state. */
export function useCitations(): Record<string, RankedChunk> {
  const retrievals = useView((s) => s.retrievals);
  return useMemo(() => {
    const out: Record<string, RankedChunk> = {};
    for (const chunks of Object.values(retrievals)) {
      for (const c of chunks) out[String(c.chunk_id).slice(0, 8).toLowerCase()] = c;
    }
    return out;
  }, [retrievals]);
}
