"use client";

import { useCallback, useState } from "react";
import type { Backend } from "@/lib/backend";
import { useRunStore } from "@/lib/store";
import { useToasts } from "@/lib/toast";

/** Wraps a backend call with pending + error state. */
export function useAction() {
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** Resolves true on success, false on failure (error is exposed via `error`). */
  const run = useCallback(async (name: string, fn: (b: Backend) => Promise<unknown>): Promise<boolean> => {
    const backend = useRunStore.getState().backend;
    if (!backend) return false;
    setPending(name);
    setError(null);
    try {
      await fn(backend);
      return true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      useToasts.getState().push(`Couldn't ${name}: ${msg}`);
      return false;
    } finally {
      setPending(null);
    }
  }, []);
  return { run, pending, error, clearError: () => setError(null) };
}
