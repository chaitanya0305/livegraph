/**
 * Run WebSocket with resume (`since=lastSeq`) and exponential reconnect.
 * The server replays every persisted event with seq > since, then streams live.
 */
import { wsUrl } from "./api";
import type { EventEnvelope, UUID } from "./protocol";

export type ConnectionState = "connecting" | "open" | "reconnecting" | "closed";

export interface SubscribeHandlers {
  onEvent: (e: EventEnvelope) => void;
  onState?: (s: ConnectionState, info?: { attempt: number; retryInMs?: number }) => void;
}

export interface Subscription {
  close: () => void;
}

export interface BackoffOptions {
  baseMs?: number;
  maxMs?: number;
}

/** Exponential backoff with full jitter on the upper half. Exported for tests. */
export function backoffDelay(attempt: number, { baseMs = 500, maxMs = 10_000 }: BackoffOptions = {}): number {
  const exp = Math.min(maxMs, baseMs * 2 ** Math.max(0, attempt));
  return Math.round(exp / 2 + Math.random() * (exp / 2));
}

export function subscribeRun(runId: UUID, since: number, handlers: SubscribeHandlers): Subscription {
  let lastSeq = since;
  let attempt = 0;
  let ws: WebSocket | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let closed = false;

  const connect = () => {
    if (closed) return;
    handlers.onState?.(attempt === 0 ? "connecting" : "reconnecting", { attempt });
    const socket = new WebSocket(wsUrl(`/api/runs/${runId}/ws?since=${lastSeq}`));
    ws = socket;

    socket.onopen = () => {
      attempt = 0;
      handlers.onState?.("open", { attempt });
    };
    socket.onmessage = (msg) => {
      let data: unknown;
      try {
        data = JSON.parse(typeof msg.data === "string" ? msg.data : "");
      } catch {
        return;
      }
      const batch = Array.isArray(data) ? data : [data];
      for (const raw of batch) {
        if (!raw || typeof raw !== "object" || !("type" in raw)) continue;
        const e = raw as EventEnvelope;
        if (e.seq !== null) {
          if (e.seq <= lastSeq) continue; // duplicate after resume
          lastSeq = e.seq;
        }
        handlers.onEvent(e);
      }
    };
    socket.onclose = () => {
      if (ws !== socket) return;
      ws = null;
      if (closed) return;
      const delay = backoffDelay(attempt++);
      handlers.onState?.("reconnecting", { attempt, retryInMs: delay });
      timer = setTimeout(connect, delay);
    };
    socket.onerror = () => {
      // onclose follows; reconnect handled there.
    };
  };

  connect();

  return {
    close: () => {
      closed = true;
      if (timer) clearTimeout(timer);
      const s = ws;
      ws = null;
      s?.close();
      handlers.onState?.("closed", { attempt });
    },
  };
}
