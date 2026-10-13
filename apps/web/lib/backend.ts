/**
 * The UI talks to a `Backend`: either the real REST + WebSocket client, or (when
 * NEXT_PUBLIC_MOCK=1) the in-browser fake event generator in lib/mock.ts.
 */
import { api, type Api } from "./api";
import { subscribeRun, type SubscribeHandlers, type Subscription } from "./socket";
import type { UUID } from "./protocol";

export interface Backend extends Api {
  subscribe: (runId: UUID, since: number, handlers: SubscribeHandlers) => Subscription;
  mock: boolean;
}

export const MOCK = process.env.NEXT_PUBLIC_MOCK === "1" || process.env.NEXT_PUBLIC_MOCK === "true";

const realBackend: Backend = { ...api, subscribe: subscribeRun, mock: false };

let mockBackend: Backend | null = null;

export async function getBackend(): Promise<Backend> {
  if (!MOCK) return realBackend;
  if (!mockBackend) {
    const { createMockBackend } = await import("./mock");
    mockBackend = createMockBackend();
  }
  return mockBackend;
}
