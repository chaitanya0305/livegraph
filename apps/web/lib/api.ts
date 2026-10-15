/** Typed REST client for the LiveGraph API (see docs/CONTRACT.md §REST API). */
import type {
  ContextItem,
  CreateRunBody,
  ImpactAction,
  ImpactOut,
  Node,
  NodePatch,
  Document,
  EventEnvelope,
  NodeVersion,
  RunSnapshot,
  RunSummary,
  Trust,
  UUID,
} from "./protocol";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export function wsUrl(path: string): string {
  const u = new URL(API_URL + path);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  return u.toString();
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Thrown when the backend can't be reached at all (network error / CORS / down). */
export class BackendUnreachableError extends Error {
  constructor(cause?: unknown) {
    super(`Can't reach the LiveGraph API at ${API_URL}`);
    this.name = "BackendUnreachableError";
    this.cause = cause;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
        ...init.headers,
      },
      cache: "no-store",
    });
  } catch (err) {
    throw new BackendUnreachableError(err);
  }
  const text = await res.text();
  const body: unknown = text ? safeJson(text) : null;
  if (!res.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText;
    throw new ApiError(detail || `HTTP ${res.status}`, res.status, body);
  }
  return body as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

const json = (data: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(data) });

export const api = {
  health: () => request<{ ok: boolean }>("/health"),

  listRuns: () => request<RunSummary[]>("/runs"),
  createRun: (body: CreateRunBody) => request<RunSnapshot>("/runs", json({ workflow: "living_brief", ...body })),
  getRun: (id: UUID) => request<RunSnapshot>(`/runs/${id}`),
  getEvents: (id: UUID, since = 0) => request<EventEnvelope[]>(`/runs/${id}/events?since=${since}`),
  pauseRun: (id: UUID) => request<RunSummary>(`/runs/${id}/pause`, { method: "POST" }),
  resumeRun: (id: UUID) => request<RunSummary>(`/runs/${id}/resume`, { method: "POST" }),
  cancelRun: (id: UUID) => request<RunSummary>(`/runs/${id}/cancel`, { method: "POST" }),

  listVersions: (nodeId: UUID) => request<NodeVersion[]>(`/nodes/${nodeId}/versions`),
  steer: (nodeId: UUID, text: string) => request<unknown>(`/nodes/${nodeId}/steer`, json({ text })),
  edit: (nodeId: UUID, output_text: string) =>
    request<NodeVersion>(`/nodes/${nodeId}/edit`, json({ output_text })),
  rerun: (nodeId: UUID) => request<unknown>(`/nodes/${nodeId}/rerun`, { method: "POST" }),
  restore: (nodeId: UUID, version_id: UUID) =>
    request<unknown>(`/nodes/${nodeId}/restore`, json({ version_id })),
  answer: (nodeId: UUID, answer: string) => request<unknown>(`/nodes/${nodeId}/input`, json({ answer })),
  veto: (nodeId: UUID) => request<unknown>(`/nodes/${nodeId}`, { method: "DELETE" }),
  /** Read-only preview of the boxes an action would rerun (upper bound). */
  impact: (nodeId: UUID, action: ImpactAction) =>
    request<ImpactOut>(`/nodes/${nodeId}/impact?action=${encodeURIComponent(action)}`),
  patchNode: (nodeId: UUID, body: NodePatch) =>
    request<Node>(`/nodes/${nodeId}`, { method: "PATCH", body: JSON.stringify(body) }),
  getContext: (runId: UUID, ref: string) =>
    request<ContextItem>(`/runs/${runId}/context/${encodeURIComponent(ref)}`),

  uploadDocument: (file: File, runId?: UUID) => {
    const fd = new FormData();
    fd.append("file", file);
    if (runId) fd.append("run_id", runId);
    return request<Document>("/documents", { method: "POST", body: fd });
  },
  listDocuments: (runId?: UUID) =>
    request<Document[]>(`/documents${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`),
  setTrust: (documentId: UUID, trust: Trust) =>
    request<Document>(`/documents/${documentId}`, { method: "PATCH", body: JSON.stringify({ trust }) }),
};

export type Api = typeof api;
