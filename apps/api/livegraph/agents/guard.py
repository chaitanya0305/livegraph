"""Guarded tool execution: supersede check, soft budgets, repeat cache, error capture, tool events.

Every agent tool body is ``return await run_tool(ctx, "name", {args}, impl)``. ``run_tool``:
- raises ``CancelledError`` if the attempt was superseded (before and after the work, so a stale attempt
  never emits ``tool.result``);
- counts the call (``step``) against the tool's soft budget; past it the model gets
  ``BUDGET EXHAUSTED ...; finalize now`` instead of an exception (the run is never aborted for it);
- answers identical repeat calls from a per-agent cache (they keep the original call's ``ref``);
- turns exceptions into ``error: ...`` strings the model can react to;
- stores the result in the context store (``context_items``, own session; not for budget-exhausted calls or
  ``store=False`` tools) and registers its compaction stub, so the result can be compacted out of the
  history and recalled by ``ref`` later (agents/context.py, agents/history.py);
- emits ``tool.call`` / ``tool.result`` (with ``ref`` and ``reads``) and folds them into the attempt trace.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext

from livegraph.agents import context
from livegraph.agents.base import wrap_untrusted
from livegraph.agents.deps import NodeDeps
from livegraph.rag.schemas import RetrievedChunk
from livegraph.trace import summarize_args, truncate

__all__ = ["ToolOut", "run_tool", "wrap_untrusted", "wrap_chunks", "tool_chunk"]


@dataclass
class ToolOut:
    text: str  # what the model sees
    summary: str  # one line for the UI
    ok: bool = True
    error: str | None = None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    reads: list[dict[str, str]] = field(default_factory=list)  # [{node_key}] another agent's work was read
    ref: str | None = None  # set by the tool itself to point at an existing context item (recall); not stored


def tool_chunk(c: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": str(c.chunk_id), "document_id": str(c.document_id), "title": c.document_title,
        "score": round(float(c.score), 4), "trust": c.trust, "snippet": truncate(c.text, 160),
    }


def wrap_chunks(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(wrap_untrusted(c.model_dump(mode="json")) for c in chunks)


async def run_tool(
    ctx: RunContext[NodeDeps],
    tool: str,
    args: dict[str, Any],
    impl: Callable[..., Awaitable[ToolOut]],
    *,
    cacheable: bool = True,
    pass_call_id: bool = False,
    store: bool = True,
    kind: str = "tool_result",
) -> str:
    deps = ctx.deps
    deps.guard()
    step = deps.counts.get(tool, 0) + 1
    deps.counts[tool] = step
    budget = deps.budgets.get(tool)
    call_id = uuid.uuid4().hex[:16]  # provider ids are not unique across rounds/nested agents
    ref = context.new_ref()  # allocated up front: nested runs of this call record it as their parent_ref
    deps.call_refs[call_id] = ref
    await deps.emit(
        "tool.call",
        {
            "attempt_id": deps.attempt_id, "call_id": call_id, "tool": tool, "args": _jsonable(args),
            "args_summary": summarize_args(args), "agent": deps.agent, "parent_call_id": deps.parent_call_id,
            "step": step, "budget": budget,
        },
    )
    t0 = time.monotonic()
    key = f"{tool}:{json.dumps(args, sort_keys=True, default=str)}"
    fresh = False
    if budget is not None and step > budget:
        out = ToolOut(
            text=f"BUDGET EXHAUSTED for {tool} ({budget} calls used). Do not call {tool} again; "
            "finalize your answer now with the evidence you already have.",
            summary="budget exhausted", ok=False, error="budget exhausted",
        )
    elif cacheable and key in deps.cache:
        out = ToolOut(text=deps.cache[key], summary="repeat call (cached result)", ref=deps.cache_refs.get(key))
    else:
        fresh = True
        try:
            out = await (impl(call_id) if pass_call_id else impl())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failing tool must never kill the agent run
            out = ToolOut(text=f"error: {exc}", summary=truncate(f"error: {exc}"), ok=False, error=truncate(str(exc)))
        if out.ok and cacheable:
            deps.cache[key] = out.text
    deps.guard()
    if fresh and out.ref is None and store:
        if await context.put(deps, ref=ref, tool=tool, args=args, out=out, kind=kind):
            out.ref = ref
    if fresh and out.ref is not None and out.ok and cacheable:
        deps.cache_refs[key] = out.ref  # a repeat call answers with the same ref
    if out.ref is not None and ctx.tool_call_id is not None:
        deps.refs[context.part_key(ctx.tool_call_id, out.text)] = out.ref  # lets the history compact it
    await deps.emit(
        "tool.result",
        {
            "attempt_id": deps.attempt_id, "call_id": call_id, "ok": out.ok, "summary": truncate(out.summary),
            "error": out.error, "ms": int((time.monotonic() - t0) * 1000),
            "chunks": [tool_chunk(c) for c in out.chunks], "node_ids": out.node_ids,
            "ref": out.ref, "reads": out.reads,
        },
    )
    return out.text


def _jsonable(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in args.items():
        out[k] = truncate(v) if isinstance(v, str) else v
    return out
