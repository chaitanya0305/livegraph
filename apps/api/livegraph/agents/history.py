"""History compaction: a Pydantic AI ``ProcessHistory`` capability on every agent (``AgentFactory.get``).

Before each model request, once the history is above ``HISTORY_BUDGET_TOKENS`` (estimated as chars/4), the
content of older long tool returns (oldest first, until under budget) is replaced by the one-line stub of
their context item: ``[ctx_ab12] tool(args) -> summary [chunk_id=...] (recall('ctx_ab12') for full text)``.

- The last ``HISTORY_KEEP_RECENT`` model requests stay verbatim; user prompts, retry prompts and model
  responses are never touched; parts are never dropped or reordered, so every tool call keeps its paired
  return (same ``tool_call_id``).
- Only agents that can ``recall`` (``deps.recall_enabled``) are compacted, and only returns the context store
  knows (``deps.refs``, filled by ``guard.run_tool``); the stub comes from ``deps.stubs``: no database access.
- Compaction is permanent (ProcessHistory writes the processed list back into the run) and idempotent: a
  compacted part carries ``metadata["lg_ref"]`` and is skipped from then on.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart, ToolReturnPart

from livegraph.agents.context import part_key
from livegraph.config import settings

MIN_CHARS = 600  # shorter returns are cheaper than their stub's round trip
COMPACTABLE = frozenset({
    "search_documents", "read_source", "web_search", "fetch_page", "list_documents", "ask_rebuttal",
    "read_agent_output", "read_agent_trace", "recall", "search_context",
})


def estimate_tokens(messages: list[ModelMessage]) -> int:
    chars = 0
    for m in messages:
        for p in m.parts:
            if isinstance(p, ToolCallPart):
                chars += len(p.args_as_json_str())
                continue
            content = getattr(p, "content", None)
            if content is not None:
                chars += len(content) if isinstance(content, str) else len(str(content))
    return chars // 4


def compact(messages: list[ModelMessage], deps: Any, *, budget: int, keep_recent: int) -> list[ModelMessage]:
    refs: dict[str, str] = getattr(deps, "refs", None) or {}
    stubs: dict[str, str] = getattr(deps, "stubs", None) or {}
    if not refs or not getattr(deps, "recall_enabled", False):
        return messages
    total = estimate_tokens(messages)
    if total <= budget:
        return messages
    requests = [i for i, m in enumerate(messages) if isinstance(m, ModelRequest)]
    recent = set(requests[-keep_recent:]) if keep_recent > 0 else set()
    out = list(messages)
    for i in requests:
        if total <= budget:
            break
        if i in recent:
            continue
        parts = list(out[i].parts)
        changed = False
        for j, p in enumerate(parts):
            if total <= budget:
                break
            if not isinstance(p, ToolReturnPart) or p.tool_name not in COMPACTABLE:
                continue
            if not isinstance(p.content, str) or len(p.content) <= MIN_CHARS:
                continue
            if p.metadata is not None and (not isinstance(p.metadata, dict) or p.metadata.get("lg_ref")):
                continue  # already compacted (or foreign metadata we must not clobber)
            ref = refs.get(part_key(p.tool_call_id, p.content))
            stub = stubs.get(ref) if ref else None
            if stub is None:
                continue
            parts[j] = replace(p, content=stub, metadata={**(p.metadata or {}), "lg_ref": ref})
            total -= (len(p.content) - len(stub)) // 4
            changed = True
        if changed:
            out[i] = replace(out[i], parts=parts)
    return out


async def compact_history(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
    return compact(
        messages, ctx.deps, budget=settings.history_budget_tokens, keep_recent=settings.history_keep_recent
    )


def compaction() -> ProcessHistory[Any]:
    return ProcessHistory(compact_history)
