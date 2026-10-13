"""Shared agent plumbing: the context handed to prompt builders, and the streaming runner.

Streaming approach: every LLM node uses a *typed* Pydantic output (tool-call structured output) and is
run with ``agent.run_stream``. ``stream_output()`` yields progressively validated partial outputs; each
partial is rendered to markdown with the node's deterministic ``render`` function and the *suffix* that
was not yet sent is emitted as a ``node.delta``. Renderers are append-only by construction (fields are
rendered in schema order, list items in order), so the live buffer the frontend appends to matches
the final ``output_text``. Whenever the rendering is not a pure extension (rare: partial-validation
rewrites the tail) we just send the new tail; the authoritative text arrives with ``node.version``.
This keeps one code path for typed outputs (needed for claims/judge side-effects) and still gives
token-level streaming, instead of streaming free text and parsing it afterwards (fragile with
real models).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model


@dataclass
class ParentOutput:
    key: str
    type: str
    title: str
    output: dict[str, Any]
    output_text: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    goal: str
    node_key: str
    node_type: str
    title: str
    config: dict[str, Any]
    steer_texts: list[str] = field(default_factory=list)
    parents: list[ParentOutput] = field(default_factory=list)
    # evidence chunks (RetrievedChunk dicts) from research ancestors / own retrieval
    evidence: list[dict[str, Any]] = field(default_factory=list)
    # current claims [{key, text, confidence, status}]
    claims: list[dict[str, Any]] = field(default_factory=list)

    def parents_of(self, type_: str) -> list[ParentOutput]:
        return [p for p in self.parents if p.type == type_]


@dataclass
class AgentResult:
    output: BaseModel
    output_text: str
    model_name: str | None
    tokens_in: int | None
    tokens_out: int | None


def short(cid: Any) -> str:
    return str(cid)[:8]


def cite(ids: list[Any] | None) -> str:
    if not ids:
        return ""
    return " " + "".join(f"[{short(i)}]" for i in ids)


def format_evidence(chunks: list[dict[str, Any]], limit: int = 24, max_chars: int = 900) -> str:
    if not chunks:
        return "(no evidence chunks available)"
    lines = []
    for c in chunks[:limit]:
        text = " ".join(str(c.get("text", "")).split())
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        src = c.get("document_title") or ""
        lines.append(f"[chunk_id={c['chunk_id']}] (source: {src}; trust: {c.get('trust', 'normal')})\n{text}")
    return "\n\n".join(lines)


def format_steer(ctx: AgentContext) -> str:
    if not ctx.steer_texts:
        return ""
    joined = "\n".join(f"- {t}" for t in ctx.steer_texts)
    return f"\n\n## User steering (highest priority — follow these instructions)\n{joined}\n"


def format_instructions(ctx: AgentContext) -> str:
    ins = (ctx.config or {}).get("instructions")
    return f"\n\n## Node instructions\n{ins}\n" if ins else ""


def format_claims(claims: list[dict[str, Any]]) -> str:
    if not claims:
        return "(no claims)"
    out = []
    for c in claims:
        conf = c.get("confidence")
        conf_s = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
        status = " [STRUCK]" if c.get("status") == "struck" else ""
        out.append(f"- {c['key']}: {c['text']}{conf_s}{status}")
    return "\n".join(out)


DeltaFn = Callable[[str], None]


async def stream_structured(
    agent: Agent[Any, Any],
    prompt: str,
    *,
    model: Model,
    render: Callable[[Any], str],
    on_delta: DeltaFn,
    deps: Any = None,
    debounce: float | None = 0.05,
) -> AgentResult:
    sent = ""
    async with agent.run_stream(prompt, model=model, deps=deps) as res:
        async for partial in res.stream_output(debounce_by=debounce):
            try:
                text = render(partial)
            except Exception:  # partial objects may be incomplete; skip unrenderable frames
                continue
            if len(text) > len(sent):
                on_delta(text[len(sent) :] if text.startswith(sent) else text[len(sent) :])
                sent = text
        output = await res.get_output()
        usage = res.usage
    final_text = render(output)
    if len(final_text) > len(sent) and final_text.startswith(sent):
        on_delta(final_text[len(sent) :])
    return AgentResult(
        output=output,
        output_text=final_text,
        model_name=getattr(model, "model_name", None),
        tokens_in=getattr(usage, "input_tokens", None),
        tokens_out=getattr(usage, "output_tokens", None),
    )


AsyncFn = Callable[..., Awaitable[Any]]
