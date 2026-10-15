"""Judge tool ``ask_rebuttal``: the prosecutor/defender agent answers one pointed question (agent-as-tool).

The nested run shares the judge's evidence accumulator and trace (its own tool calls appear with
``agent=<side>`` and ``parent_call_id=<this call>``), rolls its usage into the judge's, and is capped at a
few requests. Question and answer are emitted as ``agent.message`` events and stored on the judge version
(``JudgeOutput.rebuttals``), so they are replayable and part of what the editor sees. No graph nodes.
"""

from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext, Tool
from pydantic_ai.usage import UsageLimits

from livegraph.agents.deps import NodeDeps
from livegraph.agents.guard import ToolOut, run_tool
from livegraph.config import settings, tools_enabled
from livegraph.trace import truncate

REBUTTAL_REQUESTS = 4
REBUTTAL_TIMEOUT = 60.0
REBUTTAL_BUDGETS = {
    "search_documents": 2, "read_source": 2, "web_search": 0,
    # context tools: the debater may look back at stored results, not browse the graph
    "list_agent_runs": 0, "read_agent_output": 0, "read_agent_trace": 0, "recall": 2, "search_context": 1,
}
MESSAGE_MAX = 2000


async def ask_rebuttal(
    ctx: RunContext[NodeDeps], claim_key: str, side: Literal["prosecutor", "defender"], question: str
) -> str:
    """Ask the prosecutor or the defender one pointed follow-up question about a single claim.

    Use it when the debate leaves the decisive point unclear. The answer comes back with its citations.

    Args:
        claim_key: the claim the question is about (C1, C2, ...).
        side: "prosecutor" (the case against) or "defender" (the case for).
        question: the specific question, e.g. "Which source contradicts the 2023 figure?".
    """

    async def impl(call_id: str) -> ToolOut:
        from livegraph.agents import SPECS
        from livegraph.agents.debate import build_rebuttal_prompt
        from livegraph.agents.models import model_for
        from livegraph.agents.runner import run_agent

        deps = ctx.deps
        if side not in ("prosecutor", "defender"):
            return ToolOut(text="error: side must be prosecutor or defender", summary="bad side", ok=False,
                           error="bad side")
        claims = {c["key"]: c for c in (deps.ctx.claims if deps.ctx else [])}
        if claim_key not in claims:
            return ToolOut(text=f"error: unknown claim {claim_key}; valid: {', '.join(claims)}", summary="unknown claim",
                           ok=False, error="unknown claim")
        if any(r["claim_key"] == claim_key and r["side"] == side for r in deps.rebuttals):
            return ToolOut(text=f"you already asked the {side} about {claim_key}; use that answer",
                           summary="already asked", ok=False, error="already asked")
        q = truncate(question, MESSAGE_MAX)
        await deps.emit("agent.message", {
            "attempt_id": deps.attempt_id, "call_id": call_id, "from_agent": deps.agent, "to_agent": side,
            "kind": "question", "text": q, "chunk_ids": [],
        })
        spec = SPECS[side]
        model = model_for(side, spec.TIER)
        child = deps.child(side, call_id, REBUTTAL_BUDGETS)
        result = await run_agent(
            spec.agents.for_model(model, with_tools=tools_enabled(side)),
            build_rebuttal_prompt(deps.ctx, side, claims[claim_key], q),
            model=model, render=spec.render, deps=child,
            limits=UsageLimits(request_limit=ctx.usage.requests + REBUTTAL_REQUESTS),
            usage=ctx.usage, timeout=REBUTTAL_TIMEOUT * settings.node_timeout_scale,
        )
        items = list(getattr(result.output, "attacks", None) or getattr(result.output, "defenses", None) or [])
        item = next((i for i in items if i.claim_key == claim_key), items[0] if items else None)
        answer = truncate(item.argument, MESSAGE_MAX) if item else "(no answer)"
        chunk_ids = [c for c in (item.chunk_ids if item else []) if deps.citable(c)]
        deps.guard()
        await deps.emit("agent.message", {
            "attempt_id": deps.attempt_id, "call_id": call_id, "from_agent": side, "to_agent": deps.agent,
            "kind": "answer", "text": answer, "chunk_ids": chunk_ids,
        })
        deps.rebuttals.append(
            {"claim_key": claim_key, "side": side, "question": q, "answer": answer, "chunk_ids": chunk_ids}
        )
        cites = "".join(f" [chunk_id={c}]" for c in chunk_ids)
        return ToolOut(
            text=f"The {side} answers on {claim_key}: {answer}{cites}", summary=truncate(f"{side}: {answer}"),
            chunks=[deps.retrieved[c] for c in chunk_ids if c in deps.retrieved],
        )

    return await run_tool(
        ctx, "ask_rebuttal", {"claim_key": claim_key, "side": side, "question": question}, impl,
        cacheable=False, pass_call_id=True, kind="rebuttal",
    )


ASK_REBUTTAL = Tool(ask_rebuttal, takes_ctx=True, sequential=True)
