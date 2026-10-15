"""Shared prompts and tool set for prosecutor/defender (also used for the judge's ask_rebuttal)."""

from __future__ import annotations

from typing import Any

from livegraph.agents.base import (
    AgentContext,
    AgentLimits,
    format_claims,
    format_evidence,
    format_instructions,
    format_steer,
)
from livegraph.agents.tools import context_tools, graph_tools, rag_tools

TOOLS = [rag_tools.SEARCH, rag_tools.WEB, rag_tools.READ, *context_tools.TOOLS, graph_tools.UPSTREAM]

TOOL_SYSTEM = """

You have tools to go beyond the evidence you were given: `search_documents(query, k)` for the run's
documents, `web_search(query)` for fresh sources, `read_source(chunk_id)` to read a chunk in full. Use a few
targeted calls where they would change your argument, then answer. Cite any chunk you use. When a tool says
BUDGET EXHAUSTED, stop calling it and finalize. If an upstream input is missing something decisive,
`request_upstream_work(target_node_key, instruction)` asks that agent to redo its work after you answer."""

LIMITS = AgentLimits(request_limit=8, timeout=300, budgets={"web_search": 1, "search_documents": 4, "read_source": 3})


def build_debate_prompt(ctx: AgentContext) -> str:
    return (
        f"## Goal\n{ctx.goal}\n\n## Claims\n{format_claims(ctx.claims)}\n\n## Evidence chunks\n"
        f"{format_evidence(ctx.evidence)}{format_instructions(ctx)}{format_steer(ctx)}"
    )


def build_rebuttal_prompt(ctx: AgentContext | None, side: str, claim: dict[str, Any], question: str) -> str:
    """Prompt for one ask_rebuttal answer: just the claim, the evidence and the judge's question."""
    role = "against" if side == "prosecutor" else "for"
    goal = ctx.goal if ctx else ""
    evidence = format_evidence(ctx.evidence if ctx else [], limit=12)
    mine = ""
    if ctx:
        for p in ctx.parents_of(side):
            mine += f"\n\n## Your earlier arguments\n{p.output_text}"
    return (
        f"## Goal\n{goal}\n\n## Claims\n- {claim['key']}: {claim['text']}{mine}\n\n## Evidence chunks\n{evidence}"
        f"\n\n## The judge asks you (argue {role} {claim['key']}; answer with exactly one item for {claim['key']})\n"
        f"{question}"
    )
