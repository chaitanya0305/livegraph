"""Critic agent: checks a research node's output; if thin it calls request_follow_up_research (graph grows)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from livegraph.agents.base import AgentContext, AgentFactory, AgentLimits, format_instructions, format_steer
from livegraph.agents.tools import context_tools, graph_tools

TYPE = "critic"
TIER = "fast"


class CriticOutput(BaseModel):
    verdict: Literal["ok", "thin"] = Field(description="'thin' if the research is weak, unsupported or incomplete.")
    follow_up_question: str | None = Field(
        default=None, description="When thin: one focused research question that would fill the biggest gap."
    )
    reason: str = Field(default="", description="One sentence.")


Output = CriticOutput

SYSTEM = """You are a demanding research critic. Judge whether the research result adequately answers its
question with cited evidence. Answer `thin` only if there are clear gaps, few/no citations, or the findings
do not address the question; then give exactly one focused `follow_up_question`. Otherwise answer `ok`."""

TOOL_SYSTEM = """

When the verdict is `thin`, FIRST call `request_follow_up_research(question)` exactly once with your
follow-up question (it adds a research node to the graph that runs before the claims are drafted), then
return your verdict with the same `follow_up_question`. Never call it for `ok` research."""

LIMITS = AgentLimits(request_limit=4, timeout=90, budgets={"request_follow_up_research": 1})

agents = AgentFactory(
    "critic", CriticOutput, SYSTEM, tool_system=TOOL_SYSTEM, tools=[graph_tools.FOLLOW_UP, *context_tools.TOOLS]
)
agent = agents.get()


def build_prompt(ctx: AgentContext) -> str:
    research = "\n\n".join(
        f"### {p.title} ({p.key})\nQuestion: {p.config.get('question', '')}\n\n{p.output_text}" for p in ctx.parents
    )
    return f"## Goal\n{ctx.goal}\n\n## Research to critique\n{research or '(none)'}{format_instructions(ctx)}{format_steer(ctx)}"


def render(out: CriticOutput) -> str:
    s = f"**Verdict:** {out.verdict}"
    if out.reason:
        s += f"\n\n{out.reason}"
    if out.follow_up_question:
        s += f"\n\n**Follow-up:** {out.follow_up_question}"
    return s


def parse_edit(text: str, prev: dict) -> dict:
    return {**prev, "edited_markdown": text}
