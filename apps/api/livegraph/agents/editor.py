"""Editor: writes the final Living Brief markdown."""

from __future__ import annotations

from pydantic import BaseModel, Field

from livegraph.agents import validators
from livegraph.agents.base import AgentContext, AgentFactory, AgentLimits, format_claims, format_instructions, format_steer
from livegraph.agents.tools import context_tools

TYPE = "editor"
TIER = "smart"


class EditorOutput(BaseModel):
    markdown: str = Field(default="", description="The complete brief in markdown.")


Output = EditorOutput

SYSTEM = """You are the Editor. Write a crisp, well-structured brief in markdown that answers the goal.
Rules: use only the claims provided; EXCLUDE struck claims entirely; state high-confidence claims (>= 0.75)
plainly; hedge medium-confidence claims (0.4-0.75) with words like "likely"/"some evidence suggests";
mention low-confidence claims (< 0.4) only as open questions. Reference claims inline as (C1), (C2).
Start with a one-paragraph bottom line, then sections, then "Open questions"."""

# no judge ran (DISABLED_NODE_TYPES ablation): claims carry no confidence
UNJUDGED = (
    "Claims (not judged: no confidence scores; treat each as supported by its cited evidence, state it plainly "
    "unless its wording is itself uncertain, and leave the confidence-based hedging rules aside)"
)

LIMITS = AgentLimits(request_limit=3, timeout=300)

# context tools only run with CONTEXT_TOOLS_SINGLE_SHOT=true (otherwise this stays a validated single shot)
agents = AgentFactory(
    "editor", EditorOutput, SYSTEM, tools=context_tools.TOOLS, validator=validators.editor
)
agent = agents.get()


def build_prompt(ctx: AgentContext) -> str:
    inputs = "\n\n".join(f"### {p.title} ({p.key})\n{p.output_text}" for p in ctx.parents)
    judged = any(c.get("confidence") is not None for c in ctx.claims) or not ctx.claims
    head = "Claims (with judged confidence)" if judged else UNJUDGED
    return (
        f"## Goal\n{ctx.goal}\n\n## {head}\n{format_claims(ctx.claims)}\n\n"
        f"## Inputs\n{inputs or '(none)'}{format_instructions(ctx)}{format_steer(ctx)}"
    )


def render(out: EditorOutput) -> str:
    return out.markdown


def parse_edit(text: str, prev: dict) -> dict:
    return {"markdown": text}
