"""Editor: writes the final Living Brief markdown."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from livegraph.agents.base import AgentContext, format_claims, format_instructions, format_steer

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

agent: Agent[None, EditorOutput] = Agent(output_type=EditorOutput, instructions=SYSTEM, retries=2, name="editor")


def build_prompt(ctx: AgentContext) -> str:
    inputs = "\n\n".join(f"### {p.title} ({p.key})\n{p.output_text}" for p in ctx.parents)
    return (
        f"## Goal\n{ctx.goal}\n\n## Claims (with judged confidence)\n{format_claims(ctx.claims)}\n\n"
        f"## Inputs\n{inputs or '(none)'}{format_instructions(ctx)}{format_steer(ctx)}"
    )


def render(out: EditorOutput) -> str:
    return out.markdown


def parse_edit(text: str, prev: dict) -> dict:
    return {"markdown": text}
