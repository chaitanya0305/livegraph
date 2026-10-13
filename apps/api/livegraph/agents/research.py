"""Research node: answers config.question from retrieved chunks, citing chunk ids."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from livegraph.agents.base import AgentContext, cite, format_evidence, format_instructions, format_steer

TYPE = "research"
TIER = "fast"


class Finding(BaseModel):
    point: str = Field(description="A single factual finding, one or two sentences.")
    chunk_ids: list[str] = Field(default_factory=list, description="chunk_id values that support the point.")


class ResearchOutput(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list, description="What the evidence does not answer.")


Output = ResearchOutput

SYSTEM = """You are a meticulous research analyst. Answer the research question using ONLY the evidence
chunks provided. Each finding must be a concrete, checkable statement and MUST cite the exact `chunk_id`
values (copy them verbatim, full UUIDs) of the chunks that support it in `chunk_ids`. Never invent
chunk ids. If the evidence is thin or missing, say so in `gaps` rather than speculating.
Prefer chunks from pinned sources. Produce 3-7 findings when evidence allows."""

agent: Agent[None, ResearchOutput] = Agent(output_type=ResearchOutput, instructions=SYSTEM, retries=2, name="research")


def build_prompt(ctx: AgentContext) -> str:
    q = ctx.config.get("question") or ctx.title
    return (
        f"## Overall goal\n{ctx.goal}\n\n## Research question\n{q}\n\n## Evidence chunks\n"
        f"{format_evidence(ctx.evidence)}{format_instructions(ctx)}{format_steer(ctx)}"
    )


def render(out: ResearchOutput) -> str:
    s = ""
    if out.findings:
        s += "**Findings**\n"
        for f in out.findings:
            s += f"\n- {f.point}{cite(f.chunk_ids)}"
    if out.gaps:
        s += "\n\n**Gaps**\n"
        for g in out.gaps:
            s += f"\n- {g}"
    return s


def sanitize(out: ResearchOutput, valid_ids: set[str]) -> ResearchOutput:
    for f in out.findings:
        f.chunk_ids = [c for c in f.chunk_ids if c in valid_ids]
    return out


def parse_edit(text: str, prev: dict) -> dict:
    return {**prev, "edited_markdown": text}
