"""Research agent: answers config.question, starting from seeded chunks and gathering more with tools.

Tools: web_search, search_documents, read_source (+ fetch_page when FETCH_PAGE=true). Every chunk it sees is
recorded as the version's retrievals; findings may cite any of them (validated).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from livegraph.agents import validators
from livegraph.agents.base import (
    AgentContext,
    AgentFactory,
    AgentLimits,
    cite,
    format_evidence,
    format_instructions,
    format_steer,
)
from livegraph.agents.tools import context_tools, rag_tools
from livegraph.config import settings

TYPE = "research"
TIER = "fast"


class Finding(BaseModel):
    point: str = Field(description="A single factual finding, one or two sentences.")
    chunk_ids: list[str] = Field(default_factory=list, description="chunk_id values that support the point.")


class ResearchOutput(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list, description="What the evidence does not answer.")


Output = ResearchOutput

SYSTEM = """You are a meticulous research analyst. Answer the research question using ONLY evidence
chunks (<source> blocks). Each finding must be a concrete, checkable statement and MUST cite the exact
`chunk_id` values (copy them verbatim, full UUIDs) of the chunks that support it in `chunk_ids`. Never invent
chunk ids. If the evidence is thin or missing, say so in `gaps` rather than speculating.
Prefer chunks from pinned sources. Produce 3-7 findings when evidence allows."""

TOOL_SYSTEM = """

You have tools. The starting evidence is a first search of the run's documents; it may be incomplete.
- `web_search(query)` finds fresh sources on the web (use it when the documents do not cover the question).
- `search_documents(query, k)` searches the run's documents with a different angle or sub-question.
- `read_source(chunk_id)` reads a chunk in full with its neighbours when a snippet is cut off or ambiguous.
Make a few targeted calls (in parallel when independent), then write the final answer. Respect the tool
budgets: when a tool says BUDGET EXHAUSTED, stop calling it and finalize."""

LIMITS = AgentLimits(
    request_limit=12,
    timeout=240,
    budgets={
        "web_search": settings.web_search_max_per_node, "search_documents": 5, "read_source": 6, "fetch_page": 2,
    },
)

agents = AgentFactory(
    "research", ResearchOutput, SYSTEM, tool_system=TOOL_SYSTEM,
    tools=[rag_tools.WEB, rag_tools.SEARCH, rag_tools.READ, rag_tools.FETCH, *context_tools.TOOLS],
    validator=validators.research,
)
agent = agents.get()


def build_prompt(ctx: AgentContext) -> str:
    q = ctx.config.get("question") or ctx.title
    return (
        f"## Overall goal\n{ctx.goal}\n\n## Research question\n{q}\n\n## Evidence chunks\n"
        f"{format_evidence(ctx.evidence)}{format_instructions(ctx)}{format_steer(ctx)}"
    )


def cited_ids(out: ResearchOutput) -> set[str]:
    return {c for f in out.findings for c in f.chunk_ids}


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
