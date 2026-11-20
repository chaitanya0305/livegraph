"""Research agent: answers config.question, starting from seeded chunks and gathering more with tools.

Tools: web_search, search_documents, read_source, reflect (+ fetch_page when FETCH_PAGE=true). Every chunk it
sees is recorded as the version's retrievals; findings may cite any of them (validated).

With RESEARCH_LOOP=true (default) the agent researches in rounds: explore, ``reflect`` on its open gaps, then
targeted and challenge searches, ``reflect`` again; ``reflect`` ends the search when a round adds little
(agents/tools/research_tools.py). Budgets come from ``limits(config)``: the loop's budgets (or the previous
ones), scaled by RESEARCH_BUDGET_SCALE and the node's planner priority. ``finalize`` attaches
``search_stats`` (rounds, query kinds, cited documents and sites, how the search ended) for the critic and the
evaluation; they are not rendered.
"""

from __future__ import annotations

import math
from typing import Any

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
from livegraph.agents.tools import context_tools, rag_tools, research_tools
from livegraph.config import settings

TYPE = "research"
TIER = "fast"


class Finding(BaseModel):
    point: str = Field(description="A single factual finding, one or two sentences.")
    chunk_ids: list[str] = Field(default_factory=list, description="chunk_id values that support the point.")


class ResearchOutput(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list, description="What the evidence does not answer.")


class SearchStats(BaseModel):
    rounds: int = 0  # reflect calls
    stopped_by: str = ""  # "rule" (reflect stopped the search), "budget" (a tool budget ran out) or ""
    queries: dict[str, int] = Field(default_factory=dict)  # web_search calls by kind (explore/target/challenge)
    docs_seen: int = 0
    docs_cited: int = 0
    domains_cited: list[str] = Field(default_factory=list)


class ResearchResult(ResearchOutput):
    search_stats: SearchStats | None = None


Output = ResearchResult

SYSTEM_V1 = """You are a meticulous research analyst. Answer the research question using ONLY evidence
chunks (<source> blocks). Each finding must be a concrete, checkable statement and MUST cite the exact
`chunk_id` values (copy them verbatim, full UUIDs) of the chunks that support it in `chunk_ids`. Never invent
chunk ids. If the evidence is thin or missing, say so in `gaps` rather than speculating.
Prefer chunks from pinned sources. Produce 3-7 findings when evidence allows."""

TOOL_SYSTEM_V1 = """

You have tools. The starting evidence is a first search of the run's documents; it may be incomplete.
- `web_search(query)` finds fresh sources on the web (use it when the documents do not cover the question).
- `search_documents(query, k)` searches the run's documents with a different angle or sub-question.
- `read_source(chunk_id)` reads a chunk in full with its neighbours when a snippet is cut off or ambiguous.
Make a few targeted calls (in parallel when independent), then write the final answer. Respect the tool
budgets: when a tool says BUDGET EXHAUSTED, stop calling it and finalize."""

SYSTEM_LOOP = """You are a meticulous research analyst. Answer the research question using ONLY evidence
chunks (<source> blocks). Each finding must be a concrete, checkable statement (with the numbers, dates and
names the evidence gives) and MUST cite the exact `chunk_id` values (copy them verbatim, full UUIDs) of the
chunks that support it in `chunk_ids`. Never invent chunk ids and never state what the evidence does not say.
Where the evidence disagrees, say so in the finding and cite both sides. List in `gaps` what the evidence
still does not answer. Prefer chunks from pinned sources. Produce 3-7 findings when evidence allows."""

TOOL_SYSTEM_LOOP = """

You research in rounds. The starting evidence is a first search of the run's documents; it is incomplete.
1. Explore: run 2-3 searches in parallel from different angles: `web_search(query)` for the web,
   `search_documents(query, k)` for the run's documents. Write each query as a short natural-language question
   or a description of the page you want, never a keyword list.
2. Call `reflect(known, open_gaps, next_queries)`.
3. Target and challenge: one `web_search(query, kind="target")` per important gap, and at least one
   `web_search(query, kind="challenge")` looking for criticism, limitations, failed replications or evidence
   against your main finding. Use `since_year` when recency matters and `category="research paper"` for
   scientific questions.
4. Call `reflect` again. It tells you whether to continue or stop; once it says STOP the search tools are gone.
Before citing a number, date or quote from a passage that is cut off, `read_source(chunk_id)` reads it with its
surrounding text. Prefer primary sources (papers, official statistics, filings, the organisation itself) over
coverage of them, and passages from different sites over many from one site. When a tool says BUDGET
EXHAUSTED, stop calling it."""


def system() -> str:
    return SYSTEM_LOOP if settings.research_loop else SYSTEM_V1


def tool_system() -> str:
    return TOOL_SYSTEM_LOOP if settings.research_loop else TOOL_SYSTEM_V1


# the previous budgets (RESEARCH_LOOP=false) and the loop's: the loop has more room because reflect ends early
V1_REQUESTS, LOOP_REQUESTS = 12, 20
PRIORITY_SCALE = {"high": 1.5, "normal": 1.0, "low": 0.6}
TIMEOUT = 240


def limits(config: dict[str, Any]) -> AgentLimits:
    """Budgets and request cap for one research node: loop or previous budgets, times RESEARCH_BUDGET_SCALE and
    the node's planner priority (``config.params.priority``)."""
    priority = str(((config or {}).get("params") or {}).get("priority") or "normal").lower()
    scale = max(0.1, settings.research_budget_scale * PRIORITY_SCALE.get(priority, 1.0))
    if settings.research_loop:
        base = {"web_search": settings.research_web_max, "search_documents": 6, "read_source": 8, "fetch_page": 2}
        requests = LOOP_REQUESTS
    else:
        base = {"web_search": settings.web_search_max_per_node, "search_documents": 5, "read_source": 6, "fetch_page": 2}
        requests = V1_REQUESTS
    budgets = {t: max(1, math.ceil(n * scale)) for t, n in base.items()}
    if settings.research_loop:
        budgets["reflect"] = settings.research_max_rounds + 1
    return AgentLimits(request_limit=max(6, math.ceil(requests * scale)), timeout=TIMEOUT * max(1.0, scale), budgets=budgets)


LIMITS = limits({})  # default-priority limits at import time (callers that need settings use limits(config))


def recipe_settings() -> dict[str, Any]:
    """Settings that change what this agent does (part of its recipe, see base.recipe)."""
    return {
        "loop": settings.research_loop, "max_rounds": settings.research_max_rounds,
        "min_new_docs": settings.research_min_new_docs, "page_text": settings.web_page_text,
        "page_chars": settings.web_page_chars, "seed_k": settings.research_seed_k,
        "budget": limits({}).budgets, "requests": limits({}).request_limit,
    }


agents = AgentFactory(
    "research", ResearchOutput, system, tool_system=tool_system,
    tools=[rag_tools.WEB, rag_tools.SEARCH, rag_tools.READ, rag_tools.FETCH, research_tools.REFLECT, *context_tools.TOOLS],
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


def finalize(out: ResearchOutput, deps: Any) -> ResearchResult:
    """Attach ``search_stats`` from the attempt's deps (not rendered: for the critic and the evaluation)."""
    retrieved = getattr(deps, "retrieved", {}) or {}
    cited = [retrieved[c] for f in out.findings for c in f.chunk_ids if c in retrieved]
    rounds = list(getattr(deps, "rounds", []) or [])
    budgets, counts = getattr(deps, "budgets", {}) or {}, getattr(deps, "counts", {}) or {}
    exhausted = any(counts.get(t, 0) > b for t, b in budgets.items() if t in research_tools.SEARCH_BUDGETED)
    stats = SearchStats(
        rounds=len(rounds),
        stopped_by="rule" if getattr(deps, "search_stopped", False) else "budget" if exhausted else "",
        queries=dict(getattr(deps, "query_kinds", {}) or {}),
        docs_seen=len({str(c.document_id) for c in retrieved.values()}),
        docs_cited=len({str(c.document_id) for c in cited}),
        domains_cited=sorted({d for d in (rag_tools.domain_of(c.source_url) for c in cited) if d}),
    )
    return ResearchResult(findings=out.findings, gaps=out.gaps, search_stats=stats)


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
