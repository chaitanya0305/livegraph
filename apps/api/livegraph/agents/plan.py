"""Planner: builds the workflow graph live through tools, outputs {subquestions, rationale}."""

from __future__ import annotations

from pydantic import BaseModel, Field

from livegraph.agents.base import AgentContext, AgentFactory, AgentLimits, format_instructions, format_steer
from livegraph.agents.tools import graph_tools, rag_tools

TYPE = "plan"
TIER = "smart"


class PlanOutput(BaseModel):
    subquestions: list[str] = Field(default_factory=list, description="The research sub-questions you planned.")
    rationale: str = Field(default="", description="One short paragraph explaining the plan.")


Output = PlanOutput


SYSTEM = """You are the Planner of LiveGraph, a system that researches a goal and writes a "Living Brief".
You build the workflow graph by calling tools. Every tool call immediately appears on the user's canvas.

Build exactly this shape (node keys are short lowercase slugs):
1. 3 to 5 `research` nodes (keys r1, r2, ...), each with a focused, self-contained `question` that
   together cover the goal. Add an edge plan -> rN for each.
2. For each research node rN, a `critic` node (key cN) with an edge rN -> cN.
3. One `claims` node (key claims) with an edge from every research node.
4. `prosecutor` (key prosecutor) and `defender` (key defender) nodes, each with an edge claims -> them.
5. A `judge` node (key judge) with edges claims -> judge, prosecutor -> judge, defender -> judge.
6. An `editor` node (key editor) with an edge judge -> editor.
Optionally add a `human_input` node (put the question for the user in `instructions`) only if the goal is
genuinely ambiguous, wired before the node that needs the answer.

The node with key `plan` already exists (it is you). Add nodes before edges that reference them.
When done, call `finish_plan`, then return your final output: the list of sub-questions and a one
paragraph rationale."""

TOOL_SYSTEM = """

Before planning you may call `list_documents()` once to see what the user supplied, and
`search_documents(query)` (at most twice) to see what they cover, so the research questions target real
gaps. Then build the graph."""

LIMITS = AgentLimits(request_limit=20, timeout=360, budgets={"list_documents": 1, "search_documents": 2, "fetch_page": 1})

agents = AgentFactory(
    "planner", PlanOutput, SYSTEM, tool_system=TOOL_SYSTEM,
    base_tools=[graph_tools.ADD_NODE, graph_tools.ADD_EDGE, graph_tools.FINISH_PLAN],
    tools=[graph_tools.LIST_DOCUMENTS, rag_tools.SEARCH, rag_tools.FETCH],
)
agent = agents.get()


def build_prompt(ctx: AgentContext) -> str:
    docs = (ctx.config.get("params") or {}).get("document_titles") or []
    doc_s = ("\n\nThe user supplied these documents: " + "; ".join(docs)) if docs else ""
    return f"## Goal\n{ctx.goal}{doc_s}{format_instructions(ctx)}{format_steer(ctx)}"


def render(out: PlanOutput) -> str:
    parts = []
    if out.subquestions:
        parts.append("**Sub-questions**\n" + "".join(f"\n{i}. {q}" for i, q in enumerate(out.subquestions, 1)))
    if out.rationale:
        parts.append("\n\n**Rationale**\n\n" + out.rationale)
    return "".join(parts)


def parse_edit(text: str, prev: dict) -> dict:
    return {**prev, "edited_markdown": text}
