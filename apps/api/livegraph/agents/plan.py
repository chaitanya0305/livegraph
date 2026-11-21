"""Planner: builds the workflow graph live through tools, outputs {subquestions, rationale}."""

from __future__ import annotations

from pydantic import BaseModel, Field

from livegraph.agents.base import AgentContext, AgentFactory, AgentLimits, format_instructions, format_steer
from livegraph.agents.tools import graph_tools, rag_tools
from livegraph.config import disabled_types, settings

TYPE = "plan"
TIER = "smart"


class PlanOutput(BaseModel):
    subquestions: list[str] = Field(default_factory=list, description="The research sub-questions you planned.")
    rationale: str = Field(default="", description="One short paragraph explaining the plan.")


Output = PlanOutput


SYSTEM_TEMPLATE = """You are the Planner of LiveGraph, a system that researches a goal and writes a "Living Brief".
You build the workflow graph by calling tools. Every tool call immediately appears on the user's canvas.

Build exactly this shape (node keys are short lowercase slugs):
1. 3 to {max_research} `research` nodes (keys r1, r2, ...), each with a focused, self-contained `question` that
   together cover the goal: as many as the goal really has distinct sub-questions, no more. Give each a
   `priority`: "high" for the sub-questions the answer hinges on (they get a larger search budget), "low" for
   background. When the goal is contested or a claim could be wrong, make one research node look for the
   strongest evidence and criticism against the prevailing view. Add an edge plan -> rN for each.
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


def system() -> str:
    return SYSTEM_TEMPLATE.format(max_research=max(3, settings.plan_max_research))


TOOL_SYSTEM = """

Before planning you may call `list_documents()` once to see what the user supplied, and
`search_documents(query)` (at most twice) to see what they cover, so the research questions target real
gaps."""
TOOL_SYSTEM_WEB = """ You may also scout the topic with `web_search(query)` (at most twice: one broad query on the
goal, one on how it is debated) to learn its main sub-topics, recent developments and points of disagreement,
so each research question targets something real."""


def tool_system() -> str:
    return TOOL_SYSTEM + (TOOL_SYSTEM_WEB if settings.plan_web_search else "") + " Then build the graph."


LIMITS = AgentLimits(
    request_limit=20, timeout=360,
    budgets={"list_documents": 1, "search_documents": 2, "web_search": 2, "fetch_page": 1},
)


def recipe_settings() -> dict:
    return {"web": settings.plan_web_search, "max_research": settings.plan_max_research}


agents = AgentFactory(
    "planner", PlanOutput, system, tool_system=tool_system,
    base_tools=[graph_tools.ADD_NODE, graph_tools.ADD_EDGE, graph_tools.FINISH_PLAN],
    tools=[graph_tools.LIST_DOCUMENTS, rag_tools.SEARCH, rag_tools.PLAN_WEB, rag_tools.FETCH],
)
agent = agents.get()


def build_prompt(ctx: AgentContext) -> str:
    docs = (ctx.config.get("params") or {}).get("document_titles") or []
    doc_s = ("\n\nThe user supplied these documents: " + "; ".join(docs)) if docs else ""
    return f"## Goal\n{ctx.goal}{doc_s}{format_disabled()}{format_instructions(ctx)}{format_steer(ctx)}"


def format_disabled() -> str:
    """DISABLED_NODE_TYPES: tell the planner to leave those steps out and where the editor's edge comes from."""
    off = disabled_types()
    if not off:
        return ""
    debaters = [t for t in ("prosecutor", "defender") if t not in off]
    feeder = "judge" if "judge" not in off else " and ".join(debaters) if debaters else "claims"
    return (
        f"\n\n## Disabled node types\nThese node types are disabled in this deployment: {', '.join(sorted(off))}. "
        f"Do not add them or edges to them (add_node rejects them); skip those steps of the shape. "
        f"Wire the editor with an edge from {feeder}."
    )


def render(out: PlanOutput) -> str:
    parts = []
    if out.subquestions:
        parts.append("**Sub-questions**\n" + "".join(f"\n{i}. {q}" for i, q in enumerate(out.subquestions, 1)))
    if out.rationale:
        parts.append("\n\n**Rationale**\n\n" + out.rationale)
    return "".join(parts)


def parse_edit(text: str, prev: dict) -> dict:
    return {**prev, "edited_markdown": text}
