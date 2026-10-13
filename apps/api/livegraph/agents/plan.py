"""Planner: builds the workflow graph live through tools, outputs {subquestions, rationale}."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from livegraph.agents.base import AgentContext, format_instructions, format_steer

TYPE = "plan"
TIER = "smart"


class PlanOutput(BaseModel):
    subquestions: list[str] = Field(default_factory=list, description="The research sub-questions you planned.")
    rationale: str = Field(default="", description="One short paragraph explaining the plan.")


Output = PlanOutput


@dataclass
class PlannerDeps:
    add_node: Callable[..., Awaitable[str]]
    add_edge: Callable[[str, str], Awaitable[str]]
    finish_plan: Callable[[], Awaitable[str]]
    finished: list[bool] = field(default_factory=list)


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

agent: Agent[PlannerDeps, PlanOutput] = Agent(
    output_type=PlanOutput, deps_type=PlannerDeps, instructions=SYSTEM, retries=2, name="planner"
)


@agent.tool
async def add_node(
    ctx: RunContext[PlannerDeps],
    key: str,
    type: str,
    title: str,
    question: str | None = None,
    instructions: str | None = None,
) -> str:
    """Add a node to the graph.

    Args:
        key: short unique slug, e.g. r1, c1, claims, judge.
        type: one of research, critic, claims, prosecutor, defender, judge, editor, human_input.
        title: short human readable title.
        question: for research nodes, the research question.
        instructions: optional extra instructions for the node (for human_input: the question for the user).
    """
    return await ctx.deps.add_node(key=key, type=type, title=title, question=question, instructions=instructions)


@agent.tool
async def add_edge(ctx: RunContext[PlannerDeps], source_key: str, target_key: str) -> str:
    """Add a dependency edge: target runs after source and receives its output."""
    return await ctx.deps.add_edge(source_key, target_key)


@agent.tool
async def finish_plan(ctx: RunContext[PlannerDeps]) -> str:
    """Signal that the graph is complete."""
    ctx.deps.finished.append(True)
    return await ctx.deps.finish_plan()


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
