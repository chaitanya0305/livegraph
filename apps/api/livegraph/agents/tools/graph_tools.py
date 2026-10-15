"""Graph tools: the planner's add_node/add_edge/finish_plan/list_documents, the critic's follow-up request and
the judge's/debaters' ``request_upstream_work``.

Only list_documents is kept in the context store; the graph edits are short acknowledgements.

Mutations are delegated to the run's scheduler (``deps.sched``), which applies them under the run mutex
and only while the calling attempt still owns its node. These tools are ``sequential`` so a model's batch
of graph edits applies in the order it was emitted.
"""

from __future__ import annotations

import html

from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition

from livegraph import rag
from livegraph.agents.deps import NodeDeps
from livegraph.agents.guard import ToolOut, run_tool


async def add_node(
    ctx: RunContext[NodeDeps],
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

    async def impl() -> ToolOut:
        return await ctx.deps.sched.tool_add_node(
            ctx.deps, key=key, type=type, title=title, question=question, instructions=instructions
        )

    args = {"key": key, "type": type, "title": title, "question": question, "instructions": instructions}
    return await run_tool(
        ctx, "add_node", {k: v for k, v in args.items() if v is not None}, impl, cacheable=False, store=False
    )


async def add_edge(ctx: RunContext[NodeDeps], source_key: str, target_key: str) -> str:
    """Add a dependency edge: target runs after source and receives its output."""

    async def impl() -> ToolOut:
        return await ctx.deps.sched.tool_add_edge(ctx.deps, source_key, target_key)

    args = {"source_key": source_key, "target_key": target_key}
    return await run_tool(ctx, "add_edge", args, impl, cacheable=False, store=False)


async def finish_plan(ctx: RunContext[NodeDeps]) -> str:
    """Signal that the graph is complete."""

    async def impl() -> ToolOut:
        return ToolOut(text="plan finished; now return the final output", summary="plan finished")

    return await run_tool(ctx, "finish_plan", {}, impl, cacheable=False, store=False)


async def list_documents(ctx: RunContext[NodeDeps]) -> str:
    """List the documents available to this run (uploads, global library, web pages found so far)."""

    async def impl() -> ToolOut:
        async with ctx.deps.session() as s:
            docs = await rag.list_documents(s, ctx.deps.run_id)
        if not docs:
            return ToolOut(text="no documents yet (research nodes will search the web)", summary="0 documents")
        lines = [
            f'<source document_id={d.id} trust={d.trust} origin={d.origin} chunks={d.chunk_count}>'
            f"{html.escape(d.title)}</source>"
            for d in docs[:50]
        ]
        return ToolOut(text="\n".join(lines), summary=f"{len(docs)} documents")

    return await run_tool(ctx, "list_documents", {}, impl)


async def request_follow_up_research(ctx: RunContext[NodeDeps], question: str) -> str:
    """Add ONE follow-up research node that fills the biggest gap of the research you are critiquing.

    Only call this when the research is thin. The new node runs before the claims are drafted.

    Args:
        question: one focused, self-contained research question.
    """

    async def impl() -> ToolOut:
        return await ctx.deps.sched.tool_follow_up(ctx.deps, question)

    return await run_tool(
        ctx, "request_follow_up_research", {"question": question}, impl, cacheable=False, store=False
    )


async def request_upstream_work(ctx: RunContext[NodeDeps], target_node_key: str, instruction: str) -> str:
    """Ask an upstream agent (one of your transitive inputs) to redo its work with a specific instruction.

    Use it only when an input is clearly missing something decisive for your answer (e.g. research that never
    looked at a key figure). The request is applied after you return your output: the target reruns with
    your instruction, and you run again only if its new result changes your inputs. At most one request per
    target, and only a few per run. Your current inputs are final for this answer: finish it now.

    Args:
        target_node_key: key of an upstream node (see list_agent_runs); not the plan or a human input.
        instruction: what the target should do differently (at most 500 characters).
    """

    async def impl() -> ToolOut:
        return await ctx.deps.sched.tool_request_upstream(ctx.deps, target_node_key, instruction)

    args = {"target_node_key": target_node_key, "instruction": instruction}
    return await run_tool(ctx, "request_upstream_work", args, impl, cacheable=False, store=False)


async def _top_level_only(ctx: RunContext[NodeDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    """Hidden inside nested runs (the judge's ask_rebuttal) and without a scheduler."""
    deps = ctx.deps
    return tool_def if deps.parent_call_id is None and deps.sched is not None else None


ADD_NODE = Tool(add_node, takes_ctx=True, sequential=True)
ADD_EDGE = Tool(add_edge, takes_ctx=True, sequential=True)
FINISH_PLAN = Tool(finish_plan, takes_ctx=True, sequential=True)
LIST_DOCUMENTS = Tool(list_documents, takes_ctx=True)
FOLLOW_UP = Tool(request_follow_up_research, takes_ctx=True, sequential=True)
UPSTREAM = Tool(request_upstream_work, takes_ctx=True, sequential=True, prepare=_top_level_only)
