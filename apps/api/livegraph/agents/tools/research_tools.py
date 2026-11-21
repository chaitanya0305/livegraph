"""``reflect``: the research agent's round boundary (RESEARCH_LOOP=true).

The agent works in rounds (explore -> reflect -> target its gaps + challenge -> reflect -> finalize). ``reflect``
calls no model or network: it records what the agent knows, which gaps are still open and what it plans next
(visible on the canvas as a tool step), measures how many new documents the round added, and decides whether
searching goes on. The search stops when the agent says it is done, when the round cap is reached, or from
round 2 on when a round added fewer than RESEARCH_MIN_NEW_DOCS documents (diminishing returns). Stopping sets
``deps.search_stopped``, and ``base.wind_down`` then withdraws the search tools: the stop is enforced.
"""

from __future__ import annotations

from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition

from livegraph.agents.deps import NodeDeps
from livegraph.agents.guard import ToolOut, run_tool
from livegraph.config import settings

SEARCH_BUDGETED = ("web_search", "search_documents", "read_source")


def _remaining(deps: NodeDeps) -> str:
    parts = []
    for t in SEARCH_BUDGETED:
        b = deps.budgets.get(t)
        if b is not None:
            parts.append(f"{t} {max(0, b - deps.counts.get(t, 0))} of {b}")
    return ", ".join(parts)


def stop_reason(deps: NodeDeps, round_no: int, new_docs: int, done: bool) -> str | None:
    if done:
        return "you said the question is answered"
    if round_no >= settings.research_max_rounds:
        return f"round limit ({settings.research_max_rounds}) reached"
    if round_no >= 2 and new_docs < settings.research_min_new_docs:
        return f"the last round added only {new_docs} new document(s): further searching is unlikely to help"
    return None


async def reflect(
    ctx: RunContext[NodeDeps],
    known: str,
    open_gaps: list[str],
    next_queries: list[str],
    done: bool = False,
) -> str:
    """End a search round: say what you know, what is still missing and what you will search next.

    Args:
        known: two or three sentences: what the evidence so far establishes (and how solid it is).
        open_gaps: the specific questions the evidence does not answer yet, most important first.
        next_queries: the searches you will run next round: one per important gap, plus a challenge query
            looking for criticism or contradicting evidence if you have not run one.
        done: true when the evidence already answers the research question well.
    """

    async def impl() -> ToolOut:
        deps = ctx.deps
        seen = deps.doc_ids_seen()
        new_docs = len(seen - deps.docs_at_reflect)
        deps.docs_at_reflect = seen
        round_no = len(deps.rounds) + 1
        gaps = [g.strip() for g in open_gaps or [] if g and g.strip()]
        reason = stop_reason(deps, round_no, new_docs, done)
        deps.rounds.append({
            "round": round_no, "new_docs": new_docs, "open_gaps": gaps, "next_queries": list(next_queries or []),
            "done": bool(done), "stopped": reason,
        })
        head = f"Round {round_no}: {new_docs} new document(s) this round, {len(seen)} in total."
        if reason:
            deps.search_stopped = True
            text = (
                f"{head}\nSTOP: {reason}. The search tools are withdrawn. If a number, date or quote you will "
                "cite comes from a cut-off passage, read_source it; then return your final answer. List the gaps "
                "that are still open in `gaps`."
            )
            return ToolOut(text=text, summary=f"round {round_no}: stop ({new_docs} new docs)")
        challenge = "" if deps.query_kinds.get("challenge") else (
            " You have not looked for contradicting evidence yet: include one web_search with kind=\"challenge\"."
        )
        text = (
            f"{head}\nContinue: run the next round now (in parallel where independent): one kind=\"target\" "
            f"search per important gap.{challenge} Then reflect again.\nRemaining budget: {_remaining(deps)}."
        )
        return ToolOut(text=text, summary=f"round {round_no}: {new_docs} new docs, {len(gaps)} gaps open")

    args = {"known": known, "open_gaps": open_gaps, "next_queries": next_queries, "done": done}
    return await run_tool(ctx, "reflect", args, impl, cacheable=False, store=False)


async def _only_in_loop(ctx: RunContext[NodeDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    return tool_def if settings.research_loop else None


REFLECT = Tool(reflect, takes_ctx=True, prepare=_only_in_loop, sequential=True)
