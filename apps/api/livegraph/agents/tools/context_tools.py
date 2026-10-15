"""Cross-agent read tools: list_agent_runs, read_agent_output, read_agent_trace, recall, search_context.

Scope rule: an agent reads only its own attempt's stored results and the current versions of its transitive
active ancestors (``deps.readable``, built by the engine in ``_context``); anything else answers
``not readable: <key> is not upstream of you``. Everything read is upstream of the reader, so its input hash
(parent versions, transitively) already covers it. Results are wrapped as untrusted ``<source>`` data, and
every read of another agent's work is reported as ``reads: [{node_key}]`` on ``tool.result``.
Hidden when ``CONTEXT_TOOLS=false``.
"""

from __future__ import annotations

import html

from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition
from sqlalchemy import func, literal_column, select

from livegraph.agents import context
from livegraph.agents.deps import NodeDeps
from livegraph.agents.guard import ToolOut, run_tool
from livegraph.config import settings
from livegraph.db.models import ContextItem, Node, NodeVersion

BUDGETS = {"list_agent_runs": 2, "read_agent_output": 4, "read_agent_trace": 3, "recall": 6, "search_context": 3}
TRACE_MAX_STEPS = 60
SEARCH_K_MAX = 10
_EN = literal_column("'english'")  # a literal (not a bind) so the GIN expression index applies
_HEADLINE = 'MaxWords=35, MinWords=12, StartSel="", StopSel=""'  # plain-text snippet around the match

TOOL_SYSTEM = """

## Reading other agents' work
- `list_agent_runs()` lists the upstream agents whose work you may read.
- `read_agent_output(node_key, offset)` reads an upstream agent's full output (paged).
- `read_agent_trace(node_key)` shows what an upstream agent did (its tool calls, each with a ref).
- `recall(ref, offset)` returns the full text of a stored result: one of your own earlier tool results (older
  ones in your history are shortened to a `[ctx_...]` line) or a ref from an upstream trace.
- `search_context(query, k)` full-text searches the stored results you may read.
Use them only when your inputs do not already contain what you need."""


def _wrap(attrs: str, body: str) -> str:
    return f"<source {attrs}>\n{body.replace('</source', '&lt;/source')}\n</source>"


def _not_readable(key: str) -> ToolOut:
    msg = f"not readable: {key} is not upstream of you"
    return ToolOut(text=msg, summary=msg, ok=False, error=msg)


async def _check_key(deps: NodeDeps, node_key: str) -> ToolOut | None:
    key = (node_key or "").strip()
    if key == deps.node_key:
        return ToolOut(text="that is your own node; read your inputs instead", summary="own node", ok=False,
                       error="own node")
    if key in deps.readable:
        return None
    async with deps.session() as s:
        exists = await s.scalar(select(Node.id).where(Node.run_id == deps.run_id, Node.key == key))
    if exists is None:
        return ToolOut(text=f"error: unknown node {key}; call list_agent_runs()", summary="unknown node", ok=False,
                       error="unknown node")
    return _not_readable(key)


async def list_agent_runs(ctx: RunContext[NodeDeps]) -> str:
    """List the upstream agents (your transitive inputs) whose output and tool trace you can read."""

    async def impl() -> ToolOut:
        deps = ctx.deps
        if not deps.readable:
            return ToolOut(text="no upstream agent work is readable", summary="0 agents")
        async with deps.session() as s:
            rows = (await s.execute(
                select(Node.key, Node.type, Node.title, NodeVersion.version, func.length(NodeVersion.output_text),
                       func.jsonb_array_length(NodeVersion.trace))
                .join(NodeVersion, NodeVersion.id == Node.current_version_id)
                .where(Node.run_id == deps.run_id, Node.key.in_(list(deps.readable)))
                .order_by(Node.created_at, Node.key)
            )).all()
        lines = [
            f'- {key} ({type_}, v{version}) "{html.escape(title)}": {chars} chars of output, {steps} tool steps'
            for key, type_, title, version, chars, steps in rows
        ]
        return ToolOut(text=_wrap("kind=agent_runs", "\n".join(lines)), summary=f"{len(rows)} agents")

    return await run_tool(ctx, "list_agent_runs", {}, impl, store=False)


async def read_agent_output(ctx: RunContext[NodeDeps], node_key: str, offset: int = 0) -> str:
    """Read the current output of an upstream agent, 4000 characters per page.

    Args:
        node_key: the key of an upstream node (see list_agent_runs).
        offset: character offset to continue from (the previous page tells you the next one).
    """

    async def impl() -> ToolOut:
        deps = ctx.deps
        if (err := await _check_key(deps, node_key)) is not None:
            return err
        key = node_key.strip()
        async with deps.session() as s:
            v = await s.get(NodeVersion, deps.readable[key])
        text = v.output_text if v else ""
        chunk, nxt = context.page(text, offset)
        more = f"\n(more: read_agent_output('{key}', offset={nxt}))" if nxt is not None else ""
        attrs = f"node={key} version={v.version if v else 0} offset={max(0, int(offset or 0))} total={len(text)}"
        return ToolOut(text=_wrap(attrs, chunk) + more, summary=f"{key}: {len(chunk)} of {len(text)} chars",
                       reads=[{"node_key": key}])

    return await run_tool(ctx, "read_agent_output", {"node_key": node_key, "offset": offset}, impl)


async def read_agent_trace(ctx: RunContext[NodeDeps], node_key: str) -> str:
    """Show the tool calls an upstream agent made for its current output (each stored result has a ref).

    Args:
        node_key: the key of an upstream node (see list_agent_runs).
    """

    async def impl() -> ToolOut:
        deps = ctx.deps
        if (err := await _check_key(deps, node_key)) is not None:
            return err
        key = node_key.strip()
        async with deps.session() as s:
            v = await s.get(NodeVersion, deps.readable[key])
        steps = (v.trace if v else None) or []
        lines = []
        for st in steps[:TRACE_MAX_STEPS]:
            ref = f"[{st['ref']}] " if st.get("ref") else ""
            who = f"{st.get('agent')}: " if st.get("parent_call_id") else ""
            status = "" if st.get("ok") else " (failed)"
            cites = "".join(f" [chunk_id={c}]" for c in st.get("chunk_ids") or [])
            lines.append(f"- {ref}{who}{st['tool']}({st.get('args_summary', '')}) -> {st.get('summary', '')}{status}{cites}")
            for m in st.get("exchange") or []:
                lines.append(f"    {m['from_agent']} -> {m['to_agent']} ({m['kind']}): {m['text']}")
        if len(steps) > TRACE_MAX_STEPS:
            lines.append(f"(+{len(steps) - TRACE_MAX_STEPS} more steps)")
        body = "\n".join(lines) or "(no tool calls)"
        return ToolOut(text=_wrap(f"node={key} kind=trace", body) + "\nrecall(ref) returns a step's full result.",
                       summary=f"{key}: {len(steps)} steps", reads=[{"node_key": key}])

    return await run_tool(ctx, "read_agent_trace", {"node_key": node_key}, impl)


async def recall(ctx: RunContext[NodeDeps], ref: str, offset: int = 0) -> str:
    """Return the full text of a stored tool result by its ref (4000 characters per page).

    Args:
        ref: a ref such as ctx_1a2b3c4d (from a shortened result in your history or an upstream trace).
        offset: character offset to continue from.
    """

    async def impl() -> ToolOut:
        deps = ctx.deps
        async with deps.session() as s:
            row, key, err = await context.load_readable(s, deps, ref)
        if row is None:
            return ToolOut(text=err, summary=err, ok=False, error=err)
        deps.stubs.setdefault(row.ref, context.stub_for(row.ref, row.summary))
        chunk, nxt = context.page(row.text, offset)
        more = f"\n(more: recall('{row.ref}', offset={nxt}))" if nxt is not None else ""
        attrs = f"ref={row.ref} node={key} agent={row.agent} tool={row.tool} total={len(row.text)}"
        return ToolOut(
            text=_wrap(attrs, chunk) + more, summary=f"{row.ref} ({key} {row.tool}): {len(chunk)} chars",
            ref=row.ref, reads=[] if key == deps.node_key else [{"node_key": key}],
        )

    return await run_tool(ctx, "recall", {"ref": ref, "offset": offset}, impl, store=False)


async def search_context(ctx: RunContext[NodeDeps], query: str, k: int = 5) -> str:
    """Full-text search over the stored tool results you may read (yours and your upstream agents').

    Args:
        query: words to look for (web-search syntax: quotes, OR, -exclude).
        k: how many results (1-10).
    """

    async def impl() -> ToolOut:
        deps = ctx.deps
        kk = max(1, min(int(k or 5), SEARCH_K_MAX))
        tsq = func.websearch_to_tsquery(_EN, query)
        doc = func.to_tsvector(_EN, ContextItem.text)
        async with deps.session() as s:
            rows = (await s.execute(
                select(ContextItem.ref, Node.key, ContextItem.agent, ContextItem.tool,
                       func.ts_headline(_EN, ContextItem.text, tsq, _HEADLINE))
                .join(Node, Node.id == ContextItem.node_id)
                .where(ContextItem.run_id == deps.run_id, context.scope_filter(deps), doc.op("@@")(tsq))
                .order_by(func.ts_rank(doc, tsq).desc(), ContextItem.created_at)
                .limit(kk)
            )).all()
        if not rows:
            return ToolOut(text="no stored results match", summary="0 results")
        blocks = [
            _wrap(f"ref={ref} node={key} agent={agent} tool={tool}", " ".join(snippet.split()))
            for ref, key, agent, tool, snippet in rows
        ]
        reads = [{"node_key": key} for key in dict.fromkeys(r[1] for r in rows) if key != deps.node_key]
        return ToolOut(text="\n\n".join(blocks) + "\nrecall(ref) returns a result in full.",
                       summary=f"{len(rows)} results", reads=reads)

    return await run_tool(ctx, "search_context", {"query": query, "k": k}, impl)


async def _only_if_enabled(ctx: RunContext[NodeDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    return tool_def if settings.context_tools else None


LIST_RUNS = Tool(list_agent_runs, takes_ctx=True, prepare=_only_if_enabled)
READ_OUTPUT = Tool(read_agent_output, takes_ctx=True, prepare=_only_if_enabled)
READ_TRACE = Tool(read_agent_trace, takes_ctx=True, prepare=_only_if_enabled)
RECALL = Tool(recall, takes_ctx=True, prepare=_only_if_enabled)
SEARCH_CONTEXT = Tool(search_context, takes_ctx=True, prepare=_only_if_enabled)

TOOLS = [LIST_RUNS, READ_OUTPUT, READ_TRACE, RECALL, SEARCH_CONTEXT]
NAMES = frozenset(BUDGETS)


def has_context_tools(agents) -> bool:
    """True when an ``AgentFactory`` carries the context tools."""
    return any(t.name in NAMES for t in agents.tools)
