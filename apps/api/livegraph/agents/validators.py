"""Output validators (``@agent.output_validator``): bounce bad outputs back to the model with ``ModelRetry``.

They are skipped for partial (streaming) outputs. Once the output-retry budget is spent they repair the
output instead (the old ``sanitize`` behaviour: drop what is invalid), so a stubborn model degrades
gracefully rather than failing the node.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic_ai import ModelRetry, RunContext

from livegraph.agents.deps import NodeDeps

CITE_HINT = "cite only chunk_id values (full UUIDs) of <source> blocks you were given or retrieved with tools"


def _active(ctx: RunContext[NodeDeps | None]) -> bool:
    return not ctx.partial_output and isinstance(ctx.deps, NodeDeps)


def _retry_or(ctx: RunContext[Any], problems: list[str], repair: Callable[[], Any]) -> Any:
    if problems and ctx.retry < ctx.max_retries:
        raise ModelRetry("Fix these problems and answer again: " + "; ".join(problems[:8]))
    return repair()


def _cited_lists(items: list[Any], attr: str, ctx: RunContext[NodeDeps]) -> Any:
    bad = sorted({c for it in items for c in getattr(it, attr) if not ctx.deps.citable(c)})

    def repair() -> None:
        for it in items:
            setattr(it, attr, [c for c in getattr(it, attr) if ctx.deps.citable(c)])

    return [f"unknown chunk_id {c!r}" for c in bad[:5]] + ([CITE_HINT] if bad else []), repair


async def research(ctx: RunContext[NodeDeps], out: Any) -> Any:
    if not _active(ctx):
        return out
    problems, repair = _cited_lists(out.findings, "chunk_ids", ctx)
    _retry_or(ctx, problems, repair)
    return out


async def prosecutor(ctx: RunContext[NodeDeps], out: Any) -> Any:
    if not _active(ctx):
        return out
    problems, repair = _cited_lists(out.attacks, "chunk_ids", ctx)
    _retry_or(ctx, problems, repair)
    return out


async def defender(ctx: RunContext[NodeDeps], out: Any) -> Any:
    if not _active(ctx):
        return out
    problems, repair = _cited_lists(out.defenses, "chunk_ids", ctx)
    _retry_or(ctx, problems, repair)
    return out


async def claims(ctx: RunContext[NodeDeps], out: Any) -> Any:
    if not _active(ctx):
        return out
    problems, repair = _cited_lists(out.claims, "support_chunk_ids", ctx)
    if ctx.deps.cited_ok:  # only demand support when there is evidence to cite
        unsupported = [c.key for c in out.claims if not [x for x in c.support_chunk_ids if ctx.deps.citable(x)]]
        if unsupported:
            problems.append(f"claims without support_chunk_ids: {', '.join(unsupported)} (support or drop them)")
    _retry_or(ctx, problems, repair)
    for i, c in enumerate(out.claims, 1):
        c.key = c.key.strip() or f"C{i}"
    return out


def _claim_keys(deps: NodeDeps, *, active_only: bool = False) -> list[str]:
    claims = deps.ctx.claims if deps.ctx else []
    return [c["key"] for c in claims if not (active_only and c.get("status") == "struck")]


async def judge(ctx: RunContext[NodeDeps], out: Any) -> Any:
    if not _active(ctx):
        return out
    keys = _claim_keys(ctx.deps)
    if not keys:
        return out
    seen: dict[str, int] = {}
    for v in out.verdicts:
        seen[v.claim_key] = seen.get(v.claim_key, 0) + 1
    problems = [f"missing verdict for {k}" for k in keys if k not in seen]
    problems += [f"{k} has {n} verdicts (exactly one per claim)" for k, n in seen.items() if n > 1]
    problems += [f"{k} is not a claim key" for k in seen if k not in keys]

    def repair() -> None:
        kept, done = [], set()
        for v in out.verdicts:
            if v.claim_key in keys and v.claim_key not in done:
                kept.append(v)
                done.add(v.claim_key)
        out.verdicts = kept

    _retry_or(ctx, problems, repair)
    return out


_REF = re.compile(r"\((C\d+(?:\s*,\s*C\d+)*)\)")


async def editor(ctx: RunContext[NodeDeps], out: Any) -> Any:
    if not _active(ctx) or not ctx.deps.ctx or not ctx.deps.ctx.claims:
        return out
    ok = set(_claim_keys(ctx.deps, active_only=True))
    bad = sorted({k.strip() for m in _REF.finditer(out.markdown) for k in m.group(1).split(",") if k.strip() not in ok})
    problems = [f"({k}) is struck or unknown: do not reference or state it" for k in bad]

    def repair() -> None:
        def fix(m: re.Match[str]) -> str:
            keep = [k.strip() for k in m.group(1).split(",") if k.strip() in ok]
            return f"({', '.join(keep)})" if keep else ""

        out.markdown = _REF.sub(fix, out.markdown)

    _retry_or(ctx, problems, repair)
    return out
