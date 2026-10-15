"""Opt-in live smoke test: one research agent (real OpenRouter model, real tools) outside the engine.

Checks that the configured fast model + OUTPUT_MODE_AGENTS behave inside a tool loop (prompted JSON output
mixed with tool calls, validators, soft budgets). Needs OPENROUTER_API_KEY and the dev Postgres (searches
run against it; web pages found are ingested under a throwaway run id). Tool events are printed, not
persisted. Costs a few cents.

    cd apps/api && uv run python scripts/smoke_agents.py "How fast is grid-scale battery storage growing?"
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from typing import Any

from pydantic_ai.usage import UsageLimits

from livegraph import rag
from livegraph.agents import research
from livegraph.agents.base import AgentContext
from livegraph.agents.deps import NodeDeps
from livegraph.agents.models import model_for
from livegraph.agents.runner import run_agent
from livegraph.config import settings
from livegraph.db import base as db_base
from livegraph.engine.core import format_budgets


class PrintDeps(NodeDeps):
    async def emit(self, type: str, payload: dict[str, Any], ts: str = "") -> bool:
        if type == "tool.call":
            print(f"  -> {payload['tool']}({payload['args_summary']})  step {payload['step']}/{payload['budget']}")
        elif type == "tool.result":
            print(f"  <- {'ok' if payload['ok'] else 'FAIL'} {payload['summary']}  ({payload['ms']} ms)")
        self.tracer.apply(type, payload, ts)
        return True


async def main(question: str) -> int:
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set; skipping the live smoke test.")
        return 0
    run_id = uuid.uuid4()
    async with db_base.SessionLocal() as s:
        seeds = await rag.search(s, run_id=run_id, query=question, k=settings.research_seed_k)
    ctx = AgentContext(goal=question, node_key="r1", node_type="research", title=question,
                       config={"question": question}, evidence=[c.model_dump(mode="json") for c in seeds])
    deps = PrintDeps(run_id=run_id, node_id=uuid.uuid4(), node_key="r1", node_type="research", sched=None,
                     attempt_id=uuid.uuid4().hex, me=None, ctx=ctx, budgets=dict(research.LIMITS.budgets))
    deps.add_chunks(seeds, "seed")
    model = model_for("research", research.TIER)
    agent = research.agents.for_model(model, with_tools=True)
    print(f"model={getattr(model, 'model_name', model)} output_mode={research.agents.mode_for(model, True)} "
          f"web_search={settings.web_search} seeds={len(seeds)}")
    resets = 0

    def on_reset(_: str) -> None:
        nonlocal resets
        resets += 1

    t0 = time.monotonic()
    result = await run_agent(
        agent, research.build_prompt(ctx) + format_budgets(research.LIMITS), model=model, render=research.render,
        deps=deps, on_reset=on_reset, limits=UsageLimits(request_limit=research.LIMITS.request_limit),
        timeout=research.LIMITS.scaled_timeout,
    )
    cited = research.cited_ids(result.output)
    unknown = [c for c in cited if not deps.citable(c)]
    print(f"\n{result.output_text}\n")
    print(f"{time.monotonic() - t0:.1f}s  tokens in/out {result.tokens_in}/{result.tokens_out}  "
          f"tool calls {len(deps.trace)}  retrieved {len(deps.retrieved)}  cited {len(cited)}  resets {resets}")
    if unknown or not result.output.findings:
        print(f"FAIL: unknown cited ids {unknown} / findings {len(result.output.findings)}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How fast is grid-scale battery storage growing worldwide?"
    sys.exit(asyncio.run(main(q)))
