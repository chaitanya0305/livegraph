"""`python -m lgeval <command>`. Modules plug in subcommands through `register(subparsers)`.

A plugin's `register` adds its parsers and sets `func` on each: `p.set_defaults(func=handler)`, where
`handler(args)` returns an exit code (or None), or a coroutine that does (run with asyncio.run).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
from dataclasses import dataclass, field

from lgeval.costs import (
    CONFIG_COST_FACTOR,
    EST_RUN_USAGE,
    RECOMPUTE_FACTOR,
    SCORE_USD_PER_BRIEF,
    est_run_usd,
    load_pricing,
)

PLUGINS = (
    "lgeval.runners.quality",
    "lgeval.runners.baseline",
    "lgeval.runners.recompute",
    "lgeval.metrics",
    "lgeval.report",
)

PLAN_CONFIGS = ["full", "no_debate", "no_critic", "single_shot", "no_context_tools", "single_agent"]
MUTATIONS = ["distrust", "steer", "edit", "identical"]


@dataclass
class Suite:
    questions: int
    repeats: int
    recompute_runs: int
    configs: list[str] = field(default_factory=lambda: list(PLAN_CONFIGS))
    mutations: list[str] = field(default_factory=lambda: list(MUTATIONS))
    dataset: str = "custom"


SUITES: dict[str, Suite] = {
    "pilot": Suite(questions=10, repeats=1, recompute_runs=5),
    "full": Suite(questions=30, repeats=3, recompute_runs=20),
}


@dataclass
class PlanRow:
    experiment: str
    detail: str
    runs: float  # full-run equivalents (0 for scoring)
    briefs: int
    usd: float


def plan_rows(suite: Suite, run_usd: float) -> list[PlanRow]:
    rows: list[PlanRow] = []
    for cfg in suite.configs:
        n = suite.questions * suite.repeats
        eq = n * CONFIG_COST_FACTOR.get(cfg, 1.0)
        exp = "baseline" if cfg == "single_agent" else "quality"
        rows.append(PlanRow(exp, f"{cfg}: {suite.questions} q x {suite.repeats} rep", eq, n, eq * run_usd))
    for m in suite.mutations:
        n = suite.recompute_runs
        eq = n * (RECOMPUTE_FACTOR + 1.0)  # the update + a fresh-run baseline
        rows.append(PlanRow("recompute", f"{m}: {n} runs (+ fresh baseline)", eq, 2 * n, eq * run_usd))
    briefs = sum(r.briefs for r in rows)
    rows.append(PlanRow("score", f"{briefs} briefs x ${SCORE_USD_PER_BRIEF:.2f}", 0, 0, briefs * SCORE_USD_PER_BRIEF))
    return rows


def cmd_plan(args: argparse.Namespace) -> int:
    suite = SUITES[args.suite]
    pricing = load_pricing(list(EST_RUN_USAGE), offline=args.offline)
    run_usd = est_run_usd(pricing)
    rows = plan_rows(suite, run_usd)
    print(f"Suite {args.suite!r}: dataset={suite.dataset}, est. ${run_usd:.3f} per full LiveGraph run")
    for m, p in pricing.items():
        print(f"  {m}: ${p.in_per_m:.2f}/M in, ${p.out_per_m:.2f}/M out")
    print()
    head = f"{'experiment':<10}  {'detail':<40}  {'run-eq':>7}  {'est USD':>8}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r.experiment:<10}  {r.detail:<40}  {r.runs:>7.1f}  {r.usd:>8.2f}")
    print("-" * len(head))
    total_runs, total_usd = sum(r.runs for r in rows), sum(r.usd for r in rows)
    print(f"{'total':<10}  {'':<40}  {total_runs:>7.1f}  {total_usd:>8.2f}")
    return 0


def _register_plan(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("plan", help="dry-run estimate of a whole study (no runs, no LLM calls)")
    p.add_argument("--suite", choices=sorted(SUITES), default="pilot")
    p.add_argument("--offline", action="store_true", help="use the fallback price table (no network)")
    p.set_defaults(func=cmd_plan)


def build_parser(plugins: tuple[str, ...] = PLUGINS) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lgeval", description="LiveGraph evaluation harness")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _register_plan(sub)
    missing: list[str] = []
    for name in plugins:
        try:
            importlib.import_module(name).register(sub)
        except Exception as e:  # a plugin that is missing or broken must not take the CLI down
            missing.append(f"{name} ({type(e).__name__}: {e})")
    if missing:
        parser.epilog = "not available yet: " + "; ".join(missing)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(PLUGINS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    result = func(args)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return int(result or 0)
