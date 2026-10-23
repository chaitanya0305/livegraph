"""LLM-judge metrics: factual precision, citation recall/precision, answer accuracy, pairwise preference and
calibration, plus bootstrap statistics. `register` adds the `score` and `pairwise` CLI subcommands."""

from __future__ import annotations

from lgeval.metrics.llm import DEFAULT_JUDGE_MODEL, LLM, make_llm
from lgeval.metrics.stats import aggregate, bootstrap_ci, compare_systems, paired_diff


def register(subparsers) -> None:
    from lgeval.metrics.commands import register as _register

    _register(subparsers)


__all__ = ["DEFAULT_JUDGE_MODEL", "LLM", "aggregate", "bootstrap_ci", "compare_systems", "make_llm", "paired_diff",
           "register"]
