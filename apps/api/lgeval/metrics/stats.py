"""Uncertainty for every reported number: seeded percentile bootstrap (Efron & Tibshirani, 1993), paired
bootstrap over shared questions for system differences (Koehn, 2004; Dror et al., 2018) and Wilson intervals
for proportions (Wilson, 1927). Repeats of the same question are averaged first, so the resampling unit is the
question (cluster bootstrap) and repeated runs do not inflate n."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

N_BOOT = 2000


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    k = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high): percentile bootstrap of the mean, deterministic for a given seed.
    One value -> (v, v, v); none -> NaNs."""
    xs = [float(v) for v in values]
    if not xs:
        return float("nan"), float("nan"), float("nan")
    m = mean(xs)
    if len(xs) == 1:
        return m, m, m
    rng = random.Random(seed)
    n = len(xs)
    boots = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return m, _percentile(boots, alpha / 2), _percentile(boots, 1 - alpha / 2)


def per_question(pairs: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Average values of the same question (repeats) -> {question_id: mean}."""
    acc: dict[str, list[float]] = defaultdict(list)
    for qid, v in pairs:
        acc[qid].append(float(v))
    return {q: mean(v) for q, v in acc.items()}


def paired_diff(
    a: dict[str, float], b: dict[str, float], *, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap of mean(a - b) over the questions both systems have.

    Returns {mean, ci_low, ci_high, n, p_value}; p_value is the two-sided bootstrap p for "no difference"
    (2 x the share of resampled means on the other side of 0, capped at 1)."""
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys]
    if not diffs:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "p_value": float("nan")}
    m, lo, hi = bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    if len(diffs) == 1:
        p = 1.0
    else:
        rng = random.Random(seed)
        n = len(diffs)
        boots = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
        below = sum(1 for x in boots if x <= 0) / n_boot
        above = sum(1 for x in boots if x >= 0) / n_boot
        p = min(1.0, 2 * min(below, above))
    return {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(diffs), "p_value": p}


def wilson(successes: float, n: int, *, z: float = 1.959964) -> tuple[float, float, float]:
    """(p, low, high) Wilson score interval for a proportion (successes may be fractional for half-ties)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _get(r, k):
    return r.get(k) if isinstance(r, dict) else getattr(r, k)


def aggregate(scores, *, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, dict[str, dict[str, float]]]:
    """Per-system, per-metric mean with a 95% question-level bootstrap CI (pure; for report.py).

    `scores`: ScoreRecords (or their dicts). Records with n == 0 (undefined, e.g. no verifiable facts, or a
    judge error) are skipped; repeats of a question are averaged first, then questions are resampled.
    Returns exactly::

        {system: {metric: {"mean": float, "ci_low": float, "ci_high": float, "n": int}}}

    where n = number of distinct questions behind the mean. Pass scores from one judge model at a time.
    """
    groups: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for r in scores:
        if _get(r, "n") == 0:
            continue
        v = float(_get(r, "value"))
        if math.isnan(v):
            continue
        groups[(_get(r, "system"), _get(r, "metric"))].append((_get(r, "question_id"), v))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for (system, metric), pairs in sorted(groups.items()):
        pq = per_question(pairs)
        m, lo, hi = bootstrap_ci([pq[k] for k in sorted(pq)], n_boot=n_boot, seed=seed)
        out.setdefault(system, {})[metric] = {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(pq)}
    return out


def compare_systems(scores, metric: str, a: str, b: str, *, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of system a minus system b on `metric` over shared questions (see `paired_diff`)."""
    scores = list(scores)

    def pq(system: str) -> dict[str, float]:
        return per_question((_get(r, "question_id"), float(_get(r, "value"))) for r in scores
                            if _get(r, "system") == system and _get(r, "metric") == metric and _get(r, "n") != 0)

    return paired_diff(pq(a), pq(b), n_boot=n_boot, seed=seed)
