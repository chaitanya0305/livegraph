"""Results → publishable artifacts: markdown tables, LaTeX fragments, SVG charts and summary.json."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from xml.sax.saxutils import escape

from pydantic import BaseModel, ValidationError

from lgeval.schema import ClaimLabel, JudgeCost, Manifest, PairwiseRecord, RecomputeRecord, RunRecord, ScoreRecord

T = TypeVar("T", bound=BaseModel)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "docs" / "results"

QUALITY_METRICS = ["factual_precision", "citation_precision", "citation_recall", "answer_accuracy"]
METRIC_LABELS = {
    "factual_precision": "Factual prec.",
    "citation_precision": "Cit. prec.",
    "citation_recall": "Cit. recall",
    "answer_accuracy": "Answer acc.",
    "cost_usd": "Cost ($)",
    "wall_s": "Wall (s)",
}
MUTATION_ORDER = ["identical", "distrust", "steer", "edit", "patch_question"]
SYSTEM_ORDER = ["full", "no_debate", "no_critic", "single_shot", "no_context_tools", "single_agent"]
OK_STATUSES = {"done", "partial", "imported"}
REFERENCE = "full"
N_BOOT = 2000
N_BINS = 10


# --------------------------------------------------------------------------- loading


@dataclass
class Results:
    runs: list[RunRecord] = field(default_factory=list)
    recompute: list[RecomputeRecord] = field(default_factory=list)
    scores: list[ScoreRecord] = field(default_factory=list)
    pairwise: list[PairwiseRecord] = field(default_factory=list)
    labels: list[ClaimLabel] = field(default_factory=list)
    judge_costs: list[JudgeCost] = field(default_factory=list)
    manifests: list[Manifest] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    skipped_lines: int = 0


FILES: dict[str, tuple[str, type[BaseModel]]] = {
    "runs.jsonl": ("runs", RunRecord),
    "recompute.jsonl": ("recompute", RecomputeRecord),
    "scores.jsonl": ("scores", ScoreRecord),
    "pairwise.jsonl": ("pairwise", PairwiseRecord),
    "claim_labels.jsonl": ("labels", ClaimLabel),
    "judge_cost.jsonl": ("judge_costs", JudgeCost),
}


def _read_jsonl(path: Path, model: type[T]) -> tuple[list[T], int]:
    out: list[T] = []
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(model.model_validate_json(line))
        except ValidationError:
            bad += 1
    return out, bad


def load_results(dirs: Iterable[Path]) -> Results:
    """Reads every known JSONL file (and manifest.json) found under the given directories, recursively."""
    res = Results()
    seen: set[Path] = set()
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        candidates = [d] if d.is_file() else sorted(d.rglob("*"))
        for p in candidates:
            rp = p.resolve()
            if rp in seen or not p.is_file():
                continue
            if p.name in FILES:
                seen.add(rp)
                attr, model = FILES[p.name]
                records, bad = _read_jsonl(p, model)
                getattr(res, attr).extend(records)
                res.skipped_lines += bad
                res.sources.append(str(p))
            elif p.name == "manifest.json":
                seen.add(rp)
                try:
                    res.manifests.append(Manifest.model_validate_json(p.read_text(encoding="utf-8")))
                except ValidationError:
                    res.skipped_lines += 1
    return res


# --------------------------------------------------------------------------- statistics


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs: Sequence[float], *, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (seeded, so reports are reproducible)."""
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], xs[0]
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(math.floor(alpha / 2 * (n_boot - 1)))]
    hi = means[int(math.ceil((1 - alpha / 2) * (n_boot - 1)))]
    return lo, hi


def stat(xs: Sequence[float], *, seed: int = 0) -> dict[str, float | int]:
    lo, hi = bootstrap_ci(xs, seed=seed)
    return {"mean": mean(xs), "ci_low": lo, "ci_high": hi, "n": len(xs)}


def per_question(items: Iterable[tuple[str, float]]) -> list[float]:
    """Averages repeats per question first: the question is the bootstrap unit (repeats are not independent)."""
    by_q: dict[str, list[float]] = defaultdict(list)
    for q, v in items:
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            by_q[q].append(float(v))
    return [mean(v) for _, v in sorted(by_q.items())]


def wilson(p: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def brier(conf: Sequence[float], y: Sequence[int]) -> float:
    return mean([(c - t) ** 2 for c, t in zip(conf, y, strict=True)])


def reliability_bins(conf: Sequence[float], y: Sequence[int], n_bins: int = N_BINS) -> list[dict[str, float | int]]:
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for c, t in zip(conf, y, strict=True):
        bins[min(n_bins - 1, max(0, int(c * n_bins)))].append((c, t))
    out = []
    for i, b in enumerate(bins):
        out.append(
            {
                "lo": i / n_bins,
                "hi": (i + 1) / n_bins,
                "n": len(b),
                "conf": mean([c for c, _ in b]) if b else None,
                "acc": mean([t for _, t in b]) if b else None,
            }
        )
    return out


def ece(conf: Sequence[float], y: Sequence[int], n_bins: int = N_BINS) -> float:
    n = len(conf)
    if not n:
        return float("nan")
    return sum(b["n"] / n * abs(b["acc"] - b["conf"]) for b in reliability_bins(conf, y, n_bins) if b["n"])


def auroc(conf: Sequence[float], y: Sequence[int]) -> float | None:
    """Mann–Whitney AUROC with average ranks for ties; None when only one class is present."""
    pos = sum(y)
    neg = len(y) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(conf)), key=lambda i: conf[i])
    ranks = [0.0] * len(conf)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and conf[order[j + 1]] == conf[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_pos = sum(r for r, t in zip(ranks, y, strict=True) if t)
    return (rank_pos - pos * (pos + 1) / 2) / (pos * neg)


# --------------------------------------------------------------------------- computations


def _sort_systems(names: Iterable[str]) -> list[str]:
    names = set(names)
    known = [s for s in SYSTEM_ORDER if s in names]
    return known + sorted(names - set(known))


def _try_metrics_aggregate(scores: list[ScoreRecord]) -> dict[str, dict[str, dict[str, Any]]] | None:
    """Uses lgeval.metrics.aggregate when it exists and returns the documented shape; else None."""
    try:
        from lgeval.metrics import aggregate  # type: ignore[attr-defined]
    except Exception:
        return None
    try:
        agg = aggregate(scores)
        for metrics in agg.values():
            for v in metrics.values():
                if not {"mean", "ci_low", "ci_high", "n"} <= set(v):
                    return None
        return agg
    except Exception:
        return None


def primary_judge(scores: list[ScoreRecord]) -> str | None:
    """The judge model behind most quality scores; scores from different judges are never mixed."""
    counts: dict[str | None, int] = defaultdict(int)
    for s in scores:
        if s.metric in QUALITY_METRICS:
            counts[s.judge_model] += 1
    return max(sorted(counts, key=str), key=lambda k: counts[k]) if counts else None


def quality_scores(res: Results) -> list[ScoreRecord]:
    """Quality scores of the primary judge, without undefined ones (n == 0: no facts, judge error)."""
    judge = primary_judge(res.scores)
    return [s for s in res.scores if s.metric in QUALITY_METRICS and s.judge_model == judge and s.n != 0]


def compute_quality(res: Results) -> dict[str, dict[str, dict[str, Any]]]:
    scores = quality_scores(res)
    systems = {s.system for s in scores} | {r.system for r in res.runs if r.status in OK_STATUSES}
    if not systems:
        return {}
    external = _try_metrics_aggregate(scores) if scores else None
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for sys in _sort_systems(systems):
        row: dict[str, dict[str, Any]] = {}
        for m in QUALITY_METRICS:
            if external and m in external.get(sys, {}):
                row[m] = dict(external[sys][m])
                continue
            vals = per_question((s.question_id, s.value) for s in scores if s.system == sys and s.metric == m)
            if vals:
                row[m] = stat(vals)
        ok = [r for r in res.runs if r.system == sys and r.status in OK_STATUSES]
        for m in ("cost_usd", "wall_s"):
            vals = per_question((r.question_id, getattr(r, m)) for r in ok if getattr(r, m) is not None)
            if vals:
                row[m] = stat(vals)
        runs = [r for r in res.runs if r.system == sys]
        if runs:
            row["runs"] = {"total": len(runs), "ok": len(ok)}
        if row:
            out[sys] = row
    return out


def compute_pairwise(res: Results, reference: str = REFERENCE) -> dict[str, dict[str, Any]]:
    """Reference system's win rate against each opponent; ties count half. Wilson 95% CI."""
    tally: dict[str, dict[str, int]] = defaultdict(lambda: {"win": 0, "tie": 0, "loss": 0})
    for p in res.pairwise:
        if p.system_a == reference and p.system_b != reference:
            opp, outcome = p.system_b, {"a": "win", "b": "loss", "tie": "tie"}[p.winner]
        elif p.system_b == reference and p.system_a != reference:
            opp, outcome = p.system_a, {"b": "win", "a": "loss", "tie": "tie"}[p.winner]
        else:
            continue
        tally[opp][outcome] += 1
    out: dict[str, dict[str, Any]] = {}
    for opp in _sort_systems(tally):
        t = tally[opp]
        n = t["win"] + t["tie"] + t["loss"]
        rate = (t["win"] + 0.5 * t["tie"]) / n
        lo, hi = wilson(rate, n)
        out[opp] = {**t, "n": n, "win_rate": rate, "ci_low": lo, "ci_high": hi}
    return out


def compute_calibration(res: Results) -> dict[str, dict[str, Any]]:
    by_sys: dict[str, list[ClaimLabel]] = defaultdict(list)
    for lab in res.labels:
        by_sys[lab.system].append(lab)
    out: dict[str, dict[str, Any]] = {}
    for sys in _sort_systems(by_sys):
        labs = [x for x in by_sys[sys] if x.label != "unverifiable"]
        scored = [x for x in labs if x.confidence is not None]
        conf = [min(1.0, max(0.0, float(x.confidence))) for x in scored]  # type: ignore[arg-type]
        y = [1 if x.label == "true" else 0 for x in scored]
        struck = [x for x in labs if x.struck]
        row: dict[str, Any] = {
            "n": len(scored),
            "n_unverifiable": sum(1 for x in by_sys[sys] if x.label == "unverifiable"),
            "accuracy": mean(y) if y else None,
            "brier": brier(conf, y) if y else None,
            "ece": ece(conf, y) if y else None,
            "auroc": auroc(conf, y) if y else None,
            "n_struck": len(struck),
            "struck_false_rate": mean([1 if x.label == "false" else 0 for x in struck]) if struck else None,
            "bins": reliability_bins(conf, y) if y else [],
        }
        out[sys] = row
    if not out:
        # Fall back to precomputed metric scores (no labels shipped with the results).
        for sys in _sort_systems({s.system for s in res.scores if s.metric in ("brier", "ece", "auroc")}):
            row = {"bins": []}
            for m in ("brier", "ece", "auroc"):
                vals = [s.value for s in res.scores if s.system == sys and s.metric == m]
                row[m] = mean(vals) if vals else None
            row["n"] = sum(s.n or 0 for s in res.scores if s.system == sys and s.metric == "brier") or None
            row["struck_false_rate"] = None
            out[sys] = row
    return out


def compute_recompute(res: Results) -> dict[str, dict[str, Any]]:
    by_mut: dict[str, list[RecomputeRecord]] = defaultdict(list)
    for r in res.recompute:
        by_mut[r.mutation].append(r)
    order = [m for m in MUTATION_ORDER if m in by_mut] + sorted(set(by_mut) - set(MUTATION_ORDER))
    out: dict[str, dict[str, Any]] = {}
    for mut in order:
        recs = by_mut[mut]
        cost = per_question(
            (r.question_id, 100 * r.cost_usd / r.baseline_cost_usd) for r in recs if r.baseline_cost_usd > 0
        )
        time = per_question((r.question_id, 100 * r.wall_s / r.baseline_wall_s) for r in recs if r.baseline_wall_s > 0)
        out[mut] = {
            "n": len(recs),
            "cost_pct": stat(cost),
            "time_pct": stat(time),
            "cost_usd": mean([r.cost_usd for r in recs]),
            "baseline_cost_usd": mean([r.baseline_cost_usd for r in recs]),
            "wall_s": mean([r.wall_s for r in recs]),
            "baseline_wall_s": mean([r.baseline_wall_s for r in recs]),
            "nodes_total": mean([r.nodes_total for r in recs]),
            "nodes_impacted": mean([r.nodes_impacted for r in recs]),
            "nodes_recomputed": mean([r.nodes_recomputed for r in recs]),
            "nodes_reused": mean([r.nodes_reused for r in recs]),
            "baselines": sorted({r.baseline for r in recs}),
        }
    return out


def compute_study_cost(res: Results) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = {}

    def add(name: str, usd: float | None, wall: float | None, count: int = 1) -> None:
        row = rows.setdefault(name, {"usd": 0.0, "wall_s": 0.0, "count": 0})
        row["usd"] += usd or 0.0
        row["wall_s"] += wall or 0.0
        row["count"] += count

    for r in res.runs:
        kind = "external (imported)" if r.status == "imported" else f"runs: {r.system}"
        add(kind, r.cost_usd, r.wall_s)
    for r in res.recompute:
        add(f"recompute: {r.mutation}", r.cost_usd, r.wall_s)
        if r.baseline == "fresh_run":
            add("recompute: fresh-run baselines", r.baseline_cost_usd, r.baseline_wall_s)
    judge = [s for s in res.scores if isinstance(s.details.get("cost_usd"), int | float)]
    for s in judge:
        add("judging (scores)", float(s.details["cost_usd"]), None)
    for c in res.judge_costs:
        add(f"judging ({c.command})", c.usd, None)
    if not rows:
        return {}
    total_usd = sum(r["usd"] for r in rows.values())
    total_wall = sum(r["wall_s"] for r in rows.values())
    return {
        "rows": rows,
        "total_usd": total_usd,
        "total_wall_h": total_wall / 3600,
        "judge_cost_recorded": bool(judge or res.judge_costs),
        "n_scores": len(res.scores),
        "n_pairwise": len(res.pairwise),
    }


def summarize(res: Results) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": {
            "files": res.sources,
            "skipped_lines": res.skipped_lines,
            "counts": {
                "runs": len(res.runs),
                "recompute": len(res.recompute),
                "scores": len(res.scores),
                "pairwise": len(res.pairwise),
                "claim_labels": len(res.labels),
            },
            "models": _models(res),
            "git_sha": sorted({m.git_sha for m in res.manifests if m.git_sha}),
            "judge_model": primary_judge(res.scores),
            "judge_models_seen": sorted({s.judge_model for s in res.scores if s.judge_model}),
        },
        "quality": compute_quality(res),
        "pairwise": compute_pairwise(res),
        "calibration": compute_calibration(res),
        "recompute": compute_recompute(res),
        "study_cost": compute_study_cost(res),
    }


def _models(res: Results) -> dict[str, list[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for m in res.manifests:
        for role, model in m.models.items():
            out[role].add(model)
    return {k: sorted(v) for k, v in sorted(out.items())}


# --------------------------------------------------------------------------- formatting


def _ok(x: Any) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def f(x: Any, digits: int = 2) -> str:
    return f"{x:.{digits}f}" if _ok(x) else "–"


def fci(s: dict[str, Any] | None, digits: int = 2, *, latex: bool = False) -> str:
    if not s or not _ok(s.get("mean")):
        return "–"
    if s.get("n", 0) < 2 or not _ok(s.get("ci_low")):
        return f(s["mean"], digits)
    lo, hi = f(s["ci_low"], digits), f(s["ci_high"], digits)
    if latex:
        return f"{f(s['mean'], digits)} {{\\scriptsize[{lo}, {hi}]}}"
    return f"{f(s['mean'], digits)} [{lo}, {hi}]"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                 "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def tex(s: str) -> str:
    return "".join(LATEX_SPECIAL.get(c, c) for c in s)


def latex_table(header: list[str], rows: list[list[str]], caption: str, label: str, *, wide: bool = False) -> str:
    env = "table*" if wide else "table"
    cols = "l" + "r" * (len(header) - 1)
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (
        f"% generated by `python -m lgeval report`; do not edit\n"
        f"\\begin{{{env}}}[t]\n\\centering\\small\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
        f"{' & '.join(header)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n"
    )


def latex_pending(caption: str, label: str) -> str:
    return (
        "% generated by `python -m lgeval report`; no data yet\n"
        f"\\begin{{table}}[t]\n\\centering\\small\n\\begin{{tabular}}{{l}}\n\\toprule\n"
        f"Results pending. \\\\\n\\bottomrule\n\\end{{tabular}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )


@dataclass
class Table:
    key: str
    title: str
    caption: str
    header: list[str]
    rows: list[list[str]]
    tex_header: list[str]
    tex_rows: list[list[str]]
    note: str = ""
    wide: bool = False


def quality_table(q: dict[str, dict[str, dict[str, Any]]]) -> Table | None:
    if not q:
        return None
    cols = [m for m in [*QUALITY_METRICS, "cost_usd", "wall_s"] if any(m in row for row in q.values())]
    digits = {"cost_usd": 3, "wall_s": 0}
    rows, tex_rows = [], []
    for sys, row in q.items():
        rows.append([f"`{sys}`"] + [fci(row.get(m), digits.get(m, 2)) for m in cols])
        tex_rows.append([tex(sys)] + [fci(row.get(m), digits.get(m, 2), latex=True) for m in cols])
    return Table(
        "quality",
        "Table 1. Quality (Claim A)",
        "Quality by system: mean over questions (repeats averaged per question) with 95\\% bootstrap CI.",
        ["System", *[METRIC_LABELS[m] for m in cols]],
        rows,
        ["System", *[tex(METRIC_LABELS[m]) for m in cols]],
        tex_rows,
        note="Mean over questions (repeats averaged per question first), 95% percentile bootstrap CI in brackets.",
        wide=True,
    )


def pairwise_table(p: dict[str, dict[str, Any]]) -> Table | None:
    if not p:
        return None
    rows, tex_rows = [], []
    for opp, r in p.items():
        cells = [str(r["win"]), str(r["tie"]), str(r["loss"]), str(r["n"])]
        ci = f"{f(r['win_rate'])} [{f(r['ci_low'])}, {f(r['ci_high'])}]"
        rows.append([f"`{opp}`", *cells, ci])
        tex_rows.append([tex(opp), *cells, f"{f(r['win_rate'])} {{\\scriptsize[{f(r['ci_low'])}, {f(r['ci_high'])}]}}"])
    header = ["vs. system", "Wins", "Ties", "Losses", "n", "Win rate [95% CI]"]
    return Table(
        "pairwise",
        f"Table 2. Pairwise preference of `{REFERENCE}`",
        f"Pairwise judge preference of \\texttt{{{REFERENCE}}} against each system (both orderings reconciled); "
        "ties count half; Wilson 95\\% CI.",
        header,
        rows,
        [*header[:5], r"Win rate [95\% CI]"],
        tex_rows,
        note=f"Win rate of `{REFERENCE}` = (wins + ties/2) / n, after reconciling both presentation orders; Wilson 95% CI.",
    )


def calibration_table(c: dict[str, dict[str, Any]]) -> Table | None:
    if not c:
        return None
    rows = []
    for sys, r in c.items():
        rows.append([sys, f(r.get("brier"), 3), f(r.get("ece"), 3), f(r.get("auroc"), 3), str(r.get("n") or "–"),
                     f(r.get("struck_false_rate"), 2)])
    header = ["System", "Brier ↓", "ECE ↓", "AUROC ↑", "n claims", "Struck-false rate"]
    return Table(
        "calibration",
        "Table 3. Calibration of judge confidence (Claim B)",
        "Calibration of per-claim judge confidence against claim truth labels (unverifiable claims excluded; "
        f"ECE with {N_BINS} equal-width bins). Struck-false rate: share of struck claims labelled false.",
        header,
        [[f"`{r[0]}`", *r[1:]] for r in rows],
        ["System", r"Brier $\downarrow$", r"ECE $\downarrow$", r"AUROC $\uparrow$", "$n$", "Struck-false"],
        [[tex(r[0]), *r[1:]] for r in rows],
        note=f"Unverifiable claims excluded. ECE uses {N_BINS} equal-width bins. "
        "Struck-false rate = share of claims the judge struck that are labelled false.",
    )


def recompute_table(r: dict[str, dict[str, Any]]) -> Table | None:
    if not r:
        return None
    rows, tex_rows = [], []
    for mut, s in r.items():
        nodes = [f(s["nodes_recomputed"], 1), f(s["nodes_reused"], 1), f(s["nodes_impacted"], 1), f(s["nodes_total"], 1)]
        rows.append([f"`{mut}`", str(s["n"]), fci(s["cost_pct"], 1), fci(s["time_pct"], 1), *nodes])
        tex_rows.append([tex(mut), str(s["n"]), fci(s["cost_pct"], 1, latex=True), fci(s["time_pct"], 1, latex=True),
                         *nodes])
    header = ["Mutation", "n", "Cost % of baseline", "Time % of baseline", "Recomputed", "Reused", "Impacted", "Total"]
    return Table(
        "recompute",
        "Table 4. Incremental recompute (Claim C)",
        "Cost and wall time of an update as \\% of starting over (mean, 95\\% bootstrap CI), and mean node counts.",
        header,
        rows,
        ["Mutation", "$n$", r"Cost \%", r"Time \%", "Recomp.", "Reused", "Impacted", "Total"],
        tex_rows,
        note="Per-update ratio to the baseline (fresh run or original run), averaged per question; "
        "node counts are means. Impacted = GET /impact upper bound.",
        wide=True,
    )


def study_cost_table(s: dict[str, Any]) -> Table | None:
    if not s:
        return None
    rows = [[k, str(int(v["count"])), f(v["usd"], 2), f(v["wall_s"] / 3600, 2)] for k, v in sorted(s["rows"].items())]
    rows.append(["**total**", "", f"**{f(s['total_usd'], 2)}**", f"**{f(s['total_wall_h'], 2)}**"])
    tex_rows = [[tex(r[0]), r[1], r[2], r[3]] for r in rows[:-1]]
    tex_rows.append([r"\textbf{total}", "", f(s["total_usd"], 2), f(s["total_wall_h"], 2)])
    note = "Sum of measured spend recorded in the result files."
    if not s.get("judge_cost_recorded"):
        note += " Judge/scorer spend was not recorded per score and is not included."
    return Table(
        "study_cost",
        "Table 5. Cost of the study",
        "Measured cost of the study itself (USD, wall-clock hours).",
        ["Item", "Count", "USD", "Wall (h)"],
        rows,
        ["Item", "Count", r"USD", "Wall (h)"],
        tex_rows,
        note=note,
    )


# --------------------------------------------------------------------------- SVG charts

SVG_STYLE = """<style>
  svg { color: #0b0b0b; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 12px; }
  .surface { fill: #fcfcfb; }
  .ink { fill: currentColor; }
  .muted { fill: #52514e; }
  .grid { stroke: #e4e3df; stroke-width: 1; }
  .axis { stroke: #8a8984; stroke-width: 1; }
  .ref { stroke: #8a8984; stroke-width: 1; stroke-dasharray: 4 4; fill: none; }
  .whisker { stroke: currentColor; stroke-width: 1.5; }
  .s1 { fill: #2a78d6; stroke: #2a78d6; } .s2 { fill: #eb6834; stroke: #eb6834; } .s3 { fill: #1baf7a; stroke: #1baf7a; }
  .line { fill: none !important; stroke-width: 2; }
  @media (prefers-color-scheme: dark) {
    svg { color: #f0efec; }
    .surface { fill: #1a1a19; }
    .muted { fill: #c3c2b7; }
    .grid { stroke: #2e2e2c; }
    .axis, .ref { stroke: #6f6e69; }
    .s1 { fill: #3987e5; stroke: #3987e5; } .s2 { fill: #d95926; stroke: #d95926; } .s3 { fill: #199e70; stroke: #199e70; }
  }
</style>"""


def _svg(width: int, height: int, title: str, desc: str, body: list[str], ident: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-labelledby="{ident}-title {ident}-desc">\n'
        f'<title id="{ident}-title">{escape(title)}</title>\n<desc id="{ident}-desc">{escape(desc)}</desc>\n'
        f"{SVG_STYLE}\n"
        f'<rect class="surface" x="0" y="0" width="{width}" height="{height}" rx="6"/>\n' + "\n".join(body) + "\n</svg>\n"
    )


def _text(x: float, y: float, s: str, *, cls: str = "ink", anchor: str = "start", size: int | None = None,
          weight: str | None = None, rotate: bool = False) -> str:
    extra = f' font-size="{size}"' if size else ""
    extra += f' font-weight="{weight}"' if weight else ""
    extra += f' transform="rotate(-90 {x:.1f} {y:.1f})"' if rotate else ""
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{extra}>{escape(s)}</text>'


def _legend(x: float, y: float, items: list[tuple[str, str]]) -> list[str]:
    out = []
    for i, (cls, label) in enumerate(items):
        xi = x + i * 150
        out.append(f'<rect class="{cls}" x="{xi:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2"/>')
        out.append(_text(xi + 15, y, label, cls="muted"))
    return out


def reliability_svg(cal: dict[str, dict[str, Any]]) -> str | None:
    series = [(s, r) for s, r in cal.items() if any(b["n"] for b in r.get("bins", []))][:3]
    if not series:
        return None
    w, h, ml, mr, mt, mb = 440, 420, 56, 20, 56, 56
    pw, ph = w - ml - mr, h - mt - mb
    sx = lambda v: ml + v * pw  # noqa: E731
    sy = lambda v: mt + (1 - v) * ph  # noqa: E731
    body = [_text(ml, 24, "Reliability diagram", weight="600", size=14)]
    if len(series) > 1:
        body += _legend(ml, 44, [(f"s{i + 1}", s) for i, (s, _) in enumerate(series)])
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        body.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        body.append(_text(ml - 8, sy(t) + 4, f"{t:.2f}", cls="muted", anchor="end"))
        body.append(_text(sx(t), mt + ph + 18, f"{t:.2f}", cls="muted", anchor="middle"))
    body.append(f'<line class="ref" x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}"/>')
    body.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    body.append(_text(ml + pw / 2, h - 12, "Judge confidence (bin mean)", cls="muted", anchor="middle"))
    body.append(_text(16, mt + ph / 2, "Share of claims true", cls="muted", anchor="middle", rotate=True))
    for i, (sys, r) in enumerate(series):
        pts = [b for b in r["bins"] if b["n"]]
        cls = f"s{i + 1}"
        path = " ".join(f"{'M' if j == 0 else 'L'}{sx(b['conf']):.1f},{sy(b['acc']):.1f}" for j, b in enumerate(pts))
        body.append(f'<path class="{cls} line" d="{path}"/>')
        nmax = max(b["n"] for b in pts)
        for b in pts:
            rad = 4 + 5 * math.sqrt(b["n"] / nmax)
            body.append(
                f'<circle class="{cls}" cx="{sx(b["conf"]):.1f}" cy="{sy(b["acc"]):.1f}" r="{rad:.1f}" '
                f'fill-opacity="0.85" stroke-width="2"><title>{escape(sys)}: confidence {b["conf"]:.2f}, '
                f'true {b["acc"]:.2f}, n={b["n"]}</title></circle>'
            )
    names = ", ".join(s for s, _ in series)
    desc = (
        f"Mean judge confidence per bin against the share of claims labelled true, for {names}; "
        "the dashed diagonal is perfect calibration; marker area grows with the number of claims in the bin."
    )
    return _svg(w, h, "Reliability diagram of judge confidence", desc, body, "rel")


def _bars(
    title: str,
    desc: str,
    ident: str,
    groups: list[str],
    series: list[tuple[str, str, list[dict[str, Any] | None]]],
    *,
    y_label: str,
    fmt: Callable[[float], str],
    y_max: float | None = None,
) -> str:
    """Grouped vertical bars with optional CI whiskers. series = [(css class, legend label, per-group stat)]."""
    group_w = max(70, 30 * len(series) + 30)
    ml, mr, mt, mb = 64, 20, 64, 64
    w = ml + mr + group_w * len(groups)
    w = max(w, 360)
    h = 340
    pw, ph = w - ml - mr, h - mt - mb
    vals = [s["ci_high"] if _ok(s.get("ci_high")) else s["mean"] for _, _, ss in series for s in ss if s and _ok(s.get("mean"))]
    top = y_max if y_max is not None else _nice(max(vals) if vals else 1.0)
    sy = lambda v: mt + (1 - min(v, top) / top) * ph  # noqa: E731
    body = [_text(ml, 24, title, weight="600", size=14)]
    if len(series) > 1:
        body += _legend(ml, 46, [(cls, label) for cls, label, _ in series])
    for i in range(5):
        t = top * i / 4
        body.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        body.append(_text(ml - 8, sy(t) + 4, fmt(t), cls="muted", anchor="end"))
    body.append(_text(16, mt + ph / 2, y_label, cls="muted", anchor="middle", rotate=True))
    gw = pw / len(groups)
    bw = min(28.0, (gw - 16) / len(series) - 2)
    for gi, g in enumerate(groups):
        gx = ml + gi * gw + (gw - len(series) * (bw + 2)) / 2
        body.append(_text(ml + gi * gw + gw / 2, mt + ph + 18, g, cls="ink", anchor="middle"))
        for si, (cls, label, stats) in enumerate(series):
            s = stats[gi]
            if not s or not _ok(s.get("mean")):
                continue
            x = gx + si * (bw + 2)
            y = sy(s["mean"])
            bh = mt + ph - y
            r = min(4.0, bw / 2, bh)
            # Rounded data end, square at the baseline.
            path = (f"M{x:.1f},{mt + ph:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
                    f"L{x + bw - r:.1f},{y:.1f} Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} L{x + bw:.1f},{mt + ph:.1f} Z")
            ci = ""
            if s.get("n", 0) >= 2 and _ok(s.get("ci_low")):
                ci = f" [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}]"
            body.append(f'<path class="{cls}" d="{path}"><title>{escape(g)} · {escape(label)}: '
                        f'{fmt(s["mean"])}{ci}</title></path>')
            if ci:
                cx = x + bw / 2
                y1, y2 = sy(s["ci_low"]), sy(s["ci_high"])
                body.append(f'<line class="whisker" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y1:.1f}" y2="{y2:.1f}"/>')
                for yy in (y1, y2):
                    body.append(f'<line class="whisker" x1="{cx - 4:.1f}" x2="{cx + 4:.1f}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
    body.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    return _svg(w, h, title, desc, body, ident)


def _nice(v: float) -> float:
    if v <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * exp:
            return m * exp
    return 10 * exp


def recompute_svg(rec: dict[str, dict[str, Any]]) -> str | None:
    if not rec:
        return None
    muts = list(rec)
    base = [{"mean": rec[m]["baseline_cost_usd"], "n": 1} for m in muts]
    live = [{"mean": rec[m]["cost_usd"], "n": 1} for m in muts]
    desc = ("Mean cost in USD of starting over (baseline) versus a LiveGraph incremental update, per mutation: "
            + "; ".join(f"{m}: {rec[m]['baseline_cost_usd']:.3f} vs {rec[m]['cost_usd']:.3f}" for m in muts) + ".")
    return _bars("Update cost per mutation", desc, "rc", muts,
                 [("s2", "Start over (baseline)", base), ("s1", "LiveGraph update", live)],
                 y_label="Mean cost (USD)", fmt=lambda v: f"{v:.3f}" if v < 1 else f"{v:.2f}")


def quality_svg(q: dict[str, dict[str, dict[str, Any]]]) -> str | None:
    metrics = [m for m in QUALITY_METRICS if any(m in row for row in q.values())][:3]
    systems = [s for s in q if any(m in q[s] for m in metrics)][:8]
    if not metrics or not systems:
        return None
    series = [(f"s{i + 1}", METRIC_LABELS[m], [q[s].get(m) for s in systems]) for i, m in enumerate(metrics)]
    desc = "Mean score per system with 95% bootstrap CI whiskers: " + "; ".join(
        f"{s}: " + ", ".join(f"{METRIC_LABELS[m]} {f(q[s][m]['mean'])}" for m in metrics if m in q[s]) for s in systems
    ) + "."
    return _bars("Quality by system", desc, "q", systems, series, y_label="Score (0–1)", fmt=lambda v: f"{v:.2f}",
                 y_max=1.0)


# --------------------------------------------------------------------------- writing


def _jsonable(x: Any) -> Any:
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    return x


TEX_LABELS = {
    "quality": ("tab:quality", "Quality by system."),
    "pairwise": ("tab:pairwise", f"Pairwise preference of {REFERENCE}."),
    "calibration": ("tab:calibration", "Calibration of judge confidence."),
    "recompute": ("tab:recompute", "Incremental recompute cost."),
    "study_cost": ("tab:study_cost", "Cost of the study."),
}


def build_report(res: Results, out: Path) -> dict[str, Any]:
    """Writes results.md, summary.json, tables/*.tex and figures/*.svg into `out`; returns the summary."""
    summary = summarize(res)
    tables: list[tuple[str, str, Table | None]] = [
        ("quality", "Table 1. Quality (Claim A)", quality_table(summary["quality"])),
        ("pairwise", f"Table 2. Pairwise preference of `{REFERENCE}`", pairwise_table(summary["pairwise"])),
        ("calibration", "Table 3. Calibration (Claim B)", calibration_table(summary["calibration"])),
        ("recompute", "Table 4. Incremental recompute (Claim C)", recompute_table(summary["recompute"])),
        ("study_cost", "Table 5. Cost of the study", study_cost_table(summary["study_cost"])),
    ]
    figures = {
        "quality.svg": quality_svg(summary["quality"]),
        "reliability.svg": reliability_svg(summary["calibration"]),
        "recompute_cost.svg": recompute_svg(summary["recompute"]),
    }
    missing_why = {
        "quality": "no `scores.jsonl` or successful `runs.jsonl` records",
        "pairwise": f"no `pairwise.jsonl` records involving `{REFERENCE}`",
        "calibration": "no `claim_labels.jsonl` records and no brier/ece scores",
        "recompute": "no `recompute.jsonl` records",
        "study_cost": "no records with measured cost",
    }
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    md = ["# LiveGraph evaluation results", "",
          f"Generated by `python -m lgeval report` at {summary['generated_at']}. "
          "All numbers are also in [`summary.json`](summary.json); methodology in [`../EVAL.md`](../EVAL.md).", ""]
    c = summary["inputs"]["counts"]
    md.append("Inputs: " + ", ".join(f"{v} {k}" for k, v in c.items())
              + (f" ({res.skipped_lines} unreadable lines skipped)" if res.skipped_lines else "") + ".")
    if summary["inputs"]["models"]:
        md.append("Models: " + "; ".join(f"{k} = {', '.join(v)}" for k, v in summary["inputs"]["models"].items()) + ".")
    judges = summary["inputs"]["judge_models_seen"]
    if judges:
        md.append(f"Quality judge: `{summary['inputs']['judge_model']}`"
                  + (f" (other judges in the inputs are ignored: {', '.join(j for j in judges if j != summary['inputs']['judge_model'])})"
                     if len(judges) > 1 else "") + ".")
    md.append("")
    notes: list[str] = []
    for key, title, t in tables:
        label, caption = TEX_LABELS[key]
        md.append(f"## {title}")
        md.append("")
        if t is None:
            note = f"_Omitted: {missing_why[key]}._"
            notes.append(f"{key}: omitted ({missing_why[key]})")
            md += [note, ""]
            (out / "tables" / f"{key}.tex").write_text(latex_pending(caption, label), encoding="utf-8")
            continue
        md += [md_table(t.header, t.rows), "", f"_{t.note}_", ""]
        (out / "tables" / f"{key}.tex").write_text(
            latex_table(t.tex_header, t.tex_rows, t.caption, label, wide=t.wide), encoding="utf-8"
        )
        fig = {"quality": "quality.svg", "calibration": "reliability.svg", "recompute": "recompute_cost.svg"}.get(key)
        if fig and figures.get(fig):
            md += [f"![{t.title}](figures/{fig})", ""]
    for name, svg in figures.items():
        path = out / "figures" / name
        if svg:
            path.write_text(svg, encoding="utf-8")
        elif path.exists():
            path.unlink()  # stale figure from an earlier report
    summary["notes"] = notes
    summary["figures"] = sorted(n for n, s in figures.items() if s)
    (out / "results.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str) + "\n", encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- CLI


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("report", help="results → markdown/LaTeX tables, SVG charts, summary.json")
    p.add_argument("--results", nargs="+", type=Path, required=True, help="result dirs (searched recursively)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output dir (default {DEFAULT_OUT})")
    p.set_defaults(func=cmd_report)


def cmd_report(args: argparse.Namespace) -> int:
    res = load_results(args.results)
    summary = build_report(res, args.out)
    counts = summary["inputs"]["counts"]
    print(f"report: {sum(counts.values())} records from {len(res.sources)} files -> {args.out}")
    for n in summary["notes"]:
        print(f"  note: {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lgeval.report")
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["report"]:
        argv = ["report", *argv]
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
