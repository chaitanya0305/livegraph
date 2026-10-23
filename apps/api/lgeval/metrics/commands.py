"""`score` and `pairwise` subcommands: judge the RunRecords of results dirs, with resume, dry-run and a USD cap."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from lgeval.metrics.llm import DEFAULT_JUDGE_MODEL, LLM, BudgetExhausted, default_root, make_llm
from lgeval.metrics.stats import aggregate
from lgeval.metrics.text import PageFetcher, resolve_citations
from lgeval.schema import ClaimLabel, PairwiseRecord, Question, RunRecord, ScoreRecord

log = logging.getLogger("lgeval.metrics")

ALL_METRICS = ("factual", "citation", "accuracy", "calibration")
# metric group -> the ScoreRecord.metric whose presence marks the group done for a run (resume)
DONE_METRIC = {"factual": "factual_precision", "citation": "citation_recall", "accuracy": "answer_accuracy",
               "calibration": "brier"}
SCORABLE = {"done", "imported"}


# --- io -----------------------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_jsonl(path: Path, records: list[Any]) -> None:
    if records:
        with path.open("a", encoding="utf-8") as fh:
            for r in records:
                fh.write(r.model_dump_json() + "\n")


def load_runs(dirs: list[Path]) -> list[tuple[Path, RunRecord]]:
    return [(d, RunRecord.model_validate(r)) for d in dirs for r in read_jsonl(d / "runs.jsonl")]


def load_question_map(dirs: list[Path], runs: list[RunRecord]) -> dict[str, Question]:
    """Questions by id: `questions.jsonl` in a results dir first, then the dataset loader."""
    out: dict[str, Question] = {}
    for d in dirs:
        for r in read_jsonl(d / "questions.jsonl"):
            q = Question.model_validate(r)
            out.setdefault(q.id, q)
    missing = {r.dataset for r in runs if r.question_id not in out}
    for name in sorted(missing):
        try:
            from lgeval.datasets import load_questions

            for q in load_questions(name):
                out.setdefault(q.id, q)
        except Exception as exc:  # unknown/offline dataset: score without gold
            log.warning("could not load dataset %r (%s); scoring without question text/gold", name, exc)
    return out


def _budget(max_usd: float | None):
    from lgeval.costs import Budget

    return Budget(max_usd)


def _price(model: str) -> tuple[float, float]:
    try:
        from lgeval.costs import load_pricing

        p = load_pricing([model]).get(model)
        if p is not None:
            return p.in_per_m, p.out_per_m
    except Exception:
        pass
    return 0.40, 1.60


# --- estimates ----------------------------------------------------------------------------------------------


def estimate(run: RunRecord, question: Question | None, metric: str) -> tuple[int, int, int]:
    """(calls, tokens_in, tokens_out) the judge will need for `metric` on `run` (chars/4 estimate, no cache)."""
    brief_t = len(run.brief) // 4
    if metric == "factual":
        n_sent = len(resolve_citations(run)) if run.brief else 0
        n_facts = min(40, math.ceil(1.5 * n_sent))
        batches = math.ceil(n_facts / 8)
        ev = min(12, len(run.evidence)) * 300
        return 1 + batches, 700 + brief_t + batches * (500 + ev + 250), n_facts * 20 + batches * 320
    if metric == "citation":
        cits = [c for c in resolve_citations(run) if c.chunk_ids or c.urls]
        passages = sum(min(6, len(c.chunk_ids) + len(c.urls)) for c in cits)
        calls = math.ceil(len(cits) / 4)
        return calls, calls * 450 + passages * 375 + len(cits) * 40, len(cits) * 45
    if metric == "accuracy":
        if question is None or not question.gold_answer:
            return 0, 0, 0
        return 1, 450 + brief_t, 60
    if metric == "calibration":
        calls = math.ceil(len(run.claims) / 10)
        return calls, calls * (450 + 14 * 300 + 10 * 40), len(run.claims) * 40
    if metric == "pairwise":
        return 2, 2 * (400 + 2 * brief_t), 2 * 150
    raise ValueError(metric)


def print_estimate(rows: dict[str, list[int]], model: str, n_runs: int) -> float:
    pin, pout = _price(model)
    total = 0.0
    print(f"judge {model}: ${pin:.2f}/M in, ${pout:.2f}/M out; {n_runs} run(s) to score (cache hits are free)")
    print(f"{'metric':<12} {'calls':>7} {'tok in':>10} {'tok out':>9} {'est USD':>8}")
    for m, (calls, tin, tout) in rows.items():
        usd = tin / 1e6 * pin + tout / 1e6 * pout
        total += usd
        print(f"{m:<12} {calls:>7} {tin:>10} {tout:>9} {usd:>8.3f}")
    print(f"{'total':<12} {'':>7} {'':>10} {'':>9} {total:>8.3f}")
    return total


# --- score --------------------------------------------------------------------------------------------------


async def score_run(
    run: RunRecord, question: Question | None, metrics: list[str], llm: LLM, *, judge_model: str,
    fetcher: PageFetcher | None,
) -> tuple[list[ScoreRecord], list[ClaimLabel]]:
    from lgeval.metrics import accuracy, calibration, citation, factual

    scores: list[ScoreRecord] = []
    labels: list[ClaimLabel] = []
    for m in metrics:
        if m == "factual":
            extra = []
            if fetcher is not None and not run.evidence:
                extra = await _fetched_evidence(run, fetcher)
            scores += await factual.score_factual(run, question, llm, judge_model=judge_model, extra_evidence=extra)
        elif m == "citation":
            scores += await citation.score_citations(run, llm, judge_model=judge_model, fetcher=fetcher)
        elif m == "accuracy" and question is not None:
            scores += await accuracy.score_accuracy(run, question, llm, judge_model=judge_model)
        elif m == "calibration" and run.claims:
            run_labels = await calibration.label_claims(run, question, llm, model=judge_model)
            labels += run_labels
            scores += calibration.calibration_scores(run_labels, run, judge_model=judge_model)
    return scores, labels


async def _fetched_evidence(run: RunRecord, fetcher: PageFetcher):
    from lgeval.schema import Evidence

    urls = list(dict.fromkeys(u for c in resolve_citations(run) for u in c.urls))[:20]
    out = []
    for i, u in enumerate(urls):
        text = await fetcher.fetch(u)
        for j in range(0, min(len(text), 12000), 1200):
            out.append(Evidence(chunk_id=f"url{i}-{j // 1200}", url=u, text=text[j : j + 1200]))
    return out


def _log_cost(d: Path, command: str, model: str, llm: Any) -> None:
    from lgeval.schema import JudgeCost

    s = llm.stats
    if s.calls:  # cache-only invocations spent nothing
        append_jsonl(d / "judge_cost.jsonl", [JudgeCost(
            command=command, judge_model=model, calls=s.calls, cache_hits=s.cache_hits,
            tokens_in=s.tokens_in, tokens_out=s.tokens_out, usd=s.usd,
        )])


async def cmd_score(args: argparse.Namespace) -> int:
    dirs = [Path(d) for d in args.results]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in ALL_METRICS]
    if bad:
        print(f"unknown metric(s): {', '.join(bad)} (choose from {', '.join(ALL_METRICS)})")
        return 2
    model = args.judge_model
    runs = [(d, r) for d, r in load_runs(dirs) if r.status in SCORABLE and r.brief.strip()]
    if args.systems:
        keep = {s.strip() for s in args.systems.split(",")}
        runs = [(d, r) for d, r in runs if r.system in keep]
    questions = load_question_map(dirs, [r for _, r in runs])

    todo: list[tuple[Path, RunRecord, list[str]]] = []
    for d in dirs:
        done = set() if args.force else {
            (r["question_id"], r["system"], r.get("repeat", 0), r["metric"], r.get("judge_model"))
            for r in read_jsonl(d / "scores.jsonl")
        }
        for rd, run in runs:
            if rd != d:
                continue
            need = [m for m in metrics
                    if (run.question_id, run.system, run.repeat, DONE_METRIC[m], model) not in done]
            if need:
                todo.append((d, run, need))

    est: dict[str, list[int]] = {m: [0, 0, 0] for m in metrics}
    for _, run, need in todo:
        for m in need:
            for i, v in enumerate(estimate(run, questions.get(run.question_id), m)):
                est[m][i] += v
    usd = print_estimate(est, model, len(todo))
    if args.dry_run:
        return 0
    if args.max_usd is not None and usd > args.max_usd:
        print(f"note: estimate ${usd:.2f} exceeds --max-usd ${args.max_usd:.2f}; scoring stops at the cap")

    budget = _budget(args.max_usd)
    llm = make_llm(model, budget=budget)
    fetcher = PageFetcher(default_root() / ".url_cache") if args.fetch_urls else None
    sem = asyncio.Semaphore(max(1, args.concurrency))
    lock = asyncio.Lock()
    stopped = False

    async def one(d: Path, run: RunRecord, need: list[str]) -> None:
        nonlocal stopped
        async with sem:
            if stopped or budget.exhausted():
                stopped = True
                return
            for m in need:  # written per metric, so a budget stop keeps finished metrics
                try:
                    scores, labels = await score_run(run, questions.get(run.question_id), [m], llm,
                                                     judge_model=model, fetcher=fetcher)
                except BudgetExhausted:
                    stopped = True
                    return
                except Exception as exc:  # one broken run must not kill the batch
                    log.warning("scoring %s %s/%s/%s failed: %s", m, run.question_id, run.system, run.repeat, exc)
                    continue
                async with lock:
                    append_jsonl(d / "scores.jsonl", scores)
                    append_jsonl(d / "claim_labels.jsonl", labels)

    try:
        await asyncio.gather(*(one(d, r, n) for d, r, n in todo))
    finally:
        await llm.aclose()
    s = llm.stats
    _log_cost(dirs[0], "score", model, llm)
    print(f"judge calls {s.calls} (cache hits {s.cache_hits}), tokens {s.tokens_in} in / {s.tokens_out} out, "
          f"${s.usd:.4f}" + ("; STOPPED: --max-usd reached" if stopped else ""))
    all_scores = [ScoreRecord.model_validate(r) for d in dirs for r in read_jsonl(d / "scores.jsonl")]
    _print_aggregate(aggregate([r for r in all_scores if r.judge_model == model]))
    return 0


def _print_aggregate(agg: dict[str, dict[str, dict[str, float]]]) -> None:
    for system, metrics in sorted(agg.items()):
        for metric, v in sorted(metrics.items()):
            print(f"  {system:<16} {metric:<24} {v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}] n={v['n']}")


# --- pairwise -----------------------------------------------------------------------------------------------


def pair_runs(runs: list[RunRecord], a: str, b: str) -> list[tuple[RunRecord, RunRecord]]:
    """Pair A and B briefs of the same question: repeats matched in order (zip), so a system with one run
    (an external baseline) is compared once per question."""
    by: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for r in runs:
        if r.status in SCORABLE and r.brief.strip() and r.system in (a, b):
            by[(r.system, r.question_id)].append(r)
    pairs = []
    for (system, qid), ra in sorted(by.items(), key=lambda kv: kv[0]):
        if system != a:
            continue
        rb = by.get((b, qid), [])
        pairs += list(zip(sorted(ra, key=lambda r: r.repeat), sorted(rb, key=lambda r: r.repeat), strict=False))
    return pairs


async def cmd_pairwise(args: argparse.Namespace) -> int:
    from lgeval.metrics.pairwise import compare, win_rate

    dirs = [Path(d) for d in args.results]
    out_dir = Path(args.out) if args.out else dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [r for _, r in load_runs(dirs)]
    questions = load_question_map(dirs, runs)
    model = args.judge_model
    done = set() if args.force else {
        (r["question_id"], r["system_a"], r["system_b"], r.get("repeat", 0), r["judge_model"])
        for r in read_jsonl(out_dir / "pairwise.jsonl")
    }
    pairs = [(ra, rb) for ra, rb in pair_runs(runs, args.a, args.b)
             if (ra.question_id, args.a, args.b, ra.repeat, model) not in done]
    est = [0, 0, 0]
    for ra, _ in pairs:
        for i, v in enumerate(estimate(ra, None, "pairwise")):
            est[i] += v
    print_estimate({"pairwise": est}, model, len(pairs))
    if args.dry_run:
        return 0
    budget = _budget(args.max_usd)
    llm = make_llm(model, budget=budget)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    lock = asyncio.Lock()
    stopped = False

    async def one(ra: RunRecord, rb: RunRecord) -> None:
        nonlocal stopped
        async with sem:
            if stopped or budget.exhausted():
                stopped = True
                return
            q = questions.get(ra.question_id)
            try:
                rec = await compare(q.question if q else "", ra, rb, llm, judge_model=model)
            except BudgetExhausted:
                stopped = True
                return
            except Exception as exc:
                log.warning("pairwise %s failed: %s", ra.question_id, exc)
                return
            async with lock:
                append_jsonl(out_dir / "pairwise.jsonl", [rec])

    try:
        await asyncio.gather(*(one(ra, rb) for ra, rb in pairs))
    finally:
        await llm.aclose()
    _log_cost(out_dir, "pairwise", model, llm)
    recs = [PairwiseRecord.model_validate(r) for r in read_jsonl(out_dir / "pairwise.jsonl")]
    recs = [r for r in recs if r.system_a == args.a and r.system_b == args.b and r.judge_model == model]
    w = win_rate(recs)
    print(f"judge calls {llm.stats.calls} (cache hits {llm.stats.cache_hits}), ${llm.stats.usd:.4f}"
          + ("; STOPPED: --max-usd reached" if stopped else ""))
    print(f"{args.a} vs {args.b}: n={w['n']} wins {w['a_wins']}-{w['b_wins']} ties {w['ties']}; "
          f"{args.a} win rate {w['a_rate']:.3f} [{w['ci_low']:.3f}, {w['ci_high']:.3f}], "
          f"position consistency {w['position_consistency']:.2f}")
    return 0


# --- registration -------------------------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("score", help="LLM-judge metrics for the briefs in results dirs (scores.jsonl)")
    p.add_argument("--results", nargs="+", required=True, help="results dir(s) containing runs.jsonl")
    p.add_argument("--metrics", default=",".join(ALL_METRICS), help=f"comma list of {','.join(ALL_METRICS)}")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--systems", default="", help="only score these systems (comma list)")
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--dry-run", action="store_true", help="print the estimated judge calls/tokens/USD and exit")
    p.add_argument("--fetch-urls", action="store_true",
                   help="fetch cited URL pages (external baselines) as evidence; cached, size-capped")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--force", action="store_true", help="re-score runs already in scores.jsonl")
    p.set_defaults(func=cmd_score)

    q = sub.add_parser("pairwise", help="blind position-swapped pairwise preference A vs B (pairwise.jsonl)")
    q.add_argument("--results", nargs="+", required=True)
    q.add_argument("--a", required=True, help="system A")
    q.add_argument("--b", required=True, help="system B")
    q.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    q.add_argument("--out", default="", help="dir for pairwise.jsonl (default: first results dir)")
    q.add_argument("--max-usd", type=float, default=None)
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--concurrency", type=int, default=4)
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=cmd_pairwise)


__all__ = ["register", "cmd_score", "cmd_pairwise", "estimate", "pair_runs"]
