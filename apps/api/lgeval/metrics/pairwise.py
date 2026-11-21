"""Blind pairwise preference between two systems' briefs, judged in both orders.

LLM-as-a-judge pairwise comparison (Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
NeurIPS 2023): position bias is controlled by judging both orders and only counting a win when they agree
(otherwise tie). System names are hidden and citation formats are normalized to "[ref]" (reference lists
dropped) so formatting does not reveal the system. Win rates use Wilson intervals; multi-system rankings use a
Bradley-Terry model (Bradley & Terry, 1952) fitted with Hunter's (2004) MM algorithm, ties as half wins.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from lgeval.metrics import prompts
from lgeval.metrics.llm import LLM, JudgeError, ask_json
from lgeval.metrics.stats import wilson
from lgeval.metrics.text import strip_markers
from lgeval.schema import PairwiseRecord, RunRecord

Verdict = Literal["a", "b", "tie"]
MAX_BRIEF_CHARS = 16000


def _winner(data: Any) -> dict:
    w = str(data["winner"]).strip().lower().strip("\"' .")
    w = {"response 1": "1", "response 2": "2", "a": "1", "b": "2", "draw": "tie", "equal": "tie"}.get(w, w)
    if w not in ("1", "2", "tie"):
        raise ValueError(f"bad winner {w!r}")
    return {"winner": w, "reason": str(data.get("reason", ""))}


async def judge_once(question: str, first: str, second: str, llm: LLM) -> dict:
    """{"winner": "1"|"2"|"tie", "reason"} for briefs shown in this order (already normalized)."""
    user = prompts.PAIRWISE_USER.format(question=question, brief_1=first, brief_2=second)
    return await ask_json(llm, prompts.PAIRWISE_SYSTEM, user, _winner)


def reconcile(first_order: Verdict, swapped_order: Verdict) -> Verdict:
    """Both orders (already mapped to a/b) agree -> that verdict, else tie."""
    return first_order if first_order == swapped_order else "tie"


async def compare(
    question: str, run_a: RunRecord, run_b: RunRecord, llm: LLM, *, judge_model: str
) -> PairwiseRecord:
    """Judge A-vs-B and B-vs-A. A judge error in one order counts as tie for that order (noted in rationale)."""
    a = strip_markers(run_a.brief)[:MAX_BRIEF_CHARS]
    b = strip_markers(run_b.brief)[:MAX_BRIEF_CHARS]
    notes = []
    try:
        r1 = await judge_once(question, a, b, llm)
        first: Verdict = {"1": "a", "2": "b", "tie": "tie"}[r1["winner"]]
        notes.append(f"A first: {r1['reason']}")
    except JudgeError as exc:
        first = "tie"
        notes.append(f"A first: error {exc}")
    try:
        r2 = await judge_once(question, b, a, llm)
        swapped: Verdict = {"1": "b", "2": "a", "tie": "tie"}[r2["winner"]]
        notes.append(f"B first: {r2['reason']}")
    except JudgeError as exc:
        swapped = "tie"
        notes.append(f"B first: error {exc}")
    return PairwiseRecord(
        question_id=run_a.question_id, system_a=run_a.system, system_b=run_b.system, repeat=run_a.repeat,
        winner=reconcile(first, swapped), first_order=first, swapped_order=swapped, judge_model=judge_model,
        rationale="\n".join(notes),
    )


def win_rate(records: list[PairwiseRecord]) -> dict[str, Any]:
    """Win rate of system A over B (ties = half a win) with a Wilson 95% CI, plus position-consistency.

    Returns {system_a, system_b, n, a_wins, b_wins, ties, a_rate, ci_low, ci_high, position_consistency}."""
    n = len(records)
    a_w = sum(1 for r in records if r.winner == "a")
    b_w = sum(1 for r in records if r.winner == "b")
    ties = n - a_w - b_w
    p, lo, hi = wilson(a_w + 0.5 * ties, n)
    consistent = sum(1 for r in records if r.first_order == r.swapped_order)
    return {
        "system_a": records[0].system_a if records else None, "system_b": records[0].system_b if records else None,
        "n": n, "a_wins": a_w, "b_wins": b_w, "ties": ties, "a_rate": p, "ci_low": lo, "ci_high": hi,
        "position_consistency": consistent / n if n else float("nan"),
    }


def bradley_terry(records: list[PairwiseRecord], *, iters: int = 200, tol: float = 1e-9) -> dict[str, float]:
    """Bradley-Terry strengths (normalized to sum to 1) over all systems in the records, ties as half wins.
    A small prior (one virtual tie against every other system) keeps undefeated systems finite."""
    systems = sorted({r.system_a for r in records} | {r.system_b for r in records})
    if not systems:
        return {}
    wins: dict[str, float] = defaultdict(float)
    games: dict[tuple[str, str], float] = defaultdict(float)
    for r in records:
        a, b = r.system_a, r.system_b
        games[(a, b)] += 1
        games[(b, a)] += 1
        if r.winner == "a":
            wins[a] += 1
        elif r.winner == "b":
            wins[b] += 1
        else:
            wins[a] += 0.5
            wins[b] += 0.5
    for i in systems:  # prior
        for j in systems:
            if i != j:
                games[(i, j)] += 1
                wins[i] += 0.5
    p = {s: 1.0 / len(systems) for s in systems}
    for _ in range(iters):
        new = {}
        for i in systems:
            denom = sum(games[(i, j)] / (p[i] + p[j]) for j in systems if j != i and games[(i, j)])
            new[i] = wins[i] / denom if denom else p[i]
        total = sum(new.values())
        new = {s: v / total for s, v in new.items()}
        if max(abs(new[s] - p[s]) for s in systems) < tol:
            p = new
            break
        p = new
    return p
