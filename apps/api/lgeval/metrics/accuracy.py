"""Answer accuracy against a gold answer (FRAMES), graded correct / incorrect / not_attempted.

Follows the SimpleQA grader (Wei et al., "Measuring short-form factuality in large language models", 2024) and
the FRAMES autorater (Krishna et al., "Fact, Fetch, and Reason", 2024): an LLM decides whether the response
states the gold target, with a separate not-attempted class so abstention is not counted as a wrong answer.
"""

from __future__ import annotations

from typing import Any

from lgeval.metrics import prompts
from lgeval.metrics.llm import LLM, JudgeError, ask_json
from lgeval.schema import Question, RunRecord, ScoreRecord

GRADES = ("correct", "incorrect", "not_attempted")


def _grade(data: Any) -> dict:
    grade = str(data["grade"]).strip().lower().replace(" ", "_").replace("-", "_")
    grade = {"a": "correct", "b": "incorrect", "c": "not_attempted"}.get(grade, grade)
    if grade not in GRADES:
        raise ValueError(f"bad grade {grade!r}")
    answer, reason = str(data.get("extracted_answer", "")), str(data.get("reason", ""))
    return {"grade": grade, "extracted_answer": answer, "reason": reason}


async def grade_answer(question: str, gold: str, brief: str, llm: LLM) -> dict:
    user = prompts.ACCURACY_USER.format(question=question, gold=gold, brief=brief[:24000])
    return await ask_json(llm, prompts.ACCURACY_SYSTEM, user, _grade)


async def score_accuracy(
    run: RunRecord, question: Question, llm: LLM, *, judge_model: str | None = None
) -> list[ScoreRecord]:
    """`answer_accuracy` (1 correct / 0 otherwise, n=1) and `answer_attempted` (1 unless not_attempted).
    Returns [] when the question has no gold answer; n=0 (excluded from aggregates) on a judge error."""
    if not question.gold_answer:
        return []
    base = dict(question_id=run.question_id, system=run.system, repeat=run.repeat, judge_model=judge_model)
    try:
        g = await grade_answer(question.question, question.gold_answer, run.brief, llm)
    except JudgeError as exc:
        return [ScoreRecord(**base, metric="answer_accuracy", value=0.0, n=0, details={"error": str(exc)})]
    details = {**g, "gold": question.gold_answer}
    return [
        ScoreRecord(**base, metric="answer_accuracy", value=float(g["grade"] == "correct"), n=1, details=details),
        ScoreRecord(**base, metric="answer_attempted", value=float(g["grade"] != "not_attempted"), n=1),
    ]


def accuracy_summary(records: list[ScoreRecord]) -> dict[str, float]:
    """SimpleQA-style summary over `answer_accuracy` records: correct, not_attempted rates and
    correct_given_attempted plus their harmonic mean (SimpleQA's F-score)."""
    grades = [r.details.get("grade") for r in records if r.metric == "answer_accuracy" and r.n]
    n = len(grades)
    if not n:
        return {"n": 0}
    correct = grades.count("correct") / n
    attempted = 1 - grades.count("not_attempted") / n
    cga = (grades.count("correct") / (n * attempted)) if attempted else 0.0
    f = 2 * correct * cga / (correct + cga) if (correct + cga) else 0.0
    return {"n": n, "correct": correct, "not_attempted": 1 - attempted, "correct_given_attempted": cga, "f_score": f}
