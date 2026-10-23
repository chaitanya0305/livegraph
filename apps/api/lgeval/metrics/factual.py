"""Factual precision, FActScore/SAFE-style.

FActScore (Min et al., EMNLP 2023): split a generation into atomic facts and report the fraction supported by
a knowledge source. SAFE (Wei et al., "Long-form factuality in large language models", 2024): self-contained
facts, an irrelevance filter, precision = supported / (supported + not supported). Here the knowledge source is
the run's own evidence (chunks it retrieved; fetched cited pages for URL-citing baselines) plus the dataset's
gold answer/facts when present, so the score measures groundedness in what the system actually read.
Facts are verified in batches against BM25-selected passages to bound judge calls and prompt size.
"""

from __future__ import annotations

from typing import Any

from lgeval.metrics import prompts
from lgeval.metrics.llm import LLM, JudgeError, ask_json
from lgeval.metrics.text import rank_passages
from lgeval.schema import Evidence, Question, RunRecord, ScoreRecord

LABELS = ("supported", "unsupported", "irrelevant")
PASSAGE_CHARS = 1200


def _facts(data: Any) -> list[str]:
    facts = data["facts"] if isinstance(data, dict) else data
    if not isinstance(facts, list):
        raise TypeError("facts must be a list")
    return [str(f).strip() for f in facts if str(f).strip()]


async def extract_facts(brief: str, question: str, llm: LLM, *, max_facts: int = 40) -> list[str]:
    user = prompts.ATOMIC_USER.format(question=question, brief=brief[:24000], max_facts=max_facts)
    facts = await ask_json(llm, prompts.ATOMIC_SYSTEM, user, _facts)
    return list(dict.fromkeys(facts))[:max_facts]


def _results_validator(n: int, key: str, allowed: tuple[str, ...]):
    def validate(data: Any) -> dict[int, dict]:
        rows = data["results"] if isinstance(data, dict) else data
        out: dict[int, dict] = {}
        for row in rows:
            label = str(row.get("label", "")).strip().lower().replace(" ", "_")
            label = {"not_supported": "unsupported"}.get(label, label)
            if label not in allowed:
                continue
            out[int(row[key])] = {**row, "label": label}
        if not out or not set(range(1, n + 1)) & set(out):
            raise ValueError("no usable results")
        return out

    return validate


async def verify_facts(
    facts: list[str],
    question: Question | str,
    evidence: list[Evidence],
    llm: LLM,
    *,
    batch_size: int = 8,
    passages_per_fact: int = 4,
    max_passages: int = 12,
    use_gold: bool = True,
) -> list[dict]:
    """[{fact, label (supported|unsupported|irrelevant|error), evidence, reason}] in input order."""
    qtext = question if isinstance(question, str) else question.question
    gold_answer = None if isinstance(question, str) or not use_gold else question.gold_answer
    gold_facts = [] if isinstance(question, str) or not use_gold else question.gold_facts
    reference = prompts.reference_block(gold_answer, gold_facts)
    docs = [(f"E{i + 1}", (f"{e.title}: " if e.title else "") + e.text[:PASSAGE_CHARS]) for i, e in enumerate(evidence)]
    out: list[dict] = []
    for start in range(0, len(facts), batch_size):
        batch = facts[start : start + batch_size]
        chosen: dict[str, str] = {}
        for f in batch:
            for pid, text in rank_passages(f, docs, passages_per_fact):
                if len(chosen) < max_passages:
                    chosen.setdefault(pid, text)
        user = prompts.VERIFY_USER.format(
            question=qtext,
            reference=reference,
            passages=prompts.passages_block(list(chosen.items())),
            facts="\n".join(f"{i + 1}. {f}" for i, f in enumerate(batch)),
        )
        try:
            res = await ask_json(llm, prompts.VERIFY_SYSTEM, user, _results_validator(len(batch), "id", LABELS))
        except JudgeError as exc:
            res, err = {}, str(exc)
        else:
            err = "missing from judge reply"
        for i, f in enumerate(batch):
            row = res.get(i + 1)
            if row is None:
                out.append({"fact": f, "label": "error", "evidence": "", "reason": err})
            else:
                out.append({"fact": f, "label": row["label"], "evidence": str(row.get("evidence", "")),
                            "reason": str(row.get("reason", ""))})
    return out


def precision_of(results: list[dict]) -> tuple[float, int, dict[str, int]]:
    """(supported / (supported + unsupported), that denominator, counts per label)."""
    counts = {k: sum(1 for r in results if r["label"] == k) for k in (*LABELS, "error")}
    n = counts["supported"] + counts["unsupported"]
    return (counts["supported"] / n if n else 0.0), n, counts


async def score_factual(
    run: RunRecord,
    question: Question | None,
    llm: LLM,
    *,
    judge_model: str | None = None,
    extra_evidence: list[Evidence] | None = None,
    max_facts: int = 40,
    batch_size: int = 8,
) -> list[ScoreRecord]:
    """`factual_precision` (n = supported + unsupported facts; n == 0 means undefined) and
    `factual_supported` (count of supported facts, so a terse brief cannot win on precision alone)."""
    q = question or Question(id=run.question_id, question="", dataset=run.dataset)
    base = dict(question_id=run.question_id, system=run.system, repeat=run.repeat, judge_model=judge_model)
    try:
        facts = await extract_facts(run.brief, q.question, llm, max_facts=max_facts)
    except JudgeError as exc:
        details = {"error": f"extract: {exc}"}
        return [ScoreRecord(**base, metric="factual_precision", value=0.0, n=0, details=details)]
    evidence = list(run.evidence) + list(extra_evidence or [])
    results = await verify_facts(facts, q, evidence, llm, batch_size=batch_size)
    value, n, counts = precision_of(results)
    details = {"counts": counts, "n_facts": len(facts), "n_evidence": len(evidence), "facts": results}
    return [
        ScoreRecord(**base, metric="factual_precision", value=value, n=n, details=details),
        ScoreRecord(**base, metric="factual_supported", value=float(counts["supported"]), n=len(facts),
                    details={"counts": counts}),
    ]
