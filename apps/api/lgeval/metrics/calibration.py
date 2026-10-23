"""Claim B: does the judge's per-claim confidence predict whether a claim is true?

Brier score (Brier, 1950); expected calibration error with 10 equal-width bins and reliability diagrams
(Naeini et al., 2015; Guo et al., "On Calibration of Modern Neural Networks", 2017); AUROC of confidence for
true vs false claims (Mann-Whitney U formulation, Hanley & McNeil, 1982), as used for LLM confidence in
Kadavath et al. (2022) and Tian et al. (2023). Truth labels come from an LLM labeler (blind to confidence) or
from humans (`load_human_labels`); "unverifiable" labels are excluded from Brier/ECE/AUROC but counted.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from lgeval.metrics import prompts
from lgeval.metrics.llm import LLM, JudgeError, ask_json
from lgeval.metrics.text import evidence_lookup, find_evidence, rank_passages
from lgeval.schema import ClaimLabel, Question, RunRecord, ScoreRecord

LABELS = ("true", "false", "unverifiable")
PASSAGE_CHARS = 1200


# --- math ---------------------------------------------------------------------------------------------------


def brier(conf: list[float], y: list[int]) -> float:
    """Mean squared error of confidence vs outcome (0 best, 0.25 = always 0.5)."""
    return sum((c - t) ** 2 for c, t in zip(conf, y, strict=True)) / len(conf) if conf else float("nan")


def _bin(c: float, n_bins: int) -> int:
    return min(int(c * n_bins), n_bins - 1)  # [0,0.1), ..., [0.9,1.0]


def reliability(conf: list[float], y: list[int], n_bins: int = 10) -> list[dict[str, float]]:
    """Reliability-diagram data: one dict per equal-width bin {bin_low, bin_high, mean_conf, accuracy, count}
    (mean_conf/accuracy are NaN for empty bins)."""
    out = []
    for b in range(n_bins):
        idx = [i for i, c in enumerate(conf) if _bin(c, n_bins) == b]
        out.append({
            "bin_low": b / n_bins, "bin_high": (b + 1) / n_bins, "count": len(idx),
            "mean_conf": sum(conf[i] for i in idx) / len(idx) if idx else float("nan"),
            "accuracy": sum(y[i] for i in idx) / len(idx) if idx else float("nan"),
        })
    return out


def ece(conf: list[float], y: list[int], n_bins: int = 10) -> float:
    """Expected calibration error: sum over bins of |accuracy - mean confidence| weighted by bin share."""
    if not conf:
        return float("nan")
    n = len(conf)
    return sum(b["count"] / n * abs(b["accuracy"] - b["mean_conf"]) for b in reliability(conf, y, n_bins) if b["count"])


def auroc(conf: list[float], y: list[int]) -> float:
    """P(conf of a random true claim > conf of a random false claim), ties count 1/2. NaN if a class is empty."""
    pos = [c for c, t in zip(conf, y, strict=True) if t]
    neg = [c for c, t in zip(conf, y, strict=True) if not t]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def calibration_report(labels: list[ClaimLabel], n_bins: int = 10) -> dict[str, Any]:
    """Pooled calibration over ClaimLabels (call per system).

    Returns {n_claims, n_labeled, n_unverifiable, n_no_confidence, base_rate, brier, ece, auroc, reliability,
    struck: {n_struck, n_kept, struck_false_rate, kept_false_rate}}. Labeled = true/false with a confidence."""
    usable = [lb for lb in labels if lb.label != "unverifiable" and lb.confidence is not None]
    conf = [min(1.0, max(0.0, float(lb.confidence))) for lb in usable]
    y = [1 if lb.label == "true" else 0 for lb in usable]
    decided = [lb for lb in labels if lb.label != "unverifiable"]
    struck = [lb for lb in decided if lb.struck]
    kept = [lb for lb in decided if not lb.struck]

    def false_rate(xs: list[ClaimLabel]) -> float:
        return sum(1 for lb in xs if lb.label == "false") / len(xs) if xs else float("nan")

    return {
        "n_claims": len(labels),
        "n_labeled": len(usable),
        "n_unverifiable": sum(1 for lb in labels if lb.label == "unverifiable"),
        "n_no_confidence": sum(1 for lb in labels if lb.label != "unverifiable" and lb.confidence is None),
        "base_rate": sum(y) / len(y) if y else float("nan"),
        "brier": brier(conf, y),
        "ece": ece(conf, y, n_bins),
        "auroc": auroc(conf, y),
        "reliability": reliability(conf, y, n_bins),
        "struck": {"n_struck": len(struck), "n_kept": len(kept), "struck_false_rate": false_rate(struck),
                   "kept_false_rate": false_rate(kept)},
    }


def calibration_scores(
    labels: list[ClaimLabel], run: RunRecord, *, judge_model: str | None = None
) -> list[ScoreRecord]:
    """Per-run ScoreRecords: `brier` (n = labeled claims; n=0 when none) and `claim_true_rate`.
    ECE/AUROC need pooled claims: use `calibration_report` over claim_labels.jsonl per system."""
    rep = calibration_report(labels)
    base = dict(question_id=run.question_id, system=run.system, repeat=run.repeat, judge_model=judge_model)
    n = rep["n_labeled"]
    details = {k: rep[k] for k in ("n_claims", "n_unverifiable", "n_no_confidence", "struck")}
    decided = [lb for lb in labels if lb.label != "unverifiable"]
    true_rate = sum(1 for lb in decided if lb.label == "true") / len(decided) if decided else 0.0
    return [
        ScoreRecord(**base, metric="brier", value=rep["brier"] if n else 0.0, n=n, details=details),
        ScoreRecord(**base, metric="claim_true_rate", value=true_rate, n=len(decided)),
    ]


# --- labeling -----------------------------------------------------------------------------------------------


def _validator(keys: list[str]):
    def validate(data: Any) -> dict[str, dict]:
        rows = data["results"] if isinstance(data, dict) else data
        out = {}
        for row in rows:
            label = str(row.get("label", "")).strip().lower()
            label = {"supported": "true", "refuted": "false", "unknown": "unverifiable"}.get(label, label)
            key = str(row.get("key", "")).strip()
            if label in LABELS and key in keys:
                out[key] = {"label": label, "reason": str(row.get("reason", ""))}
        if not out:
            raise ValueError("no usable labels")
        return out

    return validate


async def label_claims(
    run: RunRecord,
    question: Question | None,
    llm: LLM,
    *,
    model: str = "judge",
    batch_size: int = 10,
    max_passages: int = 14,
) -> list[ClaimLabel]:
    """LLM truth labels (labeler="llm:<model>") for every claim of a LiveGraph run, blind to confidence.

    Evidence per batch = the claims' own support chunks first, then BM25-ranked run evidence; the dataset's gold
    answer/facts are given as a reference. Claims the judge fails on are omitted (not guessed)."""
    q = question or Question(id=run.question_id, question="", dataset=run.dataset)
    by_id = evidence_lookup(run.evidence)
    docs = [(e.chunk_id[:8], (f"{e.title}: " if e.title else "") + e.text[:PASSAGE_CHARS]) for e in run.evidence]
    reference = prompts.reference_block(q.gold_answer, q.gold_facts)
    out: list[ClaimLabel] = []
    claims = [c for c in run.claims if c.text.strip()]
    for start in range(0, len(claims), batch_size):
        batch = claims[start : start + batch_size]
        chosen: dict[str, str] = {}
        for c in batch:
            for cid in c.support_chunk_ids:
                ev = find_evidence(cid, by_id)
                if ev is not None and len(chosen) < max_passages:
                    chosen.setdefault(ev.chunk_id[:8], (f"{ev.title}: " if ev.title else "") + ev.text[:PASSAGE_CHARS])
        for c in batch:
            for pid, text in rank_passages(c.text, docs, 3):
                if len(chosen) < max_passages:
                    chosen.setdefault(pid, text)
        user = prompts.LABEL_USER.format(
            question=q.question, reference=reference, passages=prompts.passages_block(list(chosen.items())),
            claims="\n".join(f"{c.key}: {c.text}" for c in batch),
        )
        try:
            res = await ask_json(llm, prompts.LABEL_SYSTEM, user, _validator([c.key for c in batch]))
        except JudgeError:
            continue
        for c in batch:
            if c.key in res:
                out.append(ClaimLabel(
                    question_id=run.question_id, system=run.system, repeat=run.repeat, claim_key=c.key,
                    claim_text=c.text, confidence=c.confidence, struck=c.status == "struck",
                    label=res[c.key]["label"], labeler=f"llm:{model}", rationale=res[c.key]["reason"],
                ))
    return out


HUMAN_CSV_COLUMNS = ("question_id", "system", "repeat", "claim_key", "claim_text", "confidence", "struck", "label",
                     "labeler", "rationale")


def load_human_labels(path: Path | str, *, default_labeler: str = "human:anon") -> list[ClaimLabel]:
    """Human labels from a CSV with a header row.

    Required columns: question_id, system, claim_key, label (true|false|unverifiable; also 1/0/t/f/yes/no/?).
    Optional: repeat (default 0), claim_text, confidence (0..1, blank = none), struck (true/1 = struck),
    labeler (default `default_labeler`, should look like "human:<name>"), rationale. Rows with an unknown
    label are skipped."""
    truthy = {"true", "1", "t", "yes", "y"}
    alias = {"1": "true", "t": "true", "yes": "true", "y": "true", "0": "false", "f": "false", "no": "false",
             "n": "false", "?": "unverifiable", "unknown": "unverifiable", "": "unverifiable"}
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            label = row.get("label", "").lower()
            label = alias.get(label, label)
            if label not in LABELS or not row.get("question_id") or not row.get("claim_key"):
                continue
            conf = row.get("confidence", "")
            out.append(ClaimLabel(
                question_id=row["question_id"], system=row.get("system", ""), repeat=int(row.get("repeat") or 0),
                claim_key=row["claim_key"], claim_text=row.get("claim_text", ""),
                confidence=float(conf) if conf else None, struck=row.get("struck", "").lower() in truthy,
                label=label, labeler=row.get("labeler") or default_labeler, rationale=row.get("rationale", ""),
            ))
    return out
