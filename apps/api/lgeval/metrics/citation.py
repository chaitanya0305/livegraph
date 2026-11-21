"""Citation recall and precision, ALCE-style.

ALCE (Gao et al., "Enabling Large Language Models to Generate Text with Citations", EMNLP 2023), building on
AIS/AutoAIS (Rashkin et al., 2023): a sentence has citation recall 1 iff the concatenation of its cited
passages entails it (uncited sentences score 0); a citation is precise unless it is irrelevant, i.e. it does not
entail the sentence alone AND the remaining citations still entail it without it. ALCE uses a TRUE-T5 NLI model;
here the NLI judgments come from an LLM judge, several sentences per call to bound cost.

LiveGraph cites claims `(C1)` whose support_chunk_ids resolve to `RunRecord.evidence`; external baselines cite
URLs, resolved from evidence with that url or (opt-in `--fetch-urls`) by fetching the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lgeval.metrics import prompts
from lgeval.metrics.llm import LLM, JudgeError, ask_json
from lgeval.metrics.text import (
    PageFetcher,
    best_window,
    evidence_lookup,
    find_evidence,
    resolve_citations,
    strip_markers,
)
from lgeval.schema import Citation, RunRecord, ScoreRecord

PASSAGE_CHARS = 1500


@dataclass
class SentenceCheck:
    sentence: str
    refs: list[str] = field(default_factory=list)  # chunk ids / urls cited
    passages: list[tuple[str, str]] = field(default_factory=list)  # (ref, text) that resolved
    unresolved: list[str] = field(default_factory=list)
    joint: bool | None = None
    citations: list[dict] = field(default_factory=list)  # [{ref, alone, without_it}]
    error: str | None = None

    @property
    def kind(self) -> str:
        if not self.refs:
            return "uncited"
        if not self.passages:
            return "unresolved"
        if self.error is not None or self.joint is None:
            return "error"
        return "judged"


async def gather_passages(
    cit: Citation, by_id: dict, by_url: dict, fetcher: PageFetcher | None, max_passages: int
) -> SentenceCheck:
    chk = SentenceCheck(sentence=cit.sentence, refs=list(dict.fromkeys(cit.chunk_ids + cit.urls)))
    for cid in dict.fromkeys(cit.chunk_ids):
        ev = find_evidence(cid, by_id)
        if ev is None:
            chk.unresolved.append(cid)
        elif len(chk.passages) < max_passages:
            chk.passages.append((cid, ((ev.title + ": ") if ev.title else "") + ev.text[:PASSAGE_CHARS]))
    for url in dict.fromkeys(cit.urls):
        text = " ".join(e.text for e in by_url.get(url.rstrip("/"), []))
        if not text and fetcher is not None:
            text = await fetcher.fetch(url)
        if not text:
            chk.unresolved.append(url)
        elif len(chk.passages) < max_passages:
            chk.passages.append((url, best_window(text, cit.sentence, PASSAGE_CHARS)))
    return chk


def _clean(sentence: str) -> str:
    return " ".join(strip_markers(sentence).replace("[ref]", "").split())


def _validator(batch: list[SentenceCheck]):
    def validate(data: Any) -> dict[int, dict]:
        rows = data["results"] if isinstance(data, dict) else data
        out: dict[int, dict] = {}
        for row in rows:
            sid = int(row["sid"])
            if not 1 <= sid <= len(batch):
                continue
            cits = {int(c["cid"]): c for c in row.get("citations") or []}
            out[sid] = {"joint": bool(row["joint"]), "citations": cits}
        if not out:
            raise ValueError("no usable results")
        return out

    return validate


async def judge_sentences(checks: list[SentenceCheck], llm: LLM, *, batch_size: int = 4) -> None:
    """Fill joint/citations on each check with passages (in place)."""
    todo = [c for c in checks if c.passages]
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        items = []
        for sid, c in enumerate(batch, 1):
            psg = "\n".join(f'<passage id="{pid}">\n{t}\n</passage>' for pid, (_, t) in enumerate(c.passages, 1))
            items.append(f'<statement sid="{sid}">\n{_clean(c.sentence)}\n</statement>\n{psg}')
        user = prompts.CITATION_USER.format(items="\n\n".join(items))
        try:
            res = await ask_json(llm, prompts.CITATION_SYSTEM, user, _validator(batch))
        except JudgeError as exc:
            for c in batch:
                c.error = str(exc)
            continue
        for sid, c in enumerate(batch, 1):
            row = res.get(sid)
            if row is None:
                c.error = "missing from judge reply"
                continue
            c.joint = row["joint"]
            single = len(c.passages) == 1
            c.citations = []
            for pid, (ref, _) in enumerate(c.passages, 1):
                cr = row["citations"].get(pid, {})
                alone = bool(cr.get("alone", single and c.joint))
                without = False if single else bool(cr.get("without_it", False))
                c.citations.append({"ref": ref, "alone": alone, "without_it": without})


def alce_scores(checks: list[SentenceCheck]) -> dict[str, Any]:
    """ALCE recall/precision over judged + uncited sentences (unresolved and errored ones are excluded).

    recall = sum(joint entailment) / (judged + uncited); precision = precise citations / all citations of judged
    sentences, where (as in ALCE's reference code) a sentence that fails joint entailment contributes 0 precise
    citations, a single citation is precise iff it entails, and in a multi-citation sentence a citation is
    imprecise only if it does not entail alone and the others entail without it.
    """
    judged = [c for c in checks if c.kind == "judged"]
    uncited = sum(1 for c in checks if c.kind == "uncited")
    recall_hits = sum(1 for c in judged if c.joint)
    total_cites = precise = 0
    for c in judged:
        total_cites += len(c.citations)
        if not c.joint:
            continue
        if len(c.citations) == 1:
            precise += 1
            continue
        for cr in c.citations:
            precise += 0 if (not cr["alone"] and cr["without_it"]) else 1
    n_recall = len(judged) + uncited
    return {
        "citation_recall": (recall_hits / n_recall if n_recall else 0.0, n_recall),
        "citation_precision": (precise / total_cites if total_cites else 0.0, total_cites),
        "uncited_sentence_rate": (uncited / len(checks) if checks else 0.0, len(checks)),
        "counts": {k: sum(1 for c in checks if c.kind == k) for k in ("judged", "uncited", "unresolved", "error")},
    }


async def score_citations(
    run: RunRecord,
    llm: LLM,
    *,
    judge_model: str | None = None,
    fetcher: PageFetcher | None = None,
    batch_size: int = 4,
    max_passages: int = 6,
    max_sentences: int = 60,
) -> list[ScoreRecord]:
    """ScoreRecords `citation_recall`, `citation_precision`, `uncited_sentence_rate` for one run.

    Headings, tables and the "Open questions"/references sections are ignored. Sentences whose citations
    cannot be resolved to any text are excluded from recall (counted in details.counts.unresolved).
    """
    by_id = evidence_lookup(run.evidence)
    by_url: dict[str, list] = {}
    for e in run.evidence:
        if e.url:
            by_url.setdefault(e.url.rstrip("/"), []).append(e)
    cits = resolve_citations(run)[:max_sentences]
    checks = [await gather_passages(c, by_id, by_url, fetcher, max_passages) for c in cits]
    await judge_sentences(checks, llm, batch_size=batch_size)
    s = alce_scores(checks)
    sentences = [
        {"sentence": c.sentence, "kind": c.kind, "joint": c.joint, "citations": c.citations,
         "unresolved": c.unresolved, **({"error": c.error} if c.error else {})}
        for c in checks
    ]
    base = dict(question_id=run.question_id, system=run.system, repeat=run.repeat, judge_model=judge_model)
    out = []
    for metric in ("citation_recall", "citation_precision", "uncited_sentence_rate"):
        value, n = s[metric]
        details: dict[str, Any] = {"counts": s["counts"]}
        if metric == "citation_recall":
            details["sentences"] = sentences
        out.append(ScoreRecord(**base, metric=metric, value=value, n=n, details=details))
    return out
