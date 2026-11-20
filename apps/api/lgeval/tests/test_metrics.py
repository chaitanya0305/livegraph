"""Offline tests for lgeval.metrics: parsing, scoring math, citation matching and pairwise reconciliation."""

from __future__ import annotations

import argparse
import json
import math

import httpx
import pytest

from lgeval.metrics import accuracy, calibration, citation, commands, factual, pairwise, prompts, stats, text
from lgeval.metrics.llm import BudgetExhausted, JudgeError, ask_json, make_llm, parse_json
from lgeval.schema import Citation, ClaimLabel, ClaimOut, Evidence, PairwiseRecord, Question, RunRecord, ScoreRecord

BRIEF = """# Battery outlook

Solid-state batteries could reach cars by 2028 (C1). Their energy density is about 400 Wh/kg (C2).
Costs remain uncertain.

## Details
- Toyota plans a pilot line in Japan (C1, C2).

## Open questions
- Will lithium supply keep up with demand (C3)?
"""

EVIDENCE = [
    Evidence(chunk_id="aaaaaaaa-1111", title="Auto news",
             text="Toyota said solid-state batteries could reach cars by 2028."),
    Evidence(chunk_id="bbbbbbbb-2222", title="Lab report",
             text="Prototype cells reached about 400 Wh/kg energy density."),
    Evidence(chunk_id="cccccccc-3333", title="Misc", text="Lithium prices fell in 2024."),
]


def make_run(**kw) -> RunRecord:
    base = dict(
        question_id="q1", dataset="custom", system="full", status="done", brief=BRIEF, evidence=EVIDENCE,
        claims=[
            ClaimOut(key="C1", text="Solid-state batteries reach cars by 2028", confidence=0.9,
                     support_chunk_ids=["aaaaaaaa-1111"]),
            ClaimOut(key="C2", text="Energy density is about 400 Wh/kg", confidence=0.7,
                     support_chunk_ids=["bbbbbbbb"]),  # short prefix id
            ClaimOut(key="C3", text="Lithium supply is ample", confidence=0.2, status="struck",
                     support_chunk_ids=["cccccccc-3333"]),
        ],
    )
    base.update(kw)
    return RunRecord(**base)


def fake_llm(handlers: dict[str, object], calls: list | None = None):
    """Dispatch on the system prompt; a handler is a str reply or fn(user) -> str."""

    async def llm(system: str, user: str) -> str:
        if calls is not None:
            calls.append((system, user))
        h = handlers[system]
        return h(user) if callable(h) else h

    return llm


# --- parsing ------------------------------------------------------------------------------------------------


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json('Sure! Here it is:\n{"a": [1, 2]}\nHope that helps.') == {"a": [1, 2]}
    assert parse_json("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        parse_json("no json here")


async def test_ask_json_retries_once_then_errors():
    replies = iter(["garbage", '{"ok": true}'])
    calls: list = []

    async def llm(system, user):
        calls.append(user)
        return next(replies)

    assert await ask_json(llm, "s", "u") == {"ok": True}
    assert calls[1].endswith(prompts.JSON_RETRY_SUFFIX)  # retry prompt differs -> not served from cache

    async def bad(system, user):
        return "still garbage"

    with pytest.raises(JudgeError):
        await ask_json(bad, "s", "u")


# --- text ---------------------------------------------------------------------------------------------------


def test_brief_sentences_skip_headings_and_open_questions():
    sents = text.brief_sentences(BRIEF)
    assert sents == [
        "Solid-state batteries could reach cars by 2028 (C1).",
        "Their energy density is about 400 Wh/kg (C2).",
        "Costs remain uncertain.",
        "Toyota plans a pilot line in Japan (C1, C2).",
    ]


def test_resolve_citations_maps_claims_to_chunks():
    cits = text.resolve_citations(make_run())
    assert [c.claim_keys for c in cits] == [["C1"], ["C2"], [], ["C1", "C2"]]
    assert cits[0].chunk_ids == ["aaaaaaaa-1111"]
    assert cits[3].chunk_ids == ["aaaaaaaa-1111", "bbbbbbbb"]
    assert cits[2].chunk_ids == [] and cits[2].urls == []


def test_resolve_citations_prefers_runner_citations_and_urls():
    run = make_run(
        brief="Paris is the capital of France [1]. See https://example.org/x for more on Paris.",
        claims=[],
        citations=[Citation(sentence="Paris is the capital of France [1].", urls=["https://wiki.example/paris"])],
    )
    cits = text.resolve_citations(run)
    assert cits[0].urls == ["https://wiki.example/paris"]
    assert cits[1].urls == ["https://example.org/x"]


def test_strip_markers_normalizes_citation_styles():
    a = text.strip_markers("X is true (C1, C2). Y [2][3]. Z [link](https://a.b/c).\n\n## References\n1. https://a.b")
    assert "(C1" not in a and "https" not in a and "References" not in a
    assert a.count("[ref]") == 3


def test_rank_passages_prefers_overlap():
    docs = [("E1", "cats and dogs"), ("E2", "solid state battery density"), ("E3", "weather today")]
    assert text.rank_passages("battery density", docs, 1) == [("E2", "solid state battery density")]


def test_find_evidence_prefix():
    by_id = text.evidence_lookup(EVIDENCE)
    assert text.find_evidence("bbbbbbbb", by_id).chunk_id == "bbbbbbbb-2222"
    assert text.find_evidence("zzzzzzzz", by_id) is None


async def test_page_fetcher_caps_and_caches(tmp_path):
    hits = []

    def handler(request):
        hits.append(request.url)
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<html><script>x()</script><p>Hello &amp; world</p></html>" + "a" * 1000)

    f = text.PageFetcher(tmp_path, max_bytes=200, transport=httpx.MockTransport(handler))
    t = await f.fetch("https://e.org/p")
    assert t.startswith("Hello & world") and len(t) < 200
    f2 = text.PageFetcher(tmp_path, transport=httpx.MockTransport(handler))
    assert await f2.fetch("https://e.org/p") == t
    assert len(hits) == 1


# --- citation (ALCE) ----------------------------------------------------------------------------------------


def chk(joint, cites, refs=None):
    c = citation.SentenceCheck(sentence="s", refs=refs or ["x"] * max(1, len(cites)),
                               passages=[("x", "t")] * max(1, len(cites)))
    c.joint = joint
    c.citations = [{"ref": "x", "alone": a, "without_it": w} for a, w in cites]
    return c


def test_alce_scores_hand_computed():
    checks = [
        chk(True, [(True, False)]),  # single, entailed: 1 precise
        chk(True, [(True, True), (False, True)]),  # 2nd is irrelevant (not alone, others suffice)
        chk(True, [(False, False), (False, False)]),  # both needed jointly: both precise
        chk(False, [(False, False), (False, False)]),  # fails recall: 0 precise of 2
        citation.SentenceCheck(sentence="uncited"),
        citation.SentenceCheck(sentence="unresolved", refs=["gone"], unresolved=["gone"]),
    ]
    s = citation.alce_scores(checks)
    assert s["citation_recall"] == (3 / 5, 5)  # 3 entailed of 4 judged + 1 uncited
    assert s["citation_precision"] == (4 / 7, 7)  # 1 + 1 + 2 + 0 of 7 citations
    assert s["uncited_sentence_rate"] == (1 / 6, 6)
    assert s["counts"] == {"judged": 4, "uncited": 1, "unresolved": 1, "error": 0}


async def test_score_citations_end_to_end():
    def judge(user: str) -> str:
        n = user.count("<statement")
        results = []
        for sid in range(1, n + 1):
            block = user.split(f'<statement sid="{sid}">')[1].split("<statement")[0]
            k = block.count("<passage")
            results.append({"sid": sid, "joint": "Toyota plans" not in block,
                            "citations": [{"cid": i, "alone": i == 1, "without_it": False} for i in range(1, k + 1)]})
        return "```json\n" + json.dumps({"results": results}) + "\n```"

    calls: list = []
    recs = await citation.score_citations(make_run(), fake_llm({prompts.CITATION_SYSTEM: judge}, calls),
                                          judge_model="fake")
    by = {r.metric: r for r in recs}
    # 4 sentences: 3 cited (2 entailed), 1 uncited -> recall 2/4
    assert by["citation_recall"].value == pytest.approx(0.5) and by["citation_recall"].n == 4
    # citations: s1 1 precise, s2 1 precise, s4 joint fails -> 0 of 2 => 2/4
    assert by["citation_precision"].value == pytest.approx(0.5) and by["citation_precision"].n == 4
    assert by["uncited_sentence_rate"].value == pytest.approx(0.25)
    assert len(calls) == 1  # batched
    assert "(C1)" not in calls[0][1]


async def test_score_citations_judge_error_excluded():
    recs = await citation.score_citations(make_run(), fake_llm({prompts.CITATION_SYSTEM: "nope"}))
    rec = next(r for r in recs if r.metric == "citation_recall")
    assert rec.details["counts"]["error"] == 3 and rec.n == 1 and rec.value == 0.0


# --- factual ------------------------------------------------------------------------------------------------


async def test_factual_precision_batches_and_counts():
    facts = [f"fact {i}" for i in range(10)]

    def verify(user: str) -> str:
        n = sum(1 for line in user.splitlines() if line[:1].isdigit() and ". fact" in line)
        labels = ["supported", "unsupported", "irrelevant", "supported"]
        res = [{"id": i, "label": labels[(i - 1) % 4], "reason": ""} for i in range(1, n + 1)]
        return json.dumps({"results": res})

    calls: list = []
    llm = fake_llm({prompts.ATOMIC_SYSTEM: json.dumps({"facts": facts + ["fact 0"]}),
                    prompts.VERIFY_SYSTEM: verify}, calls)
    q = Question(id="q1", question="When will solid-state batteries ship?", dataset="custom", gold_answer="2028")
    recs = await factual.score_factual(make_run(), q, llm, judge_model="fake", batch_size=8)
    fp = recs[0]
    # batch1 (8): S U I S S U I S -> 4S 2U ; batch2 (2): S U -> 5S 3U
    assert fp.metric == "factual_precision" and fp.n == 8 and fp.value == pytest.approx(5 / 8)
    assert fp.details["counts"]["irrelevant"] == 2 and fp.details["n_facts"] == 10  # deduped
    assert recs[1].metric == "factual_supported" and recs[1].value == 5
    assert len(calls) == 3  # 1 extract + 2 verify batches
    assert "Gold answer: 2028" in calls[1][1]


async def test_factual_missing_items_marked_error():
    llm = fake_llm({prompts.ATOMIC_SYSTEM: '{"facts": ["a", "b"]}',
                    prompts.VERIFY_SYSTEM: '{"results": [{"id": 1, "label": "Not supported"}]}'})
    res = await factual.verify_facts(["a", "b"], "q", EVIDENCE, llm)
    assert [r["label"] for r in res] == ["unsupported", "error"]


async def test_factual_extract_failure_gives_undefined_record():
    llm = fake_llm({prompts.ATOMIC_SYSTEM: "nope"})
    recs = await factual.score_factual(make_run(), None, llm)
    assert len(recs) == 1 and recs[0].n == 0 and "error" in recs[0].details


# --- accuracy -----------------------------------------------------------------------------------------------


async def test_accuracy_grades():
    q = Question(id="q1", question="When?", dataset="frames", gold_answer="2028")
    for reply, val, att in [('{"grade": "CORRECT"}', 1.0, 1.0), ('{"grade": "not attempted"}', 0.0, 0.0),
                            ('{"grade": "incorrect"}', 0.0, 1.0)]:
        recs = await accuracy.score_accuracy(make_run(), q, fake_llm({prompts.ACCURACY_SYSTEM: reply}))
        assert recs[0].value == val and recs[1].value == att
    assert await accuracy.score_accuracy(make_run(), Question(id="q", question="x", dataset="c"),
                                         fake_llm({})) == []
    summ = accuracy.accuracy_summary([
        ScoreRecord(question_id=str(i), system="s", metric="answer_accuracy", value=0, n=1, details={"grade": g})
        for i, g in enumerate(["correct", "correct", "incorrect", "not_attempted"])
    ])
    assert summ["correct"] == 0.5 and summ["not_attempted"] == 0.25
    assert summ["correct_given_attempted"] == pytest.approx(2 / 3)


# --- pairwise -----------------------------------------------------------------------------------------------


def test_reconcile():
    assert pairwise.reconcile("a", "a") == "a"
    assert pairwise.reconcile("b", "b") == "b"
    assert pairwise.reconcile("a", "b") == "tie"
    assert pairwise.reconcile("a", "tie") == "tie"


async def test_compare_position_bias_becomes_tie_and_content_wins():
    ra = make_run(system="full", brief="GOOD brief (C1).")
    rb = make_run(system="baseline", brief="weak brief [1].")
    always_first = fake_llm({prompts.PAIRWISE_SYSTEM: '{"winner": "1", "reason": "first"}'})
    rec = await pairwise.compare("q", ra, rb, always_first, judge_model="fake")
    assert (rec.first_order, rec.swapped_order, rec.winner) == ("a", "b", "tie")

    seen: list = []

    def by_content(user: str) -> str:
        seen.append(user)
        r1 = user.split('<response id="1">')[1].split("</response>")[0]
        return json.dumps({"winner": "1" if "GOOD" in r1 else "2", "reason": "content"})

    rec = await pairwise.compare("q", ra, rb, fake_llm({prompts.PAIRWISE_SYSTEM: by_content}), judge_model="fake")
    assert (rec.first_order, rec.swapped_order, rec.winner) == ("a", "a", "a")
    assert all("(C1)" not in u and "[1]" not in u and "full" not in u for u in seen)


def test_win_rate_and_bradley_terry():
    def rec(w, f=None, s=None):
        return PairwiseRecord(question_id="q", system_a="A", system_b="B", winner=w, first_order=f or w,
                              swapped_order=s or w, judge_model="j")

    w = pairwise.win_rate([rec("a")] * 4 + [rec("b")] * 4 + [rec("tie", "a", "b")] * 2)
    assert (w["a_wins"], w["b_wins"], w["ties"], w["n"]) == (4, 4, 2, 10)
    assert w["a_rate"] == 0.5 and w["ci_low"] == pytest.approx(0.2366, abs=1e-3)
    assert w["position_consistency"] == pytest.approx(0.8)
    recs = [rec("a")] * 8 + [rec("b")] * 2
    recs += [PairwiseRecord(question_id="q", system_a="B", system_b="C", winner="a", first_order="a",
                            swapped_order="a", judge_model="j")] * 8
    bt = pairwise.bradley_terry(recs)
    assert bt["A"] > bt["B"] > bt["C"] and sum(bt.values()) == pytest.approx(1.0)


# --- calibration --------------------------------------------------------------------------------------------


def test_brier_ece_auroc_hand_computed():
    conf, y = [0.9, 0.8, 0.3, 0.6], [1, 1, 0, 0]
    assert calibration.brier(conf, y) == pytest.approx(0.125)
    assert calibration.ece(conf, y) == pytest.approx(0.3)
    assert calibration.auroc(conf, y) == 1.0
    assert calibration.auroc([0.5, 0.5], [1, 0]) == 0.5
    assert calibration.auroc([0.7, 0.4, 0.7], [1, 1, 0]) == pytest.approx(0.25)
    assert math.isnan(calibration.auroc([0.3], [1]))
    # two claims in one bin: mean conf 0.75, accuracy 0.5 -> ECE 0.25; 1.0 falls in the last bin
    assert calibration.ece([0.72, 0.78], [1, 0]) == pytest.approx(0.25)
    rel = calibration.reliability([1.0, 0.05], [1, 0])
    assert rel[9]["count"] == 1 and rel[0]["count"] == 1 and math.isnan(rel[5]["mean_conf"])


def lab(label, conf, struck=False, key="C1"):
    return ClaimLabel(question_id="q", system="full", claim_key=key, claim_text="t", confidence=conf,
                      struck=struck, label=label, labeler="llm:x")


def test_calibration_report_excludes_unverifiable_and_struck_rates():
    labels = [lab("true", 0.9), lab("false", 0.2, struck=True), lab("false", 0.4, struck=True),
              lab("true", 0.3, struck=True), lab("unverifiable", 0.5), lab("false", 0.6), lab("true", None)]
    r = calibration.calibration_report(labels)
    assert r["n_labeled"] == 5 and r["n_unverifiable"] == 1 and r["n_no_confidence"] == 1
    assert r["struck"]["struck_false_rate"] == pytest.approx(2 / 3)
    assert r["struck"]["kept_false_rate"] == pytest.approx(1 / 3)
    conf, y = [0.9, 0.2, 0.4, 0.3, 0.6], [1, 0, 0, 1, 0]
    assert r["brier"] == pytest.approx(calibration.brier(conf, y))
    recs = calibration.calibration_scores(labels, make_run())
    assert recs[0].metric == "brier" and recs[0].n == 5


async def test_label_claims_blind_and_batched():
    calls: list = []
    reply = json.dumps({"results": [{"key": "C1", "label": "true"}, {"key": "C2", "label": "Refuted"},
                                    {"key": "C3", "label": "unverifiable"}, {"key": "C9", "label": "true"}]})
    q = Question(id="q1", question="q", dataset="custom", gold_facts=["Cars by 2028"])
    out = await calibration.label_claims(make_run(), q, fake_llm({prompts.LABEL_SYSTEM: reply}, calls), model="m")
    assert [(lb.claim_key, lb.label, lb.struck) for lb in out] == [("C1", "true", False), ("C2", "false", False),
                                                                      ("C3", "unverifiable", True)]
    assert out[0].labeler == "llm:m" and out[0].confidence == 0.9
    assert len(calls) == 1 and "0.9" not in calls[0][1] and "Cars by 2028" in calls[0][1]
    assert "400 Wh/kg energy density" in calls[0][1]  # support chunk resolved by prefix


def test_load_human_labels(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("question_id,system,claim_key,label,confidence,struck,labeler\n"
                 "q1,full,C1,TRUE,0.8,,human:ann\n"
                 "q1,full,C2,0,,yes,\n"
                 "q1,full,C3,maybe,0.5,,\n"
                 "q1,full,C4,?,0.5,,\n")
    out = calibration.load_human_labels(p, default_labeler="human:bob")
    assert [(lb.claim_key, lb.label, lb.confidence, lb.struck, lb.labeler) for lb in out] == [
        ("C1", "true", 0.8, False, "human:ann"), ("C2", "false", None, True, "human:bob"),
        ("C4", "unverifiable", 0.5, False, "human:bob")]


# --- stats --------------------------------------------------------------------------------------------------


def test_bootstrap_deterministic_and_edge_cases():
    xs = [0.1, 0.5, 0.9, 0.4, 0.7, 0.3]
    a, b = stats.bootstrap_ci(xs, seed=3), stats.bootstrap_ci(xs, seed=3)
    assert a == b and a[0] == pytest.approx(sum(xs) / 6) and a[1] < a[0] < a[2]
    assert stats.bootstrap_ci([0.4]) == (0.4, 0.4, 0.4)
    assert all(math.isnan(v) for v in stats.bootstrap_ci([]))
    assert stats.bootstrap_ci([1.0] * 5) == (1.0, 1.0, 1.0)


def test_paired_diff():
    a = {"q1": 0.9, "q2": 0.8, "q3": 0.7, "q4": 0.95, "extra": 0.0}
    b = {"q1": 0.5, "q2": 0.6, "q3": 0.4, "q4": 0.5}
    d = stats.paired_diff(a, b)
    assert d["n"] == 4 and d["mean"] == pytest.approx(0.3375) and d["ci_low"] > 0 and d["p_value"] < 0.05
    same = stats.paired_diff(b, b)
    assert same["mean"] == 0 and same["p_value"] == 1.0


def test_wilson():
    p, lo, hi = stats.wilson(5, 10)
    assert p == 0.5 and lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert stats.wilson(0, 10)[1] == 0.0


def test_aggregate_shape_repeats_and_skips():
    def sr(q, sys, metric, v, rep=0, n=1):
        return ScoreRecord(question_id=q, system=sys, repeat=rep, metric=metric, value=v, n=n)

    scores = [sr("q1", "full", "m", 1.0, 0), sr("q1", "full", "m", 0.0, 1), sr("q2", "full", "m", 1.0),
              sr("q3", "full", "m", 0.0, n=0), sr("q1", "base", "m", 0.2)]
    agg = stats.aggregate(scores)
    assert set(agg) == {"full", "base"}
    assert set(agg["full"]["m"]) == {"mean", "ci_low", "ci_high", "n"}
    assert agg["full"]["m"]["mean"] == pytest.approx(0.75) and agg["full"]["m"]["n"] == 2
    assert agg == stats.aggregate([s.model_dump() for s in scores])  # dicts work, deterministic
    cmp = stats.compare_systems(scores, "m", "full", "base")
    assert cmp["n"] == 1 and cmp["mean"] == pytest.approx(0.3)


# --- llm ----------------------------------------------------------------------------------------------------


class FakeBudget:
    def __init__(self, cap):
        self.cap, self.spent = cap, 0.0

    def add(self, usd):
        self.spent += usd

    def exhausted(self):
        return self.cap is not None and self.spent >= self.cap


async def test_make_llm_cache_cost_retry_budget(tmp_path, monkeypatch):
    from lgeval.costs import Pricing

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://or.test/api/v1")

    async def no_sleep(_):
        return None

    monkeypatch.setattr("lgeval.metrics.llm.asyncio.sleep", no_sleep)
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        if len(seen) == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}],
                                         "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}})

    budget = FakeBudget(1.0)
    llm = make_llm("judge/x", budget=budget, cache_dir=tmp_path, transport=httpx.MockTransport(handler))
    llm.pricing = {"judge/x": Pricing(in_per_m=0.4, out_per_m=1.6)}
    assert await llm("sys", "user") == '{"ok": 1}'
    assert len(seen) == 2 and seen[1]["response_format"] == {"type": "json_object"} and seen[1]["temperature"] == 0
    assert llm.stats.usd == pytest.approx(1.2) and budget.spent == pytest.approx(1.2)
    assert await llm("sys", "user") == '{"ok": 1}'  # cache hit: free, no request
    assert len(seen) == 2 and llm.stats.cache_hits == 1 and budget.spent == pytest.approx(1.2)
    with pytest.raises(BudgetExhausted):
        await llm("sys", "other user")
    await llm.aclose()
    # a fresh instance reads the same disk cache
    llm2 = make_llm("judge/x", cache_dir=tmp_path, transport=httpx.MockTransport(handler))
    assert await llm2("sys", "user") == '{"ok": 1}' and len(seen) == 2


# --- commands -----------------------------------------------------------------------------------------------


def all_handlers():
    def verify(user):
        n = sum(1 for line in user.splitlines() if line[:2] in {"1.", "2."})
        return json.dumps({"results": [{"id": i, "label": "supported"} for i in range(1, n + 1)]})

    def cite(user):
        n = user.count("<statement")
        return json.dumps({"results": [{"sid": i, "joint": True, "citations": []} for i in range(1, n + 1)]})

    def label(user):
        keys = [line.split(":")[0] for line in user.split("Claims:")[1].strip().splitlines()]
        return json.dumps({"results": [{"key": k, "label": "true"} for k in keys]})

    return {
        prompts.ATOMIC_SYSTEM: '{"facts": ["fact one", "fact two"]}',
        prompts.VERIFY_SYSTEM: verify,
        prompts.CITATION_SYSTEM: cite,
        prompts.ACCURACY_SYSTEM: '{"grade": "correct"}',
        prompts.LABEL_SYSTEM: label,
        prompts.PAIRWISE_SYSTEM: '{"winner": "tie"}',
    }


class CountingFake:
    def __init__(self):
        self.inner = fake_llm(all_handlers())
        self.stats = type("S", (), {"calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0})()

    async def __call__(self, system, user):
        self.stats.calls += 1
        return await self.inner(system, user)

    async def aclose(self):
        pass


def write_results(d):
    d.mkdir(parents=True, exist_ok=True)
    runs = [make_run(), make_run(system="no_debate"), make_run(system="no_debate", repeat=1),
            make_run(system="ext", claims=[], status="imported"), make_run(system="broken", status="failed")]
    (d / "runs.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in runs))
    q = Question(id="q1", question="When?", dataset="custom", gold_answer="2028")
    (d / "questions.jsonl").write_text(q.model_dump_json() + "\n")


def score_args(d, **kw):
    base = dict(results=[str(d)], metrics="factual,citation,accuracy,calibration", judge_model="fake/judge",
                systems="", max_usd=None, dry_run=False, fetch_urls=False, concurrency=2, force=False)
    base.update(kw)
    return argparse.Namespace(**base)


async def test_cmd_score_dry_run_and_resume(tmp_path, monkeypatch, capsys):
    d = tmp_path / "res"
    write_results(d)
    monkeypatch.setattr(commands, "_price", lambda m: (0.4, 1.6))
    fakes: list = []

    def fake_make(model, budget=None):
        fakes.append(CountingFake())
        return fakes[-1]

    monkeypatch.setattr(commands, "make_llm", fake_make)
    assert await commands.cmd_score(score_args(d, dry_run=True)) == 0
    assert "4 run(s) to score" in capsys.readouterr().out and not (d / "scores.jsonl").exists()

    assert await commands.cmd_score(score_args(d)) == 0
    rows = [json.loads(x) for x in (d / "scores.jsonl").read_text().splitlines()]
    metrics = {(r["system"], r["repeat"], r["metric"]) for r in rows}
    assert ("full", 0, "factual_precision") in metrics and ("ext", 0, "citation_recall") in metrics
    assert ("ext", 0, "brier") not in metrics and not any(r["system"] == "broken" for r in rows)
    labels = (d / "claim_labels.jsonl").read_text().splitlines()
    assert len(labels) == 9  # 3 LiveGraph runs x 3 claims
    first_calls = fakes[-1].stats.calls
    assert first_calls > 0

    assert await commands.cmd_score(score_args(d)) == 0  # resume: nothing left
    assert fakes[-1].stats.calls == 0
    assert len((d / "scores.jsonl").read_text().splitlines()) == len(rows)


async def test_cmd_pairwise(tmp_path, monkeypatch):
    d = tmp_path / "res"
    write_results(d)
    monkeypatch.setattr(commands, "_price", lambda m: (0.4, 1.6))
    monkeypatch.setattr(commands, "make_llm", lambda model, budget=None: CountingFake())
    args = argparse.Namespace(results=[str(d)], a="full", b="no_debate", judge_model="fake/judge", out="",
                              max_usd=None, dry_run=False, concurrency=2, force=False)
    assert await commands.cmd_pairwise(args) == 0
    recs = [PairwiseRecord.model_validate_json(x) for x in (d / "pairwise.jsonl").read_text().splitlines()]
    assert len(recs) == 1 and recs[0].winner == "tie"  # full has 1 repeat -> zipped with no_debate rep 0


def test_pair_runs_and_estimate():
    runs = [make_run(), make_run(repeat=1), make_run(system="ext", status="imported"), make_run(system="x")]
    pairs = commands.pair_runs(runs, "full", "ext")
    assert [(a.repeat, b.system) for a, b in pairs] == [(0, "ext")]
    calls, tin, tout = commands.estimate(make_run(), None, "citation")
    assert calls == 1 and tin > 0 and tout > 0
    assert commands.estimate(make_run(), None, "accuracy") == (0, 0, 0)
    assert commands.estimate(make_run(), None, "factual")[0] >= 2


def test_register_adds_subcommands():
    from lgeval.metrics import register

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    register(sub)
    a = parser.parse_args(["score", "--results", "x", "--dry-run", "--fetch-urls"])
    assert a.dry_run and a.fetch_urls and a.judge_model == "xiaomi/mimo-v2.6-pro"
    b = parser.parse_args(["pairwise", "--results", "x", "--a", "full", "--b", "no_debate"])
    assert b.a == "full" and b.func is commands.cmd_pairwise


# --- search behaviour (from snapshots, no judge) ---------------------------------------------------------------


def _search_snapshot() -> dict:
    def step(tool, summary="3 chunks", args="q", parent=None):
        return {"tool": tool, "summary": summary, "args_summary": args, "parent_call_id": parent}

    def ret(url, cited):
        return {"chunk_id": url + str(cited), "source_url": url, "cited": cited}

    planned = {
        "id": "v1", "output": {"search_stats": {"stopped_by": "rule", "queries": {"explore": 1, "challenge": 1}}},
        "trace": [step("web_search"), step("reflect"), step("web_search", args="query=x, kind=challenge"),
                  step("reflect"), step("read_source"), step("ask_rebuttal", parent="nested")],
        "retrievals": [ret("https://www.a.org/1", True), ret("https://b.org/2", True), ret("https://a.org/3", False),
                       ret("https://c.org/4", False)],
    }
    follow_up = {
        "id": "v2", "output": {"findings": []},
        "trace": [step("web_search"), step("web_search"), step("web_search", summary="budget exhausted")],
        "retrievals": [ret("https://d.org/1", True), ret("https://d.org/2", False)],
    }
    return {
        "nodes": [
            {"id": "n1", "type": "research", "current_version_id": "v1", "config": {"params": {}}},
            {"id": "n2", "type": "research", "current_version_id": "v2",
             "config": {"params": {"follow_up": True, "depth": 2}}},
            {"id": "n9", "type": "research", "current_version_id": "v9", "removed": True, "config": {}},
            {"id": "n3", "type": "claims", "current_version_id": "v3", "config": {}},
        ],
        # GET /runs/{id} shape: node id -> that node's current version
        "versions": {"n1": planned, "n2": follow_up, "n9": {"id": "v9", "trace": [step("web_search")]},
                     "n3": {"id": "v3", "trace": [], "retrievals": []}},
    }


def test_search_metrics_from_snapshot(tmp_path):
    import json

    from lgeval.metrics import search
    from lgeval.schema import RunRecord

    m = search.run_metrics(_search_snapshot())
    assert m["search_tool_calls"] == (4.0, 2)  # 5 top-level + 3, nested call ignored
    assert m["search_budget_exhausted"] == (0.5, 2) and m["search_rule_stop"] == (0.5, 2)
    assert m["search_rounds"] == (1.0, 2) and m["search_challenge"] == (0.5, 2)
    assert m["search_domains_cited"] == (1.5, 2)  # a.org + b.org, d.org
    assert m["search_cited_share"] == (0.5, 6)
    assert m["search_follow_ups"] == (1.0, 2) and m["search_follow_up_depth"] == (2.0, 2)
    assert search.run_metrics({"nodes": [], "versions": {}}) is None

    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "r.json").write_text(json.dumps(_search_snapshot()))
    run = RunRecord(question_id="q1", dataset="custom", system="full", run_id="r", status="done",
                    snapshot_path="snapshots/r.json")
    recs = search.score_search(run, tmp_path)
    assert {r.metric for r in recs} == set(search.METRICS) and all(r.judge_model is None for r in recs)
    assert search.score_search(run.model_copy(update={"snapshot_path": None}), tmp_path) == []
