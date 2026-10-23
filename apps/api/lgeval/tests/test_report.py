from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from pydantic import BaseModel

from lgeval import report
from lgeval.schema import ClaimLabel, ClaimOut, Manifest, PairwiseRecord, RecomputeRecord, RunRecord, ScoreRecord

SVG_NS = "{http://www.w3.org/2000/svg}"


def _write(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")


def _synthetic(root: Path) -> None:
    rng = random.Random(1)
    runs, scores, pairs, labels, recs = [], [], [], [], []
    for sys, base in [("full", 0.85), ("no_debate", 0.75), ("single_agent", 0.65)]:
        for q in range(6):
            for rep in range(2):
                runs.append(RunRecord(question_id=f"q{q}", dataset="custom", system=sys, repeat=rep, status="done",
                                      wall_s=100 + q, cost_usd=0.13, claims=[ClaimOut(key="C1", text="x")]))
                for m in report.QUALITY_METRICS:
                    scores.append(ScoreRecord(question_id=f"q{q}", system=sys, repeat=rep, metric=m,
                                              value=min(1.0, base + rng.uniform(-0.05, 0.05)),
                                              details={"cost_usd": 0.001}))
    runs.append(RunRecord(question_id="q0", dataset="custom", system="full", repeat=9, status="failed"))
    for q in range(10):
        pairs.append(PairwiseRecord(question_id=f"q{q}", system_a="full", system_b="no_debate",
                                    winner="a" if q < 7 else "tie", first_order="a", swapped_order="a",
                                    judge_model="judge"))
        pairs.append(PairwiseRecord(question_id=f"q{q}", system_a="single_agent", system_b="full",
                                    winner="b" if q < 5 else "a", first_order="b", swapped_order="b",
                                    judge_model="judge"))
    for i in range(60):
        conf = rng.random()
        labels.append(ClaimLabel(question_id=f"q{i % 6}", system="full", claim_key=f"C{i}", claim_text="c",
                                 confidence=conf, struck=conf < 0.2,
                                 label="true" if rng.random() < conf else "false", labeler="llm:judge"))
    labels.append(ClaimLabel(question_id="q0", system="full", claim_key="CU", claim_text="c", confidence=0.5,
                             label="unverifiable", labeler="human:a"))
    for mut, frac in [("identical", 0.0), ("distrust", 0.4), ("steer", 0.25)]:
        for q in range(4):
            recs.append(RecomputeRecord(question_id=f"q{q}", run_id=f"r{q}", mutation=mut, target="t",
                                        wall_s=200 * frac + 1, cost_usd=0.13 * frac, nodes_total=10,
                                        nodes_impacted=6, nodes_recomputed=int(10 * frac), nodes_reused=2,
                                        baseline="fresh_run", baseline_wall_s=200, baseline_cost_usd=0.13))
    _write(root / "quality" / "t1" / "runs.jsonl", runs)
    _write(root / "score" / "t1" / "scores.jsonl", scores)
    _write(root / "pairwise" / "t1" / "pairwise.jsonl", pairs)
    _write(root / "score" / "t1" / "claim_labels.jsonl", labels)
    _write(root / "recompute" / "t1" / "recompute.jsonl", recs)
    (root / "quality" / "t1" / "manifest.json").write_text(
        Manifest(experiment="quality", models={"smart": "openai/gpt-4.1"}).model_dump_json(), encoding="utf-8")
    with (root / "quality" / "t1" / "runs.jsonl").open("a") as fh:
        fh.write("{not json\n")


def _assert_svg(path: Path) -> None:
    tree = ET.parse(path).getroot()
    assert tree.find(f"{SVG_NS}title") is not None and tree.find(f"{SVG_NS}desc") is not None
    assert tree.get("role") == "img"


def test_full_report(tmp_path: Path) -> None:
    _synthetic(tmp_path / "results")
    out = tmp_path / "out"
    summary = report.build_report(report.load_results([tmp_path / "results"]), out)

    assert summary["inputs"]["skipped_lines"] == 1
    q = summary["quality"]
    assert list(q)[:3] == ["full", "no_debate", "single_agent"]
    fp = q["full"]["factual_precision"]
    assert fp["n"] == 6 and fp["ci_low"] <= fp["mean"] <= fp["ci_high"]
    assert q["full"]["runs"] == {"total": 13, "ok": 12}
    assert q["full"]["cost_usd"]["mean"] == pytest.approx(0.13)

    p = summary["pairwise"]
    assert p["no_debate"] == pytest.approx({"win": 7, "tie": 3, "loss": 0, "n": 10, "win_rate": 0.85,
                                            "ci_low": p["no_debate"]["ci_low"], "ci_high": p["no_debate"]["ci_high"]})
    assert p["single_agent"]["win"] == 5 and p["single_agent"]["loss"] == 5
    assert 0 < p["no_debate"]["ci_low"] < 0.85 < p["no_debate"]["ci_high"] <= 1

    cal = summary["calibration"]["full"]
    assert cal["n"] == 60 and cal["n_unverifiable"] == 1
    assert 0 <= cal["brier"] <= 1 and 0 <= cal["ece"] <= 1 and 0.5 < cal["auroc"] <= 1
    assert cal["n_struck"] > 0 and 0 <= cal["struck_false_rate"] <= 1
    assert sum(b["n"] for b in cal["bins"]) == 60

    rc = summary["recompute"]
    assert list(rc) == ["identical", "distrust", "steer"]
    assert rc["identical"]["cost_pct"]["mean"] == pytest.approx(0.0)
    assert rc["distrust"]["cost_pct"]["mean"] == pytest.approx(40.0)
    assert rc["steer"]["nodes_recomputed"] == pytest.approx(2.0)

    sc = summary["study_cost"]
    assert sc["judge_cost_recorded"]
    assert "recompute: fresh-run baselines" in sc["rows"]
    assert sc["total_usd"] > 36 * 0.13

    md = (out / "results.md").read_text()
    for title in ["Table 1", "Table 2", "Table 3", "Table 4", "Table 5"]:
        assert title in md
    assert "Omitted" not in md
    assert "figures/reliability.svg" in md

    for name in ["quality", "pairwise", "calibration", "recompute", "study_cost"]:
        t = (out / "tables" / f"{name}.tex").read_text()
        assert r"\toprule" in t and r"\bottomrule" in t and "Results pending" not in t
    assert r"no\_debate" in (out / "tables" / "quality.tex").read_text()

    for fig in ["quality.svg", "reliability.svg", "recompute_cost.svg"]:
        _assert_svg(out / "figures" / fig)

    data = json.loads((out / "summary.json").read_text())
    assert data["quality"]["full"]["factual_precision"]["n"] == 6
    assert data["inputs"]["models"] == {"smart": ["openai/gpt-4.1"]}


def test_report_is_deterministic(tmp_path: Path) -> None:
    _synthetic(tmp_path / "r")
    a = report.summarize(report.load_results([tmp_path / "r"]))
    b = report.summarize(report.load_results([tmp_path / "r"]))
    a.pop("generated_at"), b.pop("generated_at")
    assert a == b


def test_empty_inputs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = report.build_report(report.load_results([tmp_path / "missing", tmp_path]), out)
    md = (out / "results.md").read_text()
    assert md.count("_Omitted:") == 5
    assert len(summary["notes"]) == 5
    for name in ["quality", "pairwise", "calibration", "recompute", "study_cost"]:
        assert "Results pending" in (out / "tables" / f"{name}.tex").read_text()
    assert not list((out / "figures").glob("*.svg"))
    assert json.loads((out / "summary.json").read_text())["quality"] == {}


def test_partial_inputs(tmp_path: Path) -> None:
    _write(tmp_path / "r" / "recompute.jsonl", [
        RecomputeRecord(question_id="q", run_id="r", mutation="edit", target="t", wall_s=1, cost_usd=0.0,
                        nodes_total=5, nodes_impacted=2, nodes_recomputed=0, nodes_reused=2, baseline="original_run",
                        baseline_wall_s=0, baseline_cost_usd=0)])
    _write(tmp_path / "r" / "claim_labels.jsonl", [
        ClaimLabel(question_id="q", system="full", claim_key="C1", claim_text="c", confidence=0.9, label="true",
                   labeler="human:a")])
    out = tmp_path / "out"
    summary = report.build_report(report.load_results([tmp_path / "r"]), out)
    md = (out / "results.md").read_text()
    assert "Table 4" in md and "Table 5" in md and md.count("_Omitted:") == 2  # quality, pairwise
    assert summary["calibration"]["full"]["auroc"] is None  # one class only
    assert summary["recompute"]["edit"]["cost_pct"]["n"] == 0  # zero baseline: no ratio, no crash
    assert json.loads((out / "summary.json").read_text())["recompute"]["edit"]["cost_pct"]["mean"] is None


def test_stats_helpers() -> None:
    assert report.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert report.auroc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert report.brier([1.0, 0.0], [1, 0]) == 0
    assert report.ece([0.95, 0.95], [1, 1]) == pytest.approx(0.05)
    lo, hi = report.wilson(0.5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert report.bootstrap_ci([0.3]) == (0.3, 0.3)
    assert report.tex("a_b & 5%") == r"a\_b \& 5\%"


def test_cli_register(tmp_path: Path) -> None:
    _synthetic(tmp_path / "r")
    out = tmp_path / "out"
    assert report.main(["--results", str(tmp_path / "r"), "--out", str(out)]) == 0
    assert (out / "summary.json").exists()


def test_quality_uses_one_judge_and_skips_undefined(tmp_path: Path) -> None:
    recs = [
        ScoreRecord(question_id="q1", system="full", metric="factual_precision", value=0.8, n=5, judge_model="j1"),
        ScoreRecord(question_id="q2", system="full", metric="factual_precision", value=0.6, n=5, judge_model="j1"),
        ScoreRecord(question_id="q3", system="full", metric="factual_precision", value=0.0, n=0, judge_model="j1"),
        ScoreRecord(question_id="q1", system="full", metric="factual_precision", value=0.1, n=5, judge_model="j2"),
    ]
    _write(tmp_path / "r" / "scores.jsonl", recs)
    out = tmp_path / "out"
    summary = report.build_report(report.load_results([tmp_path / "r"]), out)
    fp = summary["quality"]["full"]["factual_precision"]
    assert fp["n"] == 2 and fp["mean"] == pytest.approx(0.7)
    assert summary["inputs"]["judge_model"] == "j1"
    assert "other judges in the inputs are ignored: j2" in (out / "results.md").read_text()
