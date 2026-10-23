from __future__ import annotations

from pathlib import Path

from lgeval.schema import Manifest, RunRecord, TokenUsage
from lgeval.store import DEFAULT_ROOT, REPO_ROOT, ResultStore


def test_default_root_is_repo_eval_results() -> None:
    assert (REPO_ROOT / "apps" / "api" / "lgeval").is_dir()
    assert DEFAULT_ROOT.name == "eval-results" or DEFAULT_ROOT.exists()


def test_layout_append_read_resume(tmp_path: Path) -> None:
    st = ResultStore(tmp_path, "quality", run_tag="t1")
    assert st.dir == tmp_path / "quality" / "t1" and st.dir.is_dir()
    r1 = RunRecord(question_id="q1", dataset="custom", system="full", status="done",
                   usage={"m": TokenUsage(tokens_in=1, tokens_out=2)})
    r2 = RunRecord(question_id="q2", dataset="custom", system="full", repeat=1, status="failed")
    st.append("runs", r1)
    st.append("runs", r2)
    with st.path("runs").open("a") as f:
        f.write('{"question_id": "trunc')  # crash mid-write
    got = st.read("runs", RunRecord)
    assert [g.question_id for g in got] == ["q1", "q2"] and got[0].usage["m"].tokens_out == 2
    keys = st.done_keys("runs", lambda r: f"{r['question_id']}/{r['system']}/{r['repeat']}")
    assert keys == {"q1/full/0", "q2/full/1"}
    assert st.read("missing", RunRecord) == [] and st.done_keys("missing", str) == set()

    again = ResultStore.open(st.dir)
    assert again.experiment == "quality" and again.run_tag == "t1" and len(again.read("runs", RunRecord)) == 2


def test_manifest_and_snapshots(tmp_path: Path) -> None:
    st = ResultStore(tmp_path, "recompute")
    st.write_manifest(Manifest(experiment="recompute", args={"max_usd": 5}))
    m = st.read_manifest()
    assert m is not None and m.args == {"max_usd": 5}
    assert m.git_sha is None or len(m.git_sha) >= 7  # tolerated outside a git checkout

    p = st.save_snapshot("run-1", {"run": {"id": "run-1"}, "last_seq": 3})
    assert p == st.dir / "snapshots" / "run-1.json"
    assert st.load_snapshot("run-1")["last_seq"] == 3
