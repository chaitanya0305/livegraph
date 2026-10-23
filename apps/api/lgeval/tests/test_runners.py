"""Runners offline: configs, the quality matrix against a fake API (httpx.MockTransport), the single-agent
baseline with a FunctionModel + fake search, and importing external briefs."""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from lgeval import cli
from lgeval.client import LiveGraphClient
from lgeval.costs import Budget, Pricing
from lgeval.runners import baseline, configs, quality
from lgeval.schema import Question, RunRecord
from lgeval.store import ResultStore

PRICING = {"fake-smart": Pricing(in_per_m=1.0, out_per_m=2.0), "fake-fast": Pricing(in_per_m=0.5, out_per_m=1.0)}
QS = [Question(id=f"q{i}", question=goal, dataset="custom") for i, goal in enumerate(["solar", "ask", "hang", "boom"], 1)]


# ---------------------------------------------------------------- configs


def test_configs_pin_every_knob_and_differ_in_one():
    envs = configs.resolve("full,no_debate,no_critic,single_shot,no_context_tools")
    assert envs["full"] == configs.BASE
    for name, env in envs.items():
        assert set(env) == set(configs.BASE), name
        diff = {k for k in env if env[k] != configs.BASE[k]}
        assert len(diff) == (0 if name == "full" else 1), name
    assert envs["no_debate"]["DISABLED_NODE_TYPES"] == "prosecutor,defender,judge"
    assert envs["no_critic"]["DISABLED_NODE_TYPES"] == "critic"
    assert envs["single_shot"]["AGENT_TOOLS"] == ""
    assert envs["no_context_tools"]["CONTEXT_TOOLS"] == "false"
    assert all(c.ablates for c in configs.CONFIGS.values())
    assert configs.parse_names(" full, full ,no_critic") == ["full", "no_critic"]
    with pytest.raises(ValueError, match="nope"):
        configs.resolve("full,nope")


def test_config_env_parses_in_livegraph_settings(monkeypatch):
    from livegraph.config import Settings

    for k, v in configs.resolve("single_shot")["single_shot"].items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None)
    assert s.agent_tools == "" and s.context_tools is True and s.disabled_node_types == ""


# ---------------------------------------------------------------- fake LiveGraph API


class FakeApi:
    """Minimal /api: goal "solar" finishes, "ask" waits for a human_input answer first, "hang" never settles,
    "boom" fails at POST /runs. Each poll of a run advances it one step."""

    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}
        self.answers: list[tuple[str, str]] = []
        self.cancelled: list[str] = []
        self.created = 0

    def _snap(self, rid: str) -> dict:
        r = self.runs[rid]
        status = r["status"]
        nodes = [
            {"id": f"{rid}-claims", "type": "claims", "key": "claims", "status": "done", "removed": False},
            {"id": f"{rid}-editor", "type": "editor", "key": "editor",
             "status": "done" if status == "done" else "pending", "removed": False},
        ]
        if r["goal"] == "ask":
            nodes.append({"id": f"{rid}-hi", "type": "human_input", "key": "audience",
                          "status": "waiting_input" if status == "waiting_input" else "done", "removed": False})
        versions = {
            f"{rid}-claims": {"id": f"{rid}-cv2", "output": {"claims": [
                {"key": "C1", "text": "Solar grew.", "support_chunk_ids": ["ch1"]}]},
                "output_text": "- C1", "retrievals": [
                {"chunk_id": "ch1", "document_title": "Solar", "source_url": "https://s.example", "text": "Solar grew 20%."}]},
        }
        if status == "done":
            versions[f"{rid}-editor"] = {"id": f"{rid}-ev", "output": {"markdown": "x"},
                                         "output_text": "# Brief\n\nSolar capacity grew (C1)."}
        return {"run": {"id": rid, "status": status}, "nodes": nodes, "edges": [], "versions": versions,
                "claims": [{"key": "C1", "text": "Solar grew.", "confidence": 0.8, "status": "active"}],
                "last_seq": r["seq"]}

    def handler(self, req: httpx.Request) -> httpx.Response:
        path, m = req.url.path, req.method
        if m == "POST" and path == "/api/runs":
            goal = json.loads(req.content)["goal"]
            if goal == "boom":
                return httpx.Response(500, json={"detail": "boom"})
            self.created += 1
            rid = f"run{self.created}"
            self.runs[rid] = {"goal": goal, "status": "planning", "seq": 1, "polls": 0}
            return httpx.Response(200, json=self._snap(rid))
        if mm := re.fullmatch(r"/api/runs/(\w+)", path):
            r = self.runs[mm.group(1)]
            r["polls"] += 1
            if r["goal"] != "hang" and r["status"] == "planning" and r["polls"] >= 2:
                nxt = {"ask": "waiting_input", "fail": "failed"}.get(r["goal"], "done")
                r["status"], r["seq"] = nxt, r["seq"] + 5
            return httpx.Response(200, json=self._snap(mm.group(1)))
        if mm := re.fullmatch(r"/api/runs/(\w+)/cancel", path):
            self.cancelled.append(mm.group(1))
            self.runs[mm.group(1)]["status"] = "cancelled"
            return httpx.Response(200, json={})
        if re.fullmatch(r"/api/runs/(\w+)/events", path):
            return httpx.Response(200, json=[])
        if mm := re.fullmatch(r"/api/nodes/(\w+)-hi/input", path):
            self.answers.append((mm.group(1), json.loads(req.content)["answer"]))
            r = self.runs[mm.group(1)]
            r["status"], r["seq"] = "done", r["seq"] + 3
            return httpx.Response(202)
        if mm := re.fullmatch(r"/api/nodes/(\w+)-(\w+)/versions", path):
            rid, key = mm.groups()
            if key == "claims":  # two versions: tokens of the superseded one count too
                return httpx.Response(200, json=[
                    {"id": f"{rid}-cv1", "model": "fake-fast", "tokens_in": 1000, "tokens_out": 100},
                    {"id": f"{rid}-cv2", "model": "fake-fast", "tokens_in": 1000, "tokens_out": 100,
                     "retrievals": self._snap(rid)["versions"][f"{rid}-claims"]["retrievals"]},
                ])
            if key == "editor" and self.runs[rid]["status"] == "done":
                return httpx.Response(200, json=[{"id": f"{rid}-ev", "model": "fake-smart", "tokens_in": 2000,
                                                  "tokens_out": 500}])
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={"detail": path})


class FakeServer:
    def __init__(self, envs: list) -> None:
        self.envs = envs

    def __call__(self, name: str, env: dict[str, str]):
        @asynccontextmanager
        async def cm():
            self.envs.append((name, env))
            yield type("Srv", (), {"base_url": f"http://fake-{name}"})()

        return cm()


def _client_factory(api: FakeApi):
    return lambda base_url: LiveGraphClient(base_url, transport=httpx.MockTransport(api.handler))


async def test_quality_matrix_records_every_outcome(tmp_path: Path):
    api, envs = FakeApi(), []
    store = ResultStore(tmp_path, "quality", run_tag="t1")
    cfg = configs.resolve("full,no_debate")
    jobs = quality.plan_jobs(QS, list(cfg), repeats=1)
    res = await quality.run_matrix(
        jobs, cfg, store=store, budget=Budget(None), pricing=PRICING, concurrency=2, timeout=0.5, poll=0.01,
        server_factory=FakeServer(envs), client_factory=_client_factory(api),
    )
    assert [n for n, _ in envs] == ["full", "no_debate"] and envs[1][1]["DISABLED_NODE_TYPES"]
    recs = {(r.system, r.question_id): r for r in store.read("runs", RunRecord)}
    assert len(recs) == 8 and res.skipped == 0
    ok = recs[("full", "q1")]
    assert ok.status == "done" and ok.run_id and ok.brief.startswith("# Brief")
    # tokens over ALL versions: two claims versions + the editor
    assert ok.usage["fake-fast"].tokens_in == 2000 and ok.usage["fake-smart"].tokens_out == 500
    assert ok.cost_usd == pytest.approx(2000 * 0.5e-6 + 200 * 1e-6 + 2000 * 1e-6 + 500 * 2e-6)
    assert ok.claims[0].key == "C1" and ok.claims[0].confidence == 0.8
    assert any(c.claim_keys == ["C1"] and c.chunk_ids == ["ch1"] for c in ok.citations)
    assert ok.evidence and ok.evidence[0].url == "https://s.example"
    assert ok.n_nodes == 2
    snap = json.loads((store.dir / ok.snapshot_path).read_text())
    assert snap["run"]["id"] == ok.run_id and len(snap["node_versions"][f"{ok.run_id}-claims"]) == 2
    asked = recs[("full", "q2")]
    assert asked.status == "done" and res.auto_answers == 2
    assert all(a == quality.NEUTRAL_ANSWER for _, a in api.answers)
    hung = recs[("full", "q3")]
    assert hung.status == "timeout" and hung.run_id in api.cancelled and hung.usage  # spend still recorded
    boom = recs[("full", "q4")]
    assert boom.status == "failed" and "500" in (boom.error or "") and boom.run_id is None


async def test_failed_run_with_pending_nodes_settles_fast(tmp_path: Path):
    api = FakeApi()
    store = ResultStore(tmp_path, "quality", run_tag="f")
    job = quality.Job(Question(id="qf", question="fail", dataset="custom"), "full", 0)
    async with _client_factory(api)("http://fake") as client:
        rec, _ = await quality.run_one(client, job, store=store, pricing=PRICING, timeout=5, poll=0.01)
    assert rec.status == "failed" and (rec.wall_s or 0) < 2 and rec.brief == "" and rec.usage


async def test_quality_resume_and_budget(tmp_path: Path):
    api = FakeApi()
    store = ResultStore(tmp_path, "quality", run_tag="t2")
    store.append("runs", RunRecord(question_id="q1", dataset="custom", system="full", status="done"))
    done = store.done_keys("runs", quality.run_key)
    jobs = quality.plan_jobs(QS[:2], ["full"], repeats=2, done=done)
    assert [j.key for j in jobs["full"]] == ["q2|full|0", "q1|full|1", "q2|full|1"]
    budget = Budget(0.001)  # the first finished run exhausts it
    res = await quality.run_matrix(
        jobs, configs.resolve("full"), store=store, budget=budget, pricing=PRICING, concurrency=1, timeout=5,
        poll=0.01, server_factory=FakeServer([]), client_factory=_client_factory(api),
    )
    assert len(res.records) == 1 and res.skipped == 2 and budget.exhausted()
    assert api.created == 1


async def test_quality_server_failure_does_not_crash(tmp_path: Path):
    def broken(name, env):
        raise RuntimeError("no uvicorn")

    store = ResultStore(tmp_path, "quality", run_tag="t3")
    jobs = quality.plan_jobs(QS[:2], ["full"], repeats=1)
    res = await quality.run_matrix(jobs, configs.resolve("full"), store=store, budget=Budget(None), pricing=PRICING,
                                   server_factory=broken)
    assert res.records == [] and res.skipped == 2


def test_quality_dry_run_and_registration(tmp_path: Path, capsys):
    assert cli.main(["quality", "--configs", "full,no_debate", "--limit", "2", "--repeats", "3", "--dry-run",
                     "--offline", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "no_debate" in out and "6 runs" in out and "est $" in out
    assert not (tmp_path / "quality").exists()  # a dry run writes nothing
    assert cli.main(["quality", "--configs", "bogus", "--dry-run", "--offline", "--out", str(tmp_path)]) == 2
    assert cli.main(["baseline", "--limit", "1", "--dry-run", "--offline", "--out", str(tmp_path)]) == 0
    assert "single_agent" in capsys.readouterr().out
    parser = cli.build_parser()
    assert not any(m in (parser.epilog or "") for m in ("runners.quality", "runners.baseline"))


# ---------------------------------------------------------------- single-agent baseline


def _baseline_model() -> FunctionModel:
    def respond(messages, info: AgentInfo) -> ModelResponse:
        returns = [p for m in messages for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart)]
        if len(returns) < 2:
            return ModelResponse(parts=[ToolCallPart("web_search", {"query": f"solar {len(returns)}"})])
        return ModelResponse(parts=[TextPart(
            "Solar capacity doubled [1]. Batteries got cheaper [2][1].\n\nUnsourced aside.\n\n"
            "## Sources\n[1] https://a.example/solar\n[2] https://b.example/battery\n"
        )])

    return FunctionModel(respond, model_name="fake-smart")


async def _fake_search(query: str, max_results: int) -> list[dict]:
    return [
        {"url": "https://a.example/solar", "title": "Solar", "content": "Solar capacity doubled in 2024."},
        {"url": "https://b.example/battery", "title": "Battery", "content": f"Battery prices fell ({query})."},
    ]


async def test_single_agent_baseline_record():
    q = Question(id="q1", question="Is solar growing?", dataset="custom")
    rec = await baseline.run_single_agent(q, model=_baseline_model(), model_name="fake-smart", search=_fake_search,
                                          pricing=PRICING, search_budget=10)
    assert rec.status == "done" and rec.system == "single_agent" and rec.run_id is None
    assert "Sources" not in rec.brief and "https://" not in rec.brief
    cits = {c.sentence: c.urls for c in rec.citations}
    assert cits["Solar capacity doubled [1]."] == ["https://a.example/solar"]
    assert cits["Batteries got cheaper [2][1]."] == ["https://b.example/battery", "https://a.example/solar"]
    assert cits["Unsourced aside."] == []
    assert {e.url for e in rec.evidence} == {"https://a.example/solar", "https://b.example/battery"}
    assert len(rec.evidence) == 3  # same solar snippet twice is kept once; battery text differs per query
    assert rec.usage["fake-smart"].tokens_in > 0 and rec.cost_usd and rec.cost_usd > 0


async def test_search_budget_is_enforced():
    calls = []

    async def search(query, n):
        calls.append(query)
        return await _fake_search(query, n)

    def respond(messages, info):
        returns = [p for m in messages for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart)]
        if len(returns) < 4:
            return ModelResponse(parts=[ToolCallPart("web_search", {"query": f"q{len(returns)}"})])
        assert "BUDGET EXHAUSTED" in str(returns[-1].content)
        return ModelResponse(parts=[TextPart("Done [1].")])

    q = Question(id="q1", question="x", dataset="custom")
    rec = await baseline.run_single_agent(q, model=FunctionModel(respond), model_name="m", search=search,
                                          pricing=PRICING, search_budget=2)
    assert rec.status == "done" and len(calls) == 2


async def test_baseline_failure_is_recorded(tmp_path: Path):
    def respond(messages, info):
        raise RuntimeError("provider down")

    store = ResultStore(tmp_path, "baseline", run_tag="b")
    jobs = quality.plan_jobs(QS[:2], ["single_agent"], repeats=1)["single_agent"]
    recs = await baseline.run_baselines(jobs, store=store, budget=Budget(None), model=FunctionModel(respond),
                                        model_name="m", search=_fake_search, pricing=PRICING)
    assert len(recs) == 2 and all(r.status == "failed" and "provider down" in r.error for r in recs)
    assert len(store.read("runs", RunRecord)) == 2


# ---------------------------------------------------------------- import


def test_import_external_briefs(tmp_path: Path, capsys):
    d = tmp_path / "briefs"
    d.mkdir()
    (d / "q1.md").write_text(
        "# Answer\n\nSolar grew fast [1]. See also [the IEA](https://iea.example/report).\n\n"
        "- Batteries fell 90% (https://bnef.example/x).\n\nNo source here.\n\n"
        "## References\n1. [Ember](https://ember.example/solar)\n2. https://unused.example\n"
    )
    (d / "q9.md").write_text("Footnote style.[^1]\n\n[^1]: https://fn.example/a\n")
    (d / "notes.txt").write_text("ignored")
    recs = baseline.import_briefs(d, system="chatgpt_dr", questions={}, dataset="custom")
    assert [r.question_id for r in recs] == ["q1", "q9"]
    r = recs[0]
    assert r.status == "imported" and r.system == "chatgpt_dr" and r.evidence == []
    assert "References" not in r.brief and "unused" not in r.brief
    cits = {c.sentence: c.urls for c in r.citations}
    assert cits["Solar grew fast [1]."] == ["https://ember.example/solar"]
    assert cits["See also [the IEA](https://iea.example/report)."] == ["https://iea.example/report"]
    assert cits["Batteries fell 90% (https://bnef.example/x)."] == ["https://bnef.example/x"]
    assert cits["No source here."] == []

    out = tmp_path / "res"
    assert cli.main(["import", "--system", "chatgpt_dr", "--dir", str(d), "--out", str(out), "--tag", "x"]) == 0
    stored = ResultStore(out, "import", run_tag="x").read("runs", RunRecord)
    assert len(stored) == 2 and (out / "import" / "x" / "manifest.json").exists()
    assert cli.main(["import", "--system", "s", "--dir", str(tmp_path / "missing"), "--out", str(out)]) == 2


def test_editor_done_marks_partial_runs() -> None:
    from lgeval.runners.quality import _editor_done

    snap = {"nodes": [{"type": "critic", "status": "failed"}, {"type": "editor", "status": "done"}]}
    assert _editor_done(snap)
    assert not _editor_done({"nodes": [{"type": "editor", "status": "pending"}]})
