"""Claim C runner against a fake LiveGraph API (httpx.MockTransport) with a tiny hash-reuse engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from lgeval.client import LiveGraphClient
from lgeval.runners import recompute as rc
from lgeval.schema import RecomputeRecord, RunRecord, TokenUsage

T0 = datetime(2026, 10, 1, tzinfo=UTC)
CHUNKS = {"c1a": "d1", "c1b": "d1", "c2a": "d2", "c3a": "d3"}
RESEARCH_CHUNKS = {"r1": ["c1a", "c2a"], "r2": ["c1b", "c3a"], "r1_fu": ["c3a"]}
CONFIDENCE = {"C1": 0.9, "C2": 0.4, "C3": 0.7}
MODEL = {"research": "fast", "claims": "fast", "prosecutor": "fast", "defender": "fast",
         "judge": "smart", "editor": "smart", "plan": "smart"}


def h(x: object) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def cost(usage: dict[str, TokenUsage]) -> float:
    return sum(u.tokens_in * 1e-6 + u.tokens_out * 2e-6 for u in usage.values())


class FakeRun:
    """A run whose scheduler only advances when the run is read (like the real one between polls)."""

    def __init__(self, server: FakeServer, run_id: str, goal: str) -> None:
        self.server, self.id, self.goal = server, run_id, goal
        self.status = "running"
        self.events: list[dict] = []
        self.nodes: dict[str, dict] = {}
        self.order: list[str] = []
        self.edges: list[tuple[str, str]] = []
        self.versions: dict[str, list[dict]] = {}
        self.steers: dict[str, list[str]] = {}
        self.docs = {d: {"id": f"{run_id}-{d}", "title": f"Doc {d}", "origin": "web", "trust": "normal",
                         "run_id": run_id} for d in ("d1", "d2", "d3")}
        self.dirty: set[str] = set()
        self.force: set[str] = set()
        self.followed_up = False
        self.emit("run.status", None, {"status": "running"})
        for key, typ, parents in [("plan", "plan", []), ("r1", "research", ["plan"]), ("r2", "research", ["plan"]),
                                  ("claims", "claims", ["r1", "r2"]), ("prosecutor", "prosecutor", ["claims"]),
                                  ("defender", "defender", ["claims"]),
                                  ("judge", "judge", ["claims", "prosecutor", "defender"]),
                                  ("editor", "editor", ["claims", "judge"])]:
            self.add_node(key, typ, parents)

    # -- graph
    def nid(self, key: str) -> str:
        return f"{self.id}-{key}"

    def add_node(self, key: str, typ: str, parents: list[str], *, before: str | None = None) -> None:
        nid = self.nid(key)
        cfg = {"question": f"What about {key}?"} if typ == "research" else {}
        self.nodes[nid] = {"id": nid, "run_id": self.id, "key": key, "type": typ, "title": key.upper(),
                           "status": "pending", "config": cfg, "position": None, "created_by": "planner",
                           "current_version_id": None, "removed": False}
        self.order.insert(self.order.index(self.nid(before)) if before else len(self.order), nid)
        self.versions[nid] = []
        self.emit("node.added", nid, {"node": dict(self.nodes[nid])})
        for p in parents:
            self.edges.append((self.nid(p), nid))
        self.dirty.add(nid)

    def parents(self, nid: str) -> list[str]:
        return [s for s, t in self.edges if t == nid and not self.nodes[s]["removed"]]

    def children(self, nid: str) -> list[str]:
        return [t for s, t in self.edges if s == nid and not self.nodes[t]["removed"]]

    def descendants(self, nid: str) -> list[str]:
        out: list[str] = []
        stack = [nid]
        while stack:
            for c in self.children(stack.pop()):
                if c not in out:
                    out.append(c)
                    stack.append(c)
        return [n for n in self.order if n in out]

    def cur(self, nid: str) -> dict | None:
        vid = self.nodes[nid]["current_version_id"]
        return next((v for v in self.versions[nid] if v["id"] == vid), None)

    def emit(self, typ: str, nid: str | None, payload: dict) -> None:
        seq = self.server.next_seq()
        self.events.append({"seq": seq, "run_id": self.id, "type": typ, "node_id": nid, "payload": payload,
                            "ts": (T0 + timedelta(seconds=seq)).isoformat()})

    # -- engine
    def fingerprint(self) -> list:
        return sorted((d["id"], d["trust"]) for d in self.docs.values() if d["trust"] != "normal")

    def input_hash(self, nid: str) -> str:
        n = self.nodes[nid]
        ins = [f"{self.nodes[p]['key']}:{self.cur(p)['output_hash']}" for p in self.parents(nid)]
        fp = self.fingerprint() if n["type"] == "research" else None
        return h([n["type"], n["config"], n["title"], ins, self.steers.get(nid, []), fp])

    def compute(self, nid: str) -> tuple[dict, str, list[dict]]:
        n = self.nodes[nid]
        typ, key = n["type"], n["key"]
        distrusted = {k for k, d in self.docs.items() if d["trust"] == "distrusted"}
        if typ == "research":
            cids = [c for c in RESEARCH_CHUNKS[key] if CHUNKS[c] not in distrusted]
            q = n["config"].get("question") or n["title"]
            rets = [{"chunk_id": f"{self.id}-{c}", "document_id": self.docs[CHUNKS[c]]["id"],
                     "document_title": f"Doc {CHUNKS[c]}", "source_url": None, "text": c, "score": 1.0,
                     "trust": "normal", "rank": i, "cited": True, "via": "seed"} for i, c in enumerate(cids)]
            steer = "|".join(self.steers.get(nid, []))
            return {"findings": [{"point": f"{q} {steer}", "chunk_ids": [r["chunk_id"] for r in rets]}],
                    "gaps": []}, f"finding: {q}", rets
        if typ == "claims":
            chunks = {self.nodes[p]["key"]: [r["chunk_id"] for r in self.cur(p)["retrievals"]]
                      for p in self.parents(nid)}
            spec = [("C1", "r1", 0), ("C2", "r2", 0), ("C3", "r1", 1)]
            claims = [{"key": k, "text": f"claim {k} from {src}", "support_chunk_ids": [chunks[src][i]]}
                      for k, src, i in spec if len(chunks.get(src, [])) > i]
            return {"claims": claims}, "\n".join(f"- **{c['key']}** {c['text']}" for c in claims), []
        digest = h([self.cur(p)["output_hash"] for p in self.parents(nid)])[:8]
        if typ == "editor":
            keys = [c["key"] for c in self.claims() if c["status"] == "active"]
            text = f"# Brief\n\nFindings {digest}. " + " ".join(f"Point ({k})." for k in keys)
            return {"markdown": text}, text, []
        return {"digest": digest}, f"{typ} {digest}", []

    def new_version(self, nid: str, ih: str, output: dict, text: str, rets: list[dict], *, llm: bool,
                    reason: str) -> dict:
        n = self.nodes[nid]
        v = {"id": f"{nid}-v{len(self.versions[nid]) + 1}", "node_id": nid, "version": len(self.versions[nid]) + 1,
             "input_hash": ih, "output_hash": h([output, " ".join(text.split()),
                                                  sorted((r["chunk_id"], r["cited"]) for r in rets)]),
             "output": output, "output_text": text, "reason": reason, "steer_text": None,
             "model": MODEL[n["type"]] if llm else None, "tokens_in": 1000 if llm else None,
             "tokens_out": 100 if llm else None, "created_at": T0.isoformat(), "retrievals": rets, "trace": []}
        self.versions[nid].append(v)
        return v

    def apply(self, nid: str, v: dict, *, reused: bool) -> bool:
        prev = self.cur(nid)
        self.nodes[nid]["current_version_id"] = v["id"]
        self.emit("node.version", nid, {"version": v, "reused": reused})
        return prev is None or prev["output_hash"] != v["output_hash"]

    def stale(self, nids: list[str]) -> None:
        for nid in nids:
            self.dirty.add(nid)
            self.nodes[nid]["status"] = "stale"
            self.emit("node.status", nid, {"status": "stale"})
        if nids:
            self.status = "running"

    def settle(self) -> None:
        if not self.dirty:
            return
        for nid in self.order:
            if nid not in self.dirty or self.nodes[nid]["removed"]:
                continue
            self.dirty.discard(nid)
            ih = self.input_hash(nid)
            existing = next((v for v in self.versions[nid] if v["input_hash"] == ih), None)
            if existing is not None and nid not in self.force:
                changed = existing["id"] != self.nodes[nid]["current_version_id"]
                self.apply(nid, existing, reused=True)
            else:
                out, text, rets = self.compute(nid)
                reason = "retry" if nid in self.force else "recompute" if self.versions[nid] else "initial"
                changed = self.apply(nid, self.new_version(nid, ih, out, text, rets, llm=True, reason=reason),
                                     reused=False)
            self.force.discard(nid)
            self.nodes[nid]["status"] = "done"
            self.emit("node.status", nid, {"status": "done"})
            if changed:
                self.dirty.update(self.children(nid))
            if (self.nodes[nid]["key"] == "r1" and self.steers.get(nid) and not self.followed_up):
                self.followed_up = True  # a critic-like follow-up node joins the graph
                self.add_node("r1_fu", "research", [], before="claims")
                self.edges.append((self.nid("r1_fu"), self.nid("claims")))
                self.dirty.add(self.nid("claims"))
                return self.settle()
        self.status = "done"
        self.emit("run.status", None, {"status": "done"})

    # -- views
    def claims(self) -> list[dict]:
        c = next((n for n in self.nodes.values() if n["type"] == "claims"), None)
        v = self.cur(c["id"]) if c else None
        return [{"id": f"cl-{x['key']}", "key": x["key"], "text": x["text"], "confidence": CONFIDENCE[x["key"]],
                 "status": "active", "history": []} for x in ((v or {}).get("output") or {}).get("claims", [])]

    def snapshot(self) -> dict:
        self.settle()
        return {"run": {"id": self.id, "goal": self.goal, "status": self.status},
                "nodes": [dict(self.nodes[n]) for n in self.order],
                "edges": [{"id": f"{s}>{t}", "run_id": self.id, "source_node_id": s, "target_node_id": t}
                          for s, t in self.edges],
                "versions": {n: self.cur(n) for n in self.order if self.cur(n)},
                "claims": self.claims(), "documents": list(self.docs.values()),
                "last_seq": self.events[-1]["seq"], "activity": {}, "attempts": {}}


class FakeServer:
    def __init__(self) -> None:
        self.runs: dict[str, FakeRun] = {}
        self.seq = 0
        self.calls: list[str] = []

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def new_run(self, goal: str) -> FakeRun:
        r = FakeRun(self, f"run{len(self.runs) + 1}", goal)
        self.runs[r.id] = r
        return r

    def run_of_node(self, nid: str) -> FakeRun:
        return next(r for r in self.runs.values() if nid in r.nodes)

    def handle(self, req: httpx.Request) -> httpx.Response:
        path, m = req.url.path.removeprefix("/api"), req.method
        self.calls.append(f"{m} {path}")
        body = json.loads(req.content) if req.content else {}
        if path == "/health":
            return httpx.Response(200, json={"ok": True})
        if m == "POST" and path == "/runs":
            return httpx.Response(200, json=self.new_run(body["goal"]).snapshot())
        if mm := re.fullmatch(r"/runs/([^/]+)", path):
            return httpx.Response(200, json=self.runs[mm[1]].snapshot())
        if mm := re.fullmatch(r"/runs/([^/]+)/events", path):
            run = self.runs[mm[1]]
            run.settle()
            since = int(req.url.params.get("since", 0))
            return httpx.Response(200, json=[e for e in run.events if e["seq"] > since])
        if mm := re.fullmatch(r"/documents/([^/]+)", path):
            run = next(r for r in self.runs.values() if any(d["id"] == mm[1] for d in r.docs.values()))
            doc = next(d for d in run.docs.values() if d["id"] == mm[1])
            doc["trust"] = body["trust"]
            run.emit("document.updated", None, {"document": dict(doc)})
            for nid in run.order:
                cv = run.cur(nid)
                if cv and any(r["document_id"] == doc["id"] for r in cv["retrievals"]):
                    run.stale([nid, *run.descendants(nid)])
            return httpx.Response(200, json=doc)
        mm = re.fullmatch(r"/nodes/([^/]+)(?:/(\w+))?", path)
        assert mm, path
        nid, action = mm[1], mm[2]
        run = self.run_of_node(nid)
        if action == "versions":
            return httpx.Response(200, json=run.versions[nid])
        if action == "impact":
            a = req.url.params["action"]
            ids = run.descendants(nid) if a in ("edit", "restore") else [nid, *run.descendants(nid)]
            return httpx.Response(200, json={"action": a, "node_ids": ids, "skipped_ids": [], "upper_bound": True})
        if action == "rerun":
            run.force.add(nid)
            run.stale([nid])
            return httpx.Response(202)
        if action == "steer":
            run.steers.setdefault(nid, []).append(body["text"])
            run.stale([nid, *run.descendants(nid)])
            return httpx.Response(202)
        if action == "restore":
            run.nodes[nid]["current_version_id"] = body["version_id"]
            run.emit("node.restored", nid, {"version_id": body["version_id"]})
            run.stale(run.descendants(nid))
            return httpx.Response(202)
        if action == "edit":
            prev = run.cur(nid)["output"]
            from livegraph.agents.claims import parse_edit  # contract check: our edit text parses

            out = parse_edit(body["output_text"], prev)
            v = run.new_version(nid, run.input_hash(nid), out, body["output_text"], [], llm=False, reason="edit")
            if run.apply(nid, v, reused=False):
                run.stale(run.descendants(nid))
            return httpx.Response(200, json=v)
        if m == "DELETE":
            run.nodes[nid]["removed"] = True
            run.nodes[nid]["status"] = "skipped"
            run.emit("node.removed", nid, {})
            run.stale([c for s, c in run.edges if s == nid])
            return httpx.Response(202)
        if m == "PATCH":
            cfg = run.nodes[nid]["config"]
            q = (body.get("question") or "").strip()
            if q:
                cfg["question"] = q
            else:
                cfg.pop("question", None)
            run.emit("node.updated", nid, {"node": dict(run.nodes[nid])})
            run.stale([nid, *run.descendants(nid)])
            return httpx.Response(200, json=run.nodes[nid])
        raise AssertionError(f"unhandled {m} {path}")


def make() -> tuple[FakeServer, FakeRun, LiveGraphClient]:
    srv = FakeServer()
    run = srv.new_run("How big is X?")
    run.snapshot()
    return srv, run, LiveGraphClient("http://fake", transport=httpx.MockTransport(srv.handle))


def snap_ids(run: FakeRun) -> dict[str, str | None]:
    return {n: run.nodes[n]["current_version_id"] for n in run.order if not run.nodes[n]["removed"]}


# ---------------------------------------------------------------- selection


def test_targets_are_deterministic() -> None:
    _, run, _ = make()
    snap = run.snapshot()
    doc, title, n = rc.pick_distrust_doc(snap)
    assert (doc, title, n) == ("run1-d1", "Doc d1", 2)  # C1 (c1a) and C2 (c1b) both rest on d1
    assert rc.pick_research_node(snap)["key"] == "r1"
    text, key = rc.drop_claim_edit(snap)
    assert key == "C2"  # lowest confidence
    assert "**C2**" not in text and "**C1**" in text and "**C3**" in text
    assert rc.reworded_question("What is X?") == "Based on the available evidence, what is X?"
    assert "What about r1?" in rc.steer_text(run.nodes["run1-r1"])


def test_account_counts_and_usage() -> None:
    v_llm = {"id": "v1", "model": "m", "tokens_in": 10, "tokens_out": 2}
    v_edit = {"id": "v2", "model": None, "tokens_in": None, "tokens_out": None}
    ev = [
        {"seq": 5, "type": "node.version", "node_id": "old", "payload": {"version": v_llm, "reused": False},
         "ts": "2026-10-01T00:00:00Z"},
        {"seq": 6, "type": "node.version", "node_id": "a", "payload": {"version": v_llm, "reused": False},
         "ts": "2026-10-01T00:00:01Z"},
        {"seq": 7, "type": "node.version", "node_id": "b", "payload": {"version": {"id": "x"}, "reused": True},
         "ts": "2026-10-01T00:00:02Z"},
        {"seq": 8, "type": "node.version", "node_id": "e", "payload": {"version": v_edit, "reused": False},
         "ts": "2026-10-01T00:00:03Z"},
        {"seq": 9, "type": "node.version", "node_id": "b", "payload": {"version": v_llm, "reused": False},
         "ts": "2026-10-01T00:00:04.5Z"},
        {"seq": 10, "type": "run.status", "node_id": None, "payload": {"status": "done"},
         "ts": "2026-10-01T00:00:05Z"},
    ]
    d = rc.account(ev, after_seq=5)
    assert d.recomputed == ["a", "b"] and d.reused == [] and d.edited == ["e"]
    assert d.usage == {"m": TokenUsage(tokens_in=20, tokens_out=4)}
    assert d.wall_s == 4.0
    assert rc.first_done_seq(ev) == 10
    assert rc.account(ev, after_seq=0, upto_seq=5).recomputed == ["old"]


# ---------------------------------------------------------------- end to end on the fake


async def test_bench_all_mutations_original_baseline() -> None:
    _, run, client = make()
    original = snap_ids(run)
    brief0 = run.cur(run.nid("editor"))["output_text"]
    recs: list[RecomputeRecord] = []
    async with client:
        out = await rc.bench_run(client, rc.Target(run.id, "q1"), list(rc.MUTATIONS), baseline="original_run",
                                 cost_fn=cost, sink=recs.append, timeout=5, poll=0)
    assert out == recs
    by = {r.mutation: r for r in recs}
    assert list(by) == list(rc.MUTATIONS)  # fixed execution order, steer last
    for r in recs:
        assert r.meta["reset_ok"] and r.meta["clean_start"], (r.mutation, r.meta)
        assert r.meta["undo_cost_usd"] == 0.0  # undo reuses versions by hash
        assert r.nodes_total == 8 and r.baseline == "original_run"
        assert r.baseline_cost_usd == cost({"x": TokenUsage(tokens_in=8000, tokens_out=800)})
        assert r.brief_before == brief0
        assert r.cost_usd < r.baseline_cost_usd and r.wall_s > 0

    ident = by["identical"]  # same output -> early cutoff at the claims node
    assert (ident.nodes_recomputed, ident.nodes_reused, ident.nodes_impacted) == (1, 0, 5)
    assert ident.meta["output_unchanged"] is True and ident.brief_after == brief0

    edit = by["edit"]  # claims edited (no LLM) -> its 4 descendants recompute
    assert (edit.nodes_recomputed, edit.nodes_impacted, edit.meta["nodes_edited"]) == (4, 4, 1)
    assert "(C2)" not in edit.brief_after and edit.meta["dropped_claim"] == "C2"
    assert edit.usage == {"fast": TokenUsage(tokens_in=2000, tokens_out=200),
                          "smart": TokenUsage(tokens_in=2000, tokens_out=200)}

    pq = by["patch_question"]
    # r1's finding changes, claims recomputes to the same output -> early cutoff for the 4 below it
    assert pq.target == "r1" and (pq.nodes_recomputed, pq.nodes_reused, pq.nodes_impacted) == (2, 4, 6)
    assert pq.detail == "Based on the available evidence, what about r1?"

    dis = by["distrust"]
    assert dis.target == "run1-d1" and dis.detail == "Doc d1"
    assert dis.nodes_impacted == 7 and dis.nodes_recomputed == 7

    st = by["steer"]  # steer adds a follow-up node; the undo vetoes it
    assert (st.nodes_impacted, st.nodes_recomputed, st.nodes_reused, st.meta["nodes_added"]) == (6, 3, 4, 1)
    assert st.meta["undo_vetoed"] == 1

    assert {k: v for k, v in snap_ids(run).items() if k in original} == original
    assert run.docs["d1"]["trust"] == "normal" and run.nodes[run.nid("r1")]["config"]["question"] == "What about r1?"


async def test_fresh_run_baseline() -> None:
    srv, run, client = make()
    recs: list[RecomputeRecord] = []
    async with client:
        await rc.bench_run(client, rc.Target(run.id, "q1"), ["edit"], baseline="fresh_run", cost_fn=cost,
                           sink=recs.append, timeout=5, poll=0)
    (r,) = recs
    assert len(srv.runs) == 2 and r.baseline == "fresh_run"
    fresh = srv.runs["run2"]
    assert fresh.goal == run.goal and r.meta["baseline_run_id"] == "run2"
    assert r.baseline_brief == fresh.cur(fresh.nid("editor"))["output_text"] != ""
    assert r.baseline_cost_usd > 0 and r.cost_usd / r.baseline_cost_usd == 0.5


async def test_budget_stops_new_mutations() -> None:
    from lgeval.costs import Budget

    _, run, client = make()
    recs: list[RecomputeRecord] = []
    budget = Budget(0.0001)
    async with client:
        await rc.bench_run(client, rc.Target(run.id, "q1"), ["edit", "steer"], baseline="original_run",
                           cost_fn=cost, sink=recs.append, budget=budget, poll=0)
    assert [r.mutation for r in recs] == ["edit"]


# ---------------------------------------------------------------- summary, inputs, CLI


def _rec(m: str, cost_usd: float, wall: float, **kw) -> RecomputeRecord:
    base = {"question_id": "q", "run_id": kw.pop("run_id", "r"), "mutation": m, "target": "t", "wall_s": wall,
            "cost_usd": cost_usd, "nodes_total": 10, "nodes_impacted": 6, "nodes_recomputed": 4,
            "nodes_reused": 2, "baseline": "original_run", "baseline_wall_s": 100.0, "baseline_cost_usd": 0.10,
            "brief_before": "a", "brief_after": "b"}
    return RecomputeRecord(**{**base, **kw})


def test_summarize_shape() -> None:
    recs = [_rec("edit", 0.02, 10), _rec("edit", 0.04, 30, run_id="r2", brief_after="a"),
            _rec("steer", 0.05, 50, meta={"reset_ok": False})]
    s = rc.summarize([recs[0], recs[1].model_dump(mode="json"), recs[2]])
    assert list(s) == ["edit", "steer", "all"]
    e = s["edit"]
    assert e["n"] == 2 and e["n_runs"] == 2 and e["baseline"] == "original_run"
    assert abs(e["cost_ratio_median"] - 0.3) < 1e-9 and abs(e["time_ratio_median"] - 0.2) < 1e-9
    assert abs(e["cost_savings_pct_median"] - 70.0) < 1e-6
    assert e["nodes_recomputed_mean"] == 4 and e["recompute_frac_mean"] == 0.4
    assert e["brief_changed_frac"] == 0.5 and e["clean_frac"] == 1.0
    assert s["steer"]["clean_frac"] == 0.0 and s["all"]["n"] == 3
    assert rc.summarize([]) == {}


def _runs_jsonl(tmp: Path) -> Path:
    rows = [RunRecord(question_id="q1", dataset="custom", system="full", run_id="aaaa1111", status="done"),
            RunRecord(question_id="q2", dataset="custom", system="full", run_id="bbbb2222", status="failed"),
            RunRecord(question_id="q3", dataset="custom", system="no_debate", run_id="cccc3333", status="done")]
    (tmp / "runs.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows))
    return tmp


def test_load_targets(tmp_path: Path) -> None:
    ts = rc.load_targets(_runs_jsonl(tmp_path), ["aaaa1111", "dddd4444"])
    assert [(t.run_id, t.question_id) for t in ts] == [("aaaa1111", "q1"), ("dddd4444", "run-dddd4444")]
    assert ts[0].record is not None


def test_cli_dry_run(tmp_path: Path, capsys) -> None:
    parser = argparse.ArgumentParser()
    rc.register(parser.add_subparsers(dest="cmd"))
    args = parser.parse_args(["recompute", "--from", str(_runs_jsonl(tmp_path)), "--mutations", "edit,identical",
                              "--dry-run", "--out", str(tmp_path / "out")])
    assert args.mutations == ["edit", "identical"]
    assert args.func(args) == 0
    assert "1 runs x 2 mutations" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
