"""Fast traversal: early cutoff (output_hash), impact preview, PATCH /nodes/{id}, request_upstream_work."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from livegraph import rag
from livegraph.agents.tools.graph_tools import _top_level_only
from livegraph.config import settings
from livegraph.db import base as db_base
from livegraph.db.models import Node, SteerMessage
from livegraph.engine.hashing import output_hash
from tests.test_api_agents import FakeWeb
from tests.util import assert_fold_matches, by_key, events, start_run, upload_docs, versions, wait_run


async def _done_run(client) -> tuple[str, dict]:
    await upload_docs(client)
    run_id = await start_run(client)
    return run_id, await wait_run(client, run_id)


async def _version_counts(client, snap) -> dict[str, int]:
    return {n["key"]: len(await versions(client, n["id"])) for n in snap["nodes"]}


def _current(snap) -> dict[str, str | None]:
    return {n["key"]: n["current_version_id"] for n in snap["nodes"]}


def _prompt_counts(fake) -> dict[str, int]:
    return {t: len(p) for t, p in fake["prompts"].items()}


async def _agent_steers(run_id: str) -> list[tuple[str, str, str]]:
    async with db_base.SessionLocal() as s:
        rows = (
            await s.execute(
                select(Node.key, SteerMessage.source, SteerMessage.text)
                .join(Node, Node.id == SteerMessage.node_id)
                .where(Node.run_id == run_id, SteerMessage.source.like("agent:%"))
                .order_by(SteerMessage.created_at)
            )
        ).all()
    return [tuple(r) for r in rows]


def _steps(v: dict, tool: str) -> list[dict]:
    return [s for s in v["trace"] if s["tool"] == tool]


# ---------------------------------------------------------------- output hash


def test_output_hash_covers_output_text_and_retrievals():
    base = output_hash({"a": 1}, "Hello  world\n", [("c1", True), ("c2", False)])
    assert base == output_hash({"a": 1}, "Hello world", [("c2", False), ("c1", True)])  # normalized, sorted
    assert base != output_hash({"a": 2}, "Hello world", [("c1", True), ("c2", False)])
    assert base != output_hash({"a": 1}, "Hello there", [("c1", True), ("c2", False)])
    assert base != output_hash({"a": 1}, "Hello world", [("c1", True), ("c2", True)])  # cited flag
    assert base != output_hash({"a": 1}, "Hello world", [("c1", True)])  # evidence seen


# ---------------------------------------------------------------- impact preview


async def test_impact_preview_is_topological_and_read_only(client):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    ids = {n["id"]: n["key"] for n in snap["nodes"]}
    n_events = len(await events(client, run_id))

    r = await client.get(f"/api/nodes/{nodes['r1']['id']}/impact", params={"action": "steer"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["upper_bound"] is True and body["action"] == "steer" and body["skipped_ids"] == []
    keys = [ids[i] for i in body["node_ids"]]
    assert keys[0] == "r1"
    assert set(keys) == {"r1", "c1", "claims", "prosecutor", "defender", "judge", "editor"}
    assert "r2" not in keys and "c2" not in keys and "plan" not in keys  # unrelated branch / upstream
    pos = {k: i for i, k in enumerate(keys)}
    for e in snap["edges"]:
        s, t = ids[e["source_node_id"]], ids[e["target_node_id"]]
        if s in pos and t in pos:
            assert pos[s] < pos[t], (s, t)

    edit = (await client.get(f"/api/nodes/{nodes['r1']['id']}/impact", params={"action": "edit"})).json()
    assert [ids[i] for i in edit["node_ids"]] == keys[1:]  # the edited node itself does not rerun

    veto = (await client.get(f"/api/nodes/{nodes['judge']['id']}/impact", params={"action": "veto"})).json()
    assert veto["node_ids"] == [] and [ids[i] for i in veto["skipped_ids"]] == ["editor"]
    veto = (await client.get(f"/api/nodes/{nodes['prosecutor']['id']}/impact", params={"action": "veto"})).json()
    assert [ids[i] for i in veto["node_ids"]] == ["judge", "editor"] and veto["skipped_ids"] == []

    assert (await client.get(f"/api/nodes/{nodes['r1']['id']}/impact", params={"action": "bogus"})).status_code == 422
    missing = "00000000-0000-0000-0000-000000000000"
    assert (await client.get(f"/api/nodes/{missing}/impact", params={"action": "steer"})).status_code == 404
    # read-only: no events, no state change
    assert len(await events(client, run_id)) == n_events
    assert (await client.get(f"/api/runs/{run_id}")).json()["nodes"] == snap["nodes"]


# ---------------------------------------------------------------- early cutoff


async def test_identical_rerun_cuts_off(client, fake):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    before_counts, before_current = await _version_counts(client, snap), _current(snap)
    prompts = _prompt_counts(fake)
    n_events = len(await events(client, run_id))

    assert (await client.post(f"/api/nodes/{nodes['r1']['id']}/rerun")).status_code == 202
    snap = await wait_run(client, run_id)
    rv = await versions(client, nodes["r1"]["id"])
    assert len(rv) == 2 and rv[-1]["reason"] == "retry"
    assert rv[0]["output_hash"] and rv[-1]["output_hash"] == rv[0]["output_hash"]
    after_counts, after_current = await _version_counts(client, snap), _current(snap)
    assert after_counts == {**before_counts, "r1": 2}  # no new downstream versions
    assert {k: v for k, v in after_current.items() if k != "r1"} == {k: v for k, v in before_current.items() if k != "r1"}
    # only the research agent was called again
    now = _prompt_counts(fake)
    assert {t for t in now if now[t] != prompts.get(t, 0)} == {"research"}
    new = (await events(client, run_id))[n_events:]
    assert not [e for e in new if e["type"] == "node.status" and e["payload"]["status"] == "stale"
                and e["node_id"] != nodes["r1"]["id"]]
    assert_fold_matches(snap, await events(client, run_id))


async def test_evidence_change_still_recomputes(client, fake):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    claims_before = len(await versions(client, nodes["claims"]["id"]))
    # same question, but the rerun also reads other chunks -> different retrievals -> different content
    fake["scripts"]["research"] = lambda text, rnd: (
        [("search_documents", {"query": "gardening tomatoes watering", "k": 3})] if rnd == 0 else None
    )
    assert (await client.post(f"/api/nodes/{nodes['r1']['id']}/rerun")).status_code == 202
    snap = await wait_run(client, run_id)
    rv = await versions(client, nodes["r1"]["id"])
    assert rv[-1]["output_hash"] != rv[0]["output_hash"]
    cv = await versions(client, nodes["claims"]["id"])
    assert len(cv) == claims_before + 1 and cv[-1]["reason"] == "recompute"
    assert_fold_matches(snap, await events(client, run_id))


async def test_grandparent_change_behind_unchanged_parent_recomputes_context_readers(client, fake):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    counts = await _version_counts(client, snap)
    r = await client.post(f"/api/nodes/{nodes['plan']['id']}/edit", json={"output_text": "**Sub-questions**\n\n1. edited"})
    assert r.status_code == 200, r.text
    snap = await wait_run(client, run_id)
    nodes = by_key(snap)
    after = await _version_counts(client, snap)
    # research recomputed (its parent changed) but came out identical: content cutoff
    rv = await versions(client, nodes["r1"]["id"])
    assert len(rv) == 2 and rv[-1]["output_hash"] == rv[0]["output_hash"]
    # claims cannot read the plan: cut off and back to its current version
    assert after["claims"] == counts["claims"]
    # the judge can read the plan through its context tools (a grandparent behind unchanged parents)
    assert after["judge"] == counts["judge"] + 1
    assert after["prosecutor"] == counts["prosecutor"] + 1
    # the editor only sees the judge (unchanged content) and claims: cut off
    assert after["editor"] == counts["editor"]
    evs = await events(client, run_id)
    assert any(e["type"] == "node.status" and e["node_id"] == nodes["claims"]["id"]
               and e["payload"].get("reason") == "upstream unchanged" for e in evs)
    assert_fold_matches(snap, evs)


# ---------------------------------------------------------------- PATCH /nodes/{id}


async def test_patch_question_recomputes_live_then_patch_back_reuses(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "web_search", True)
    web = FakeWeb()
    rag.set_web_client(web)
    try:
        run_id, snap = await _done_run(client)
        nodes = by_key(snap)
        r1 = nodes["r1"]["id"]
        orig = _current(snap)
        web_before = len(web.calls)

        r = await client.patch(f"/api/nodes/{r1}", json={"question": "solar capacity growth in 2024"})
        assert r.status_code == 200, r.text
        assert r.json()["config"]["question"] == "solar capacity growth in 2024"
        snap = await wait_run(client, run_id)
        nodes = by_key(snap)
        rv = await versions(client, r1)
        assert len(rv) == 2 and "solar capacity growth in 2024" in rv[-1]["output_text"]
        assert "solar capacity growth in 2024" in web.calls[web_before:]  # one live web pass for the new question
        assert nodes["claims"]["current_version_id"] != orig["claims"]
        evs = await events(client, run_id)
        upd = [e for e in evs if e["type"] == "node.updated"]
        assert len(upd) == 1 and upd[0]["node_id"] == r1
        assert upd[0]["payload"]["node"]["config"]["question"] == "solar capacity growth in 2024"
        assert_fold_matches(snap, evs)

        web_mid = len(web.calls)
        r = await client.patch(f"/api/nodes/{r1}", json={"question": "solar capacity growth"})
        assert r.status_code == 200
        snap = await wait_run(client, run_id)
        assert _current(snap) == orig  # everything back on its original version
        assert len(await versions(client, r1)) == 2  # reused, no new version
        assert len(web.calls) == web_mid  # and no network
    finally:
        rag.set_web_client(None)

    assert (await client.patch(f"/api/nodes/{r1}", json={})).status_code == 422
    claims = by_key(snap)["claims"]["id"]
    assert (await client.patch(f"/api/nodes/{claims}", json={"question": "x"})).status_code == 422
    # instructions/title on any node; no-op patch changes nothing
    n_events = len(await events(client, run_id))
    r = await client.patch(f"/api/nodes/{claims}", json={"title": by_key(snap)["claims"]["title"]})
    assert r.status_code == 200 and len(await events(client, run_id)) == n_events


# ---------------------------------------------------------------- request_upstream_work


async def test_upstream_tool_hidden_in_nested_runs():
    tool_def = object()
    nested = SimpleNamespace(deps=SimpleNamespace(parent_call_id="abc", sched=object()))
    top = SimpleNamespace(deps=SimpleNamespace(parent_call_id=None, sched=object()))
    assert await _top_level_only(nested, tool_def) is None
    assert await _top_level_only(top, tool_def) is tool_def


async def test_upstream_request_flow_and_settles(client, fake):
    fake["scripts"]["judge"] = lambda text, rnd: (
        [("request_upstream_work", {"target_node_key": "r1", "instruction": "Look at the 2024 figures"})]
        if rnd == 0 else None
    )
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    assert snap["run"]["status"] == "done"
    assert await _agent_steers(run_id) == [("r1", "agent:judge", "Request from judge: Look at the 2024 figures")]
    rv = await versions(client, nodes["r1"]["id"])
    assert len(rv) == 2 and rv[-1]["reason"] == "steer"
    assert rv[-1]["steer_text"] == "Request from judge: Look at the 2024 figures"
    jv = await versions(client, nodes["judge"]["id"])
    assert len(jv) == 2  # reran once on the new research, then settled
    first, second = _steps(jv[0], "request_upstream_work"), _steps(jv[1], "request_upstream_work")
    assert first[0]["ok"] and first[0]["node_ids"] == [nodes["r1"]["id"]]
    assert not second[0]["ok"] and second[0]["summary"] == "already requested"
    evs = await events(client, run_id)
    q = [e for e in evs if e["type"] == "steer.queued"]
    assert len(q) == 1 and q[0]["node_id"] == nodes["r1"]["id"] and q[0]["payload"]["source"] == "agent:judge"
    c = [e for e in evs if e["type"] == "steer.consumed"]
    assert c and c[0]["payload"]["source"] == "agent:judge"
    assert_fold_matches(snap, evs)


async def test_upstream_request_rejections_and_cap(client, fake):
    fake["scripts"]["prosecutor"] = lambda text, rnd: (
        [
            ("request_upstream_work", {"target_node_key": "defender", "instruction": "more"}),  # sibling
            ("request_upstream_work", {"target_node_key": "plan", "instruction": "replan"}),  # forbidden
            ("request_upstream_work", {"target_node_key": "r1", "instruction": "x" * 501}),  # too long
            ("request_upstream_work", {"target_node_key": "prosecutor", "instruction": "me"}),  # own node
        ]
        if rnd == 0 else None
    )
    fake["scripts"]["judge"] = lambda text, rnd: (
        [
            ("request_upstream_work", {"target_node_key": k, "instruction": f"redo {k}"})
            for k in ("r1", "r2", "claims", "prosecutor")
        ]
        if rnd == 0 else None
    )
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    assert snap["run"]["status"] == "done"  # settled
    steers = await _agent_steers(run_id)
    assert sorted(k for k, _, _ in steers) == ["claims", "r1", "r2"]
    assert len(steers) == settings.max_upstream_requests == 3
    assert all(src == "agent:judge" for _, src, _ in steers)

    pv = await versions(client, nodes["prosecutor"]["id"])
    summaries = [s["summary"] for s in _steps(pv[0], "request_upstream_work")]
    assert summaries == ["not upstream", "not requestable", "instruction too long", "own node"]
    assert not any(s["ok"] for s in _steps(pv[0], "request_upstream_work"))

    jv = await versions(client, nodes["judge"]["id"])
    first = _steps(jv[0], "request_upstream_work")
    assert [s["ok"] for s in first] == [True, True, True, False]
    assert first[-1]["summary"] == "request limit reached"
    assert all(not s["ok"] for v in jv[1:] for s in _steps(v, "request_upstream_work"))
    assert_fold_matches(snap, await events(client, run_id))


@pytest.fixture(autouse=True)
def _cap_default(monkeypatch):
    monkeypatch.setattr(settings, "max_upstream_requests", 3)
