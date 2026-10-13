from __future__ import annotations

import asyncio

from tests.util import (
    assert_fold_matches,
    by_key,
    events,
    start_run,
    upload_docs,
    versions,
    wait_run,
    wait_until,
)

EXPECTED = {"plan", "r1", "r2", "c1", "c2", "claims", "prosecutor", "defender", "judge", "editor"}


async def _done_run(client, fake=None) -> tuple[str, dict]:
    await upload_docs(client)
    run_id = await start_run(client)
    snap = await wait_run(client, run_id)
    return run_id, snap


async def test_health(client):
    r = await client.get("/api/health")
    assert r.json() == {"ok": True}


async def test_full_living_brief_run(client):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    assert set(nodes) == EXPECTED
    assert all(n["status"] == "done" for n in nodes.values())
    assert nodes["c1"]["created_by"] == "planner"
    # every done node has a current version in the snapshot
    for n in snap["nodes"]:
        assert snap["versions"][n["id"]]["id"] == n["current_version_id"]
    # research cites real retrieved chunks; invented ids are dropped
    r1v = snap["versions"][nodes["r1"]["id"]]
    assert r1v["retrievals"], "research version must carry retrievals"
    retrieved = {c["chunk_id"] for c in r1v["retrievals"]}
    cited = {cid for f in r1v["output"]["findings"] for cid in f["chunk_ids"]}
    assert cited and cited <= retrieved
    assert "not-a-real-id" not in str(r1v["output"])
    assert r1v["output_text"].startswith("**Findings**")
    # claims + judge confidence
    claims = {c["key"]: c for c in snap["claims"]}
    assert claims["C1"]["confidence"] == 0.8 and claims["C2"]["confidence"] == 0.3
    assert claims["C1"]["history"][0]["rationale"] == "weighed C1"
    editor = snap["versions"][nodes["editor"]["id"]]
    assert editor["output"]["markdown"].startswith("# Brief")
    assert "C1: Solar capacity grew rapidly. (confidence 0.80)" in editor["output_text"]
    # versions have model + token info
    assert editor["model"] == "fake-editor" and editor["reason"] == "initial"
    evs = await events(client, run_id)
    types = [e["type"] for e in evs]
    for t in ("run.status", "node.added", "edge.added", "node.status", "node.version", "retrieval.done",
              "claim.upserted", "claim.confidence"):
        assert t in types, t
    assert all(e["seq"] is not None for e in evs)
    assert_fold_matches(snap, evs)
    # planner's invalid (cyclic) edge was rejected
    assert not any(e["source_node_id"] == nodes["editor"]["id"] for e in snap["edges"])
    runs = (await client.get("/api/runs")).json()
    assert any(r["id"] == run_id for r in runs)


async def test_edit_then_restore_reuses_versions(client):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    r1 = nodes["r1"]["id"]
    orig_r1 = nodes["r1"]["current_version_id"]
    orig_claims = nodes["claims"]["current_version_id"]

    r = await client.post(f"/api/nodes/{r1}/edit", json={"output_text": "**Findings**\n\n- edited by user"})
    assert r.status_code == 200, r.text
    assert r.json()["reason"] == "edit" and r.json()["version"] == 2
    snap = await wait_run(client, run_id)
    nodes = by_key(snap)
    assert nodes["r1"]["current_version_id"] == r.json()["id"]
    cv = await versions(client, nodes["claims"]["id"])
    assert [v["reason"] for v in cv] == ["initial", "recompute"]
    assert nodes["claims"]["current_version_id"] == cv[-1]["id"] != orig_claims
    jv = await versions(client, nodes["judge"]["id"])
    assert len(jv) == 2
    # edited text flows downstream
    from tests.fakes import FAKE

    assert "edited by user" in FAKE["prompts"]["claims"][-1]

    # restore the original r1 version -> descendants recompute with identical hashes -> reuse
    evs_before = await events(client, run_id)
    r = await client.post(f"/api/nodes/{r1}/restore", json={"version_id": orig_r1})
    assert r.status_code == 202
    snap = await wait_run(client, run_id)
    nodes = by_key(snap)
    assert nodes["r1"]["current_version_id"] == orig_r1
    assert nodes["claims"]["current_version_id"] == orig_claims
    assert len(await versions(client, nodes["claims"]["id"])) == 2  # no new version
    new = (await events(client, run_id))[len(evs_before):]
    assert any(e["type"] == "node.restored" for e in new)
    reused = [e for e in new if e["type"] == "node.version" and e["payload"]["reused"]]
    assert {e["node_id"] for e in reused} >= {nodes["claims"]["id"], nodes["judge"]["id"], nodes["editor"]["id"]}
    assert not [e for e in new if e["type"] == "node.version" and not e["payload"]["reused"]]
    assert_fold_matches(snap, await events(client, run_id))


async def test_steer_done_node_and_rerun(client):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    judge = nodes["judge"]["id"]
    r = await client.post(f"/api/nodes/{judge}/steer", json={"text": "STRIKE C2"})
    assert r.status_code == 202
    snap = await wait_run(client, run_id)
    jv = await versions(client, judge)
    assert jv[-1]["reason"] == "steer" and jv[-1]["steer_text"] == "STRIKE C2"
    claims = {c["key"]: c for c in snap["claims"]}
    assert claims["C2"]["status"] == "struck"
    editor = snap["versions"][by_key(snap)["editor"]["id"]]
    assert "C2" not in editor["output"]["markdown"]
    evs = await events(client, run_id)
    q = [e for e in evs if e["type"] == "steer.queued"]
    c = [e for e in evs if e["type"] == "steer.consumed"]
    assert q and c and q[0]["payload"]["steer_id"] == c[0]["payload"]["steer_id"]

    # rerun forces a new version even with an identical hash
    ed = by_key(snap)["editor"]["id"]
    before = await versions(client, ed)
    assert (await client.post(f"/api/nodes/{ed}/rerun")).status_code == 202
    snap = await wait_run(client, run_id)
    after = await versions(client, ed)
    assert len(after) == len(before) + 1 and after[-1]["reason"] == "retry"
    assert after[-1]["input_hash"] == before[-1]["input_hash"]
    assert_fold_matches(snap, await events(client, run_id))


async def test_steer_running_node_restarts(client, fake):
    fake["slow"]["editor"] = 0.05
    await upload_docs(client)
    run_id = await start_run(client)

    async def editor_running():
        s = (await client.get(f"/api/runs/{run_id}")).json()
        n = by_key(s).get("editor")
        return n if n and n["status"] == "running" else None

    ed = await wait_until(editor_running)
    r = await client.post(f"/api/nodes/{ed['id']}/steer", json={"text": "be brief"})
    assert r.status_code == 202
    fake["slow"]["editor"] = 0
    snap = await wait_run(client, run_id)
    ev = await versions(client, ed["id"])
    assert len(ev) == 1 and ev[0]["reason"] == "steer" and ev[0]["steer_text"] == "be brief"
    assert "be brief" in fake["prompts"]["editor"][-1]
    assert_fold_matches(snap, await events(client, run_id))


async def test_veto_node(client):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    assert (await client.delete(f"/api/nodes/{nodes['prosecutor']['id']}")).status_code == 202
    snap = await wait_run(client, run_id)
    nodes = by_key(snap)
    assert nodes["prosecutor"]["removed"] and nodes["prosecutor"]["status"] == "skipped"
    jv = await versions(client, nodes["judge"]["id"])
    assert len(jv) == 2 and jv[-1]["reason"] == "recompute"
    # veto the only parent of the editor's chain head -> descendants lose all inputs -> skipped
    assert (await client.delete(f"/api/nodes/{nodes['judge']['id']}")).status_code == 202
    snap = await wait_run(client, run_id)
    nodes = by_key(snap)
    assert nodes["editor"]["status"] == "skipped" and not nodes["editor"]["removed"]
    assert_fold_matches(snap, await events(client, run_id))


async def test_distrust_document_recomputes_research(client):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    r1v = snap["versions"][nodes["r1"]["id"]]
    doc_id = r1v["retrievals"][0]["document_id"]
    r = await client.patch(f"/api/documents/{doc_id}", json={"trust": "distrusted"})
    assert r.status_code == 200 and r.json()["trust"] == "distrusted"
    snap = await wait_run(client, run_id)
    nodes = by_key(snap)
    v = await versions(client, nodes["r1"]["id"])
    assert len(v) == 2 and v[-1]["reason"] == "recompute"
    assert all(c["document_id"] != doc_id for c in v[-1]["retrievals"])
    assert v[-1]["input_hash"] != v[0]["input_hash"]
    evs = await events(client, run_id)
    assert any(e["type"] == "document.updated" for e in evs)
    assert_fold_matches(snap, evs)
    # restore trust so other tests see the doc again
    await client.patch(f"/api/documents/{doc_id}", json={"trust": "normal"})
    await wait_run(client, run_id)
    assert (await client.patch(f"/api/documents/{doc_id}", json={"trust": "bogus"})).status_code == 422


async def test_human_input_wait_and_resume(client, fake):
    fake["human_input"] = True
    await upload_docs(client)
    run_id = await start_run(client)
    snap = await wait_run(client, run_id, statuses=("waiting_input",))
    nodes = by_key(snap)
    assert nodes["audience"]["status"] == "waiting_input"
    assert nodes["editor"]["status"] == "pending"
    evs = await events(client, run_id)
    req = [e for e in evs if e["type"] == "input.requested"]
    assert req and req[0]["payload"]["prompt"] == "Who is the brief for?"
    r = await client.post(f"/api/nodes/{nodes['audience']['id']}/input", json={"answer": "policy makers"})
    assert r.status_code == 202
    snap = await wait_run(client, run_id)
    ed = snap["versions"][by_key(snap)["editor"]["id"]]
    assert "Audience: policy makers" in ed["output"]["markdown"]
    assert_fold_matches(snap, await events(client, run_id))


async def test_critic_adds_research_mid_run(client, fake):
    fake["thin"] = {"r1"}
    await upload_docs(client)
    run_id = await start_run(client)
    snap = await wait_run(client, run_id)
    nodes = by_key(snap)
    fu = nodes["r1_fu"]
    assert fu["created_by"] == "critic" and fu["type"] == "research" and fu["status"] == "done"
    edges = {(e["source_node_id"], e["target_node_id"]) for e in snap["edges"]}
    assert (nodes["c1"]["id"], fu["id"]) in edges and (fu["id"], nodes["claims"]["id"]) in edges
    # claims waited for the follow-up: a single version that already includes 3 research inputs
    cv = await versions(client, nodes["claims"]["id"])
    assert len(cv) == 1 and any(c["key"] == "C3" for c in cv[0]["output"]["claims"])
    assert_fold_matches(snap, await events(client, run_id))


async def test_node_failure_marks_failed_and_rerun_recovers(client, fake):
    fake["fail"] = {"defender"}
    await upload_docs(client)
    run_id = await start_run(client)
    snap = await wait_run(client, run_id, statuses=("failed",))
    nodes = by_key(snap)
    assert nodes["defender"]["status"] == "failed"
    assert nodes["prosecutor"]["status"] == "done"  # independent branch continued
    assert nodes["judge"]["status"] == "pending"
    evs = await events(client, run_id)
    assert any(e["type"] == "error" and e["node_id"] == nodes["defender"]["id"] for e in evs)
    fake["fail"] = set()
    await client.post(f"/api/nodes/{nodes['defender']['id']}/rerun")
    snap = await wait_run(client, run_id)
    assert by_key(snap)["editor"]["status"] == "done"


async def test_pause_resume_cancel(client, fake):
    fake["slow"]["research"] = 0.03
    await upload_docs(client)
    run_id = await start_run(client)

    async def research_running():
        s = (await client.get(f"/api/runs/{run_id}")).json()
        return any(n["type"] == "research" and n["status"] == "running" for n in s["nodes"])

    await wait_until(research_running)
    r = await client.post(f"/api/runs/{run_id}/pause")
    assert r.json()["status"] == "paused"
    await asyncio.sleep(0.6)
    snap = (await client.get(f"/api/runs/{run_id}")).json()
    assert snap["run"]["status"] == "paused"
    assert by_key(snap)["claims"]["status"] == "pending"  # nothing new started
    fake["slow"]["research"] = 0
    r = await client.post(f"/api/runs/{run_id}/resume")
    assert r.json()["status"] in ("running", "planning", "done")
    await wait_run(client, run_id)
    r = await client.post(f"/api/runs/{run_id}/cancel")
    assert r.json()["status"] == "cancelled"
    assert (await client.get("/api/runs/00000000-0000-0000-0000-000000000000")).status_code == 404


async def test_startup_recovery_pauses_inflight_runs(client):
    from livegraph.db import base as db_base
    from livegraph.db.models import Node, Run
    from livegraph.engine import engine

    async with db_base.SessionLocal() as s:
        run = Run(goal="crashed", workflow="living_brief", status="running")
        s.add(run)
        await s.commit()
        n = Node(run_id=run.id, key="plan", type="plan", title="Plan", status="running", config={}, created_by="planner")
        s.add(n)
        await s.commit()
        run_id, node_id = run.id, n.id
    await engine.recover()
    snap = (await client.get(f"/api/runs/{run_id}")).json()
    assert snap["run"]["status"] == "paused"
    assert snap["nodes"][0]["status"] == "pending"
    # resume restarts the scheduler and the run completes
    await client.post(f"/api/runs/{run_id}/resume")
    from tests.util import wait_run

    snap = await wait_run(client, str(run_id))
    assert any(x["key"] == "editor" for x in snap["nodes"])
    assert str(node_id) in {x["id"] for x in snap["nodes"]}
