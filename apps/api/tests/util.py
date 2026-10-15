from __future__ import annotations

import asyncio
import io
from typing import Any

import httpx

from livegraph.trace import TOOL_EVENTS, TraceBuilder

DOCS = {
    "solar.txt": "Solar capacity growth was rapid. Global solar capacity growth doubled as panel prices fell. "
    "Solar installations grew in China, Europe and the US.",
    "battery.txt": "Battery cost decline continued. Lithium-ion battery pack costs fell by 90 percent in a decade; "
    "the battery cost decline enabled cheap storage.",
    "noise.txt": "Unrelated text about gardening tomatoes and watering plants in summer.",
}


async def upload_docs(client: httpx.AsyncClient, run_id: str | None = None) -> list[dict]:
    out = []
    for name, body in DOCS.items():
        data = {"run_id": run_id} if run_id else {}
        r = await client.post("/api/documents", files={"file": (name, io.BytesIO(body.encode()), "text/plain")}, data=data)
        assert r.status_code == 200, r.text
        out.append(r.json())
    return out


async def wait_run(client: httpx.AsyncClient, run_id: str, statuses=("done",), timeout: float = 15.0) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout
    snap: dict = {}
    while asyncio.get_event_loop().time() < deadline:
        snap = (await client.get(f"/api/runs/{run_id}")).json()
        if snap["run"]["status"] in statuses:
            # let the scheduler settle (no pending wakeups) and re-read
            await asyncio.sleep(0.05)
            snap2 = (await client.get(f"/api/runs/{run_id}")).json()
            if snap2["run"]["status"] in statuses:
                return snap2
        await asyncio.sleep(0.03)
    raise AssertionError(f"run {run_id} stuck in {snap.get('run', {}).get('status')}: "
                         + str([(n['key'], n['status']) for n in snap.get('nodes', [])]))


async def wait_until(pred, timeout: float = 10.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        v = await pred()
        if v:
            return v
        await asyncio.sleep(0.03)
    raise AssertionError("condition not met")


async def start_run(client: httpx.AsyncClient, goal: str = "Is the energy transition accelerating?") -> str:
    r = await client.post("/api/runs", json={"goal": goal})
    assert r.status_code == 200, r.text
    return r.json()["run"]["id"]


def by_key(snap: dict) -> dict[str, dict]:
    return {n["key"]: n for n in snap["nodes"]}


async def versions(client: httpx.AsyncClient, node_id: str) -> list[dict]:
    r = await client.get(f"/api/nodes/{node_id}/versions")
    assert r.status_code == 200, r.text
    return r.json()


async def events(client: httpx.AsyncClient, run_id: str, since: int = 0) -> list[dict]:
    return (await client.get(f"/api/runs/{run_id}/events", params={"since": since})).json()


TRACE_KEYS = ("call_id", "tool", "agent", "parent_call_id", "ok", "chunk_ids", "node_ids", "summary", "ref", "reads")


def trace_sig(steps: list[dict]) -> list[tuple]:
    return [tuple(tuple(x[k]) if isinstance(x[k], list) else x[k] for k in TRACE_KEYS) for x in steps]


def fold(evs: list[dict]) -> dict[str, Any]:
    """Rebuild state from persisted events only (what the frontend scrubber does).

    Also folds tool activity: ``activity[node_id]`` = steps of the node's current attempt (tool events of a
    stale attempt_id are ignored), and ``traces[version_id]`` = the steps of the attempt that produced it."""
    st: dict[str, Any] = {
        "run_status": None, "nodes": {}, "edges": {}, "versions": {}, "claims": {},
        "attempts": {}, "activity": {}, "traces": {}, "stale_tool_events": 0,
    }
    builders: dict[str, TraceBuilder | None] = {}
    for e in evs:
        t, p, nid = e["type"], e["payload"], e["node_id"]
        if t in TOOL_EVENTS:
            b = builders.get(nid)
            if b is None or p["attempt_id"] != st["attempts"].get(nid):
                st["stale_tool_events"] += 1
                continue
            b.apply(t, p, e["ts"])
            st["activity"][nid] = b.dump()
            continue
        if t == "node.status" and p["status"] == "running":
            st["attempts"][nid] = p.get("attempt_id")
            builders[nid] = TraceBuilder()
            st["activity"][nid] = []
        if t == "node.version":
            b = builders.pop(nid, None)
            if not p["reused"]:
                st["traces"][p["version"]["id"]] = b.dump() if b is not None else []
        if t == "run.status":
            st["run_status"] = p["status"]
        elif t == "node.added":
            st["nodes"][p["node"]["id"]] = dict(p["node"])
        elif t == "edge.added":
            st["edges"][p["edge"]["id"]] = p["edge"]
        elif t == "node.status":
            st["nodes"][nid]["status"] = p["status"]
        elif t == "node.version":
            st["versions"][p["version"]["id"]] = p["version"]
            st["nodes"][nid]["current_version_id"] = p["version"]["id"]
        elif t == "node.restored":
            st["nodes"][nid]["current_version_id"] = p["version_id"]
        elif t == "node.removed":
            st["nodes"][nid]["removed"] = True
        elif t == "claim.upserted":
            st["claims"][p["claim"]["id"]] = dict(p["claim"])
        elif t == "claim.confidence":
            st["claims"][p["claim_id"]]["confidence"] = p["confidence"]
    return st


def assert_fold_matches(snap: dict, evs: list[dict]) -> None:
    seqs = [e["seq"] for e in evs]
    assert seqs == list(range(1, len(seqs) + 1)), "seq must be contiguous and monotonic"
    assert snap["last_seq"] == seqs[-1]
    st = fold(evs)
    assert st["run_status"] == snap["run"]["status"]
    assert set(st["nodes"]) == {n["id"] for n in snap["nodes"]}
    for n in snap["nodes"]:
        f = st["nodes"][n["id"]]
        assert f["status"] == n["status"], (n["key"], f["status"], n["status"])
        assert f["current_version_id"] == n["current_version_id"], n["key"]
        assert f["removed"] == n["removed"], n["key"]
    assert set(st["edges"]) == {e["id"] for e in snap["edges"]}
    assert st["stale_tool_events"] == 0, "tool events of a superseded attempt were persisted"
    for nid, v in snap["versions"].items():
        if v["id"] in st["traces"]:
            assert trace_sig(v["trace"]) == trace_sig(st["traces"][v["id"]]), nid
    for nid, steps in snap.get("activity", {}).items():
        assert trace_sig(steps) == trace_sig(st["activity"].get(nid, [])), nid
    for c in snap["claims"]:
        f = st["claims"][c["id"]]
        assert f["status"] == c["status"], c["key"]
        assert f["confidence"] == c["confidence"], c["key"]
        assert f["text"] == c["text"], c["key"]
