from __future__ import annotations

import json

import httpx
import pytest

from lgeval.client import (
    LiveGraphClient,
    brief_of,
    brief_sentences,
    claim_refs,
    extract_record_parts,
    usage_of,
)
from lgeval.schema import TokenUsage

RUN = "11111111-1111-1111-1111-111111111111"
CH1, CH2, CH3, CH_OLD = (f"aaaaaaaa-0000-0000-0000-00000000000{i}" for i in (1, 2, 3, 9))


def node(nid: str, key: str, type_: str, status: str = "done", removed: bool = False) -> dict:
    return {"id": nid, "key": key, "type": type_, "status": status, "removed": removed, "title": key}


def chunk(cid: str, text: str) -> dict:
    return {"chunk_id": cid, "document_id": "d", "document_title": f"Doc {cid[-1]}", "source_url": f"https://x/{cid[-1]}",
            "text": text, "score": 0.9, "trust": "normal", "rank": 1}


BRIEF = """# Brief

Bottom line: fasting works about as well as restriction (C1). Adherence is similar (C2, C3).

## Details
- **Weight loss** is comparable at 12 months. (C1)
- No claim cited here.
"""


def make_snapshot(status: str = "done", last_seq: int = 10, node_status: str = "done") -> dict:
    return {
        "run": {"id": RUN, "status": status, "goal": "g"},
        "nodes": [
            node("n-research", "research_1", "research"),
            node("n-claims", "claims", "claims"),
            node("n-editor", "editor", "editor", status=node_status),
            node("n-gone", "editor_old", "editor", status="skipped", removed=True),
        ],
        "versions": {
            "n-research": {"id": "v-r", "output_text": "...", "retrievals": [chunk(CH1, "one"), chunk(CH2, "two")]},
            "n-claims": {
                "id": "v-c",
                "output": {"claims": [
                    {"key": "C1", "text": "Fasting ~ restriction", "support_chunk_ids": [CH1]},
                    {"key": "C2", "text": "Adherence similar", "support_chunk_ids": [CH2, CH_OLD]},
                    {"key": "C3", "text": "No support", "support_chunk_ids": []},
                ]},
                "retrievals": [],
            },
            "n-editor": {"id": "v-e", "output": {"markdown": BRIEF}, "output_text": BRIEF, "retrievals": []},
            "n-gone": {"id": "v-g", "output_text": "stale brief"},
        },
        "claims": [
            {"key": "C1", "text": "Fasting ~ restriction", "confidence": 0.8, "status": "active"},
            {"key": "C2", "text": "Adherence similar", "confidence": 0.5, "status": "active"},
            {"key": "C3", "text": "No support", "confidence": 0.1, "status": "struck"},
            {"key": "C4", "text": "Late claim", "confidence": None, "status": "active"},
        ],
        "last_seq": last_seq,
    }


def test_brief_of() -> None:
    assert brief_of(make_snapshot()) == BRIEF
    assert brief_of({"nodes": [], "versions": {}}) == ""


def test_claim_refs_and_sentences() -> None:
    assert claim_refs("x (C2, C3) and (C1) again (C2)") == ["C2", "C3", "C1"]
    assert claim_refs("C1 without parens") == []
    sents = brief_sentences(BRIEF)
    assert sents == [
        "Bottom line: fasting works about as well as restriction (C1).",
        "Adherence is similar (C2, C3).",
        "Weight loss is comparable at 12 months. (C1)",
        "No claim cited here.",
    ]


def test_extract_record_parts() -> None:
    snap = make_snapshot()
    versions = {"n-research": [{"id": "v-r0", "retrievals": [chunk(CH_OLD, "old"), chunk(CH3, "unused")]}]}
    claims, citations, evidence = extract_record_parts(snap, versions)
    assert [c.key for c in claims] == ["C1", "C2", "C3", "C4"]
    assert claims[0].confidence == 0.8 and claims[2].status == "struck"
    assert claims[3].support_chunk_ids == []
    assert citations[1].claim_keys == ["C2", "C3"] and citations[1].chunk_ids == [CH2, CH_OLD]
    assert citations[2].chunk_ids == [CH1]
    assert citations[3].claim_keys == [] and citations[3].chunk_ids == []
    # current retrievals + a claim-cited chunk only an old version saw; never unrelated old chunks
    assert [e.chunk_id for e in evidence] == [CH1, CH2, CH_OLD]
    assert evidence[2].text == "old" and evidence[0].url == "https://x/1"


def test_usage_of() -> None:
    versions = {
        "a": [{"id": "1", "model": "m1", "tokens_in": 10, "tokens_out": 2},
              {"id": "2", "model": "m1", "tokens_in": 5, "tokens_out": 1}],
        "b": [{"id": "3", "model": "m2", "tokens_in": 7, "tokens_out": 3},
              {"id": "4", "model": None, "tokens_in": None, "tokens_out": None}],
    }
    assert usage_of(versions) == {"m1": TokenUsage(tokens_in=15, tokens_out=3), "m2": TokenUsage(tokens_in=7, tokens_out=3)}
    assert usage_of(versions, after_seq_versions={"2", "4"}) == {"m1": TokenUsage(tokens_in=5, tokens_out=1)}


def recording_transport(responses: dict[tuple[str, str], object]) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        body = responses.get((req.method, req.url.path), {"ok": True})
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler), seen


async def test_client_routes() -> None:
    transport, seen = recording_transport({("GET", "/api/nodes/n1/versions"): [{"id": "v"}]})
    async with LiveGraphClient("http://lg", transport=transport) as c:
        assert await c.health()
        await c.create_run("goal", ["d1"])
        assert await c.versions("n1") == [{"id": "v"}]
        await c.events(RUN, since=5)
        await c.impact("n1", "steer")
        await c.steer("n1", "be brief")
        await c.edit("n1", "new text")
        await c.rerun("n1")
        await c.restore("n1", "v1")
        await c.veto("n1")
        await c.patch_node("n1", question="q?")
        await c.set_trust("d1", "distrusted")
        await c.answer("n1", "yes")
        await c.cancel(RUN)
        await c.upload("a.txt", b"hello", run_id=RUN)
    got = [(r.method, r.url.path) for r in seen]
    assert got == [
        ("GET", "/api/health"), ("POST", "/api/runs"), ("GET", "/api/nodes/n1/versions"),
        ("GET", f"/api/runs/{RUN}/events"), ("GET", "/api/nodes/n1/impact"), ("POST", "/api/nodes/n1/steer"),
        ("POST", "/api/nodes/n1/edit"), ("POST", "/api/nodes/n1/rerun"), ("POST", "/api/nodes/n1/restore"),
        ("DELETE", "/api/nodes/n1"), ("PATCH", "/api/nodes/n1"), ("PATCH", "/api/documents/d1"),
        ("POST", "/api/nodes/n1/input"), ("POST", f"/api/runs/{RUN}/cancel"), ("POST", "/api/documents"),
    ]
    assert json.loads(seen[1].content) == {"goal": "goal", "document_ids": ["d1"]}
    assert seen[3].url.params["since"] == "5" and seen[4].url.params["action"] == "steer"
    assert json.loads(seen[8].content) == {"version_id": "v1"}
    assert json.loads(seen[10].content) == {"question": "q?"}
    assert b"hello" in seen[14].content and RUN.encode() in seen[14].content


async def test_client_raises_on_error() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(404, json={"detail": "nope"}))
    async with LiveGraphClient("http://lg", transport=transport) as c:
        assert not await c.health()
        with pytest.raises(httpx.HTTPStatusError):
            await c.snapshot(RUN)


def snapshot_sequence(snaps: list[dict]) -> tuple[httpx.MockTransport, list[int]]:
    calls = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        i = min(calls[0], len(snaps) - 1)
        calls[0] += 1
        return httpx.Response(200, json=snaps[i])

    return httpx.MockTransport(handler), calls


async def test_wait_settled_waits_for_quiet_and_stable_seq() -> None:
    snaps = [
        make_snapshot(status="running", last_seq=3),
        make_snapshot(status="done", last_seq=8, node_status="running"),
        make_snapshot(status="done", last_seq=9),
        make_snapshot(status="done", last_seq=9),
    ]
    transport, calls = snapshot_sequence(snaps)
    async with LiveGraphClient("http://lg", transport=transport) as c:
        snap = await c.wait_settled(RUN, poll=0, timeout=5)
    assert snap["last_seq"] == 9 and calls[0] == 4


async def test_wait_settled_since_seq_requires_new_events() -> None:
    snaps = [make_snapshot(last_seq=9)] * 3 + [make_snapshot(status="running", last_seq=12)] + [make_snapshot(last_seq=14)] * 2
    transport, calls = snapshot_sequence(snaps)
    async with LiveGraphClient("http://lg", transport=transport) as c:
        snap = await c.wait_settled(RUN, poll=0.001, timeout=5, since_seq=9)
    assert snap["last_seq"] == 14 and calls[0] == 6


async def test_wait_settled_timeout() -> None:
    transport, _ = snapshot_sequence([make_snapshot(status="running")])
    async with LiveGraphClient("http://lg", transport=transport) as c:
        with pytest.raises(TimeoutError):
            await c.wait_settled(RUN, poll=0.01, timeout=0.05)


async def test_all_versions() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        nid = req.url.path.split("/")[3]
        return httpx.Response(200, json=[{"id": f"v-{nid}"}])

    async with LiveGraphClient("http://lg", transport=httpx.MockTransport(handler)) as c:
        got = await c.all_versions(make_snapshot())
    assert got["n-claims"] == [{"id": "v-n-claims"}] and len(got) == 4


def test_table_rows_are_units_with_bare_claim_keys() -> None:
    md = "| # | Finding | Conf |\n|---|---|:-:|\n| C2 | **Adherence** is similar. Really. | 0.5 |\n"
    assert brief_sentences(md) == ["(C2) | Adherence is similar. Really. | 0.5"]
    assert claim_refs(brief_sentences(md)[0]) == ["C2"]


def test_sentence_split_keeps_abbreviations() -> None:
    md = "Fan et al. found gains, e.g. less burnout (C1). The U.S. data differ. Next one."
    assert brief_sentences(md) == ["Fan et al. found gains, e.g. less burnout (C1).", "The U.S. data differ.", "Next one."]


def test_is_quiet_ignores_pending_below_failure_or_gate() -> None:
    from lgeval.client import is_quiet

    def snap(run: str, *statuses: str) -> dict:
        return {"run": {"status": run}, "nodes": [{"status": s} for s in statuses]}

    assert is_quiet(snap("failed", "done", "failed", "pending"))
    assert is_quiet(snap("waiting_input", "done", "waiting_input", "pending"))
    assert not is_quiet(snap("failed", "running", "pending"))
    assert not is_quiet(snap("done", "done", "pending"))
    assert not is_quiet(snap("running", "done"))
