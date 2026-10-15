"""Multi-agent behaviour: tool loops, traces, budgets, supersession, validators, agent-as-tool, web cache."""

from __future__ import annotations

import asyncio
import re
import uuid

import pytest
from openai.types.chat import ChatCompletion

from livegraph import rag
from livegraph.agents import research as research_mod
from livegraph.agents.deps import NodeDeps
from livegraph.agents.runner import run_agent
from livegraph.config import settings
from tests import fakes
from tests.util import assert_fold_matches, by_key, events, start_run, upload_docs, versions, wait_run, wait_until


async def _done_run(client) -> tuple[str, dict]:
    await upload_docs(client)
    run_id = await start_run(client)
    return run_id, await wait_run(client, run_id)


def _node_events(evs: list[dict], node_id: str, *types: str) -> list[dict]:
    return [e for e in evs if e["node_id"] == node_id and e["type"] in types]


class FakeWeb:
    """OpenAI-compatible client for rag.web: one cited page per query (url derived from the query)."""

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.calls: list[str] = []
        self.gate = gate
        self.chat = type("C", (), {"completions": type("X", (), {"create": self._create})()})()

    async def _create(self, **kwargs):
        q = kwargs["messages"][-1]["content"]
        self.calls.append(q)
        if self.gate is not None:
            await self.gate.wait()
        slug = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")
        return ChatCompletion.model_validate({
            "id": "gen", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": f"About {q}.",
                "annotations": [{"type": "url_citation", "url_citation": {
                    "url": f"https://web.test/{slug}", "title": f"Web: {q}",
                    "content": f"Web page on {q}: {q} matters a lot, with figures about {q}.", "start_index": 0, "end_index": 5,
                }}],
            }}],
        })


@pytest.fixture
def web(monkeypatch):
    monkeypatch.setattr(settings, "web_search", True)
    client = FakeWeb()
    rag.set_web_client(client)
    yield client
    rag.set_web_client(None)


# ---------------------------------------------------------------- research tool loop + trace


async def test_research_tool_loop_trace_and_retrievals(client):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    r1 = nodes["r1"]["id"]
    v = snap["versions"][r1]
    trace = v["trace"]
    assert {s["tool"] for s in trace[:2]} == {"web_search", "search_documents"}  # one parallel batch
    assert trace[2]["tool"] == "read_source"
    assert all(s["ok"] and s["agent"] == "research" and s["parent_call_id"] is None for s in trace)

    evs = await events(client, run_id)
    attempt = [e for e in _node_events(evs, r1, "node.status") if e["payload"]["status"] == "running"][-1]
    attempt_id = attempt["payload"]["attempt_id"]
    calls = [e for e in _node_events(evs, r1, "tool.call") if e["payload"]["attempt_id"] == attempt_id]
    assert [e["payload"]["call_id"] for e in calls] == [s["call_id"] for s in trace]
    ws = next(e["payload"] for e in calls if e["payload"]["tool"] == "web_search")
    assert ws["step"] == 1 and ws["budget"] == settings.web_search_max_per_node and ws["args_summary"]
    results = {e["payload"]["call_id"]: e["payload"] for e in _node_events(evs, r1, "tool.result")}
    assert set(results) == {s["call_id"] for s in trace}
    assert all(isinstance(p["ms"], int) and p["error"] is None for p in results.values())

    # retrievals = seed chunks + every chunk a tool returned; cited chunks rank first
    rets = v["retrievals"]
    retrieved = {c["chunk_id"] for c in rets}
    tool_chunks = {c for s in trace for c in s["chunk_ids"]}
    seeds = {c["chunk_id"] for c in rets if c["via"] == "seed"}
    assert seeds and retrieved == seeds | tool_chunks
    cited = {c for f in v["output"]["findings"] for c in f["chunk_ids"]}
    assert cited and {c["chunk_id"] for c in rets if c["cited"]} == cited
    first_uncited = min((c["rank"] for c in rets if not c["cited"]), default=10**6)
    assert all(c["rank"] < first_uncited for c in rets if c["cited"])
    assert_fold_matches(snap, evs)


async def test_budget_exhaustion_finishes_gracefully(client, fake):
    seen: list[str] = []

    def script(text: str, rnd: int):
        seen.append(text)
        return [("web_search", {"query": f"angle {rnd}"})] if rnd < 3 else None

    fake["scripts"]["research"] = script
    run_id, snap = await _done_run(client)
    trace = snap["versions"][by_key(snap)["r1"]["id"]]["trace"]
    assert [s["tool"] for s in trace] == ["web_search"] * 3
    assert trace[0]["ok"] and trace[1]["ok"]
    assert not trace[2]["ok"] and trace[2]["summary"] == "budget exhausted"
    evs = await events(client, run_id)
    last = next(e for e in evs if e["type"] == "tool.result" and e["payload"]["call_id"] == trace[2]["call_id"])
    assert last["payload"]["error"] == "budget exhausted"
    assert by_key(snap)["r1"]["status"] == "done"
    # the model saw the soft-budget message (and then finalized)
    assert any("BUDGET EXHAUSTED for web_search" in t for t in seen)
    assert_fold_matches(snap, evs)


# ---------------------------------------------------------------- supersession mid-tool + snapshot activity


async def test_steer_mid_tool_drops_stale_attempt(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "web_search", True)
    gate = asyncio.Event()
    web = FakeWeb(gate)
    rag.set_web_client(web)
    try:
        await upload_docs(client)
        run_id = await start_run(client)

        async def r1_in_web_search():
            s = (await client.get(f"/api/runs/{run_id}")).json()
            n = by_key(s).get("r1")
            if not n or n["status"] != "running":
                return None
            steps = s["activity"].get(n["id"], [])
            return (s, n) if any(x["tool"] == "web_search" and x["ok"] is None for x in steps) else None

        snap, r1 = await wait_until(r1_in_web_search)
        old_attempt = snap["attempts"][r1["id"]]
        assert old_attempt
        stuck = next(x for x in snap["activity"][r1["id"]] if x["tool"] == "web_search")

        assert (await client.post(f"/api/nodes/{r1['id']}/steer", json={"text": "focus on 2024"})).status_code == 202
        gate.set()
        snap = await wait_run(client, run_id)
    finally:
        rag.set_web_client(None)

    evs = await events(client, run_id)
    running = [e for e in _node_events(evs, r1["id"], "node.status") if e["payload"]["status"] == "running"]
    attempts = [e["payload"]["attempt_id"] for e in running]
    assert len(attempts) == 2 and attempts[0] == old_attempt != attempts[1]
    # the cancelled call never got a result, and nothing of the old attempt was emitted after the steer
    assert not [e for e in evs if e["type"] == "tool.result" and e["payload"]["call_id"] == stuck["call_id"]]
    steer_seq = next(e["seq"] for e in evs if e["type"] == "steer.queued")
    assert not [e for e in evs if e["seq"] > steer_seq and e["type"].startswith("tool.")
                and e["payload"]["attempt_id"] == old_attempt]
    v = snap["versions"][r1["id"]]
    assert v["reason"] == "steer" and stuck["call_id"] not in {s["call_id"] for s in v["trace"]}
    assert snap["activity"] == {} and snap["attempts"] == {}
    assert_fold_matches(snap, evs)


# ---------------------------------------------------------------- validator retry -> delta reset


async def test_validator_retry_resets_live_buffer(db):
    deltas: list[tuple[str, str]] = []
    model = fakes.scripted_model("research", None, fakes._research)
    real = uuid.uuid4()
    prompt = f"## Research question\nq\n\n<source chunk_id={real} trust=normal>a</source>\n" \
             f"<source chunk_id={uuid.uuid4()} trust=normal>b</source>\n<source chunk_id={uuid.uuid4()} trust=normal>c</source>"
    ids = re.findall(r"chunk_id=([0-9a-f-]{36})", prompt)
    deps = NodeDeps(run_id=uuid.uuid4(), node_id=uuid.uuid4(), node_key="r", node_type="research", sched=None,
                    attempt_id="a", me=None, cited_ok=set(ids))
    result = await run_agent(
        research_mod.agents.get(), prompt, model=model, render=research_mod.render, deps=deps,
        on_delta=lambda t: deltas.append(("delta", t)), on_reset=lambda t: deltas.append(("reset", t)),
    )
    assert ("reset", "") in deltas  # the bounced draft was withdrawn
    buf = ""
    for kind, t in deltas:
        buf = t if kind == "reset" else buf + t
    assert buf == result.output_text
    assert "not-a-real-id" not in str(result.output.model_dump())


# ---------------------------------------------------------------- judge: agent-as-tool rebuttal


async def test_judge_rebuttal_agent_as_tool(client, fake):
    fake["scripts"]["judge"] = lambda text, rnd: (
        [("ask_rebuttal", {"claim_key": "C1", "side": "prosecutor", "question": "Which source undermines C1?"})]
        if rnd == 0 else None
    )
    run_id, snap = await _done_run(client)
    jid = by_key(snap)["judge"]["id"]
    v = snap["versions"][jid]
    assert [r["claim_key"] for r in v["output"]["rebuttals"]] == ["C1"]
    assert v["output"]["rebuttals"][0]["side"] == "prosecutor" and "overreaches" in v["output"]["rebuttals"][0]["answer"]
    assert "**Rebuttals**" in v["output_text"] and v["output_text"].startswith("- **C1**")
    ask = next(s for s in v["trace"] if s["tool"] == "ask_rebuttal")
    assert [(m["kind"], m["from_agent"], m["to_agent"]) for m in ask["exchange"]] == [
        ("question", "judge", "prosecutor"), ("answer", "prosecutor", "judge"),
    ]
    nested = [s for s in v["trace"] if s["parent_call_id"] == ask["call_id"]]
    assert nested and all(s["agent"] == "prosecutor" for s in nested)
    evs = await events(client, run_id)
    msgs = _node_events(evs, jid, "agent.message")
    assert [m["payload"]["kind"] for m in msgs] == ["question", "answer"]
    assert all(m["payload"]["call_id"] == ask["call_id"] for m in msgs)
    assert "The judge asks you" in fake["prompts"]["prosecutor"][-1]
    assert_fold_matches(snap, evs)


# ---------------------------------------------------------------- critic tool


async def test_critic_tool_adds_follow_up_without_cancelling_claims(client, fake):
    fake["thin"] = {"r1"}
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    fu, claims = nodes["r1_fu"], nodes["claims"]
    step = next(s for s in snap["versions"][nodes["c1"]["id"]]["trace"] if s["tool"] == "request_follow_up_research")
    assert step["ok"] and step["node_ids"] == [fu["id"]]
    evs = await events(client, run_id)
    claims_status = [e["payload"]["status"] for e in _node_events(evs, claims["id"], "node.status")]
    assert claims_status == ["running", "done"]  # never started early, never cancelled
    fu_done = max(e["seq"] for e in _node_events(evs, fu["id"], "node.status") if e["payload"]["status"] == "done")
    claims_run = next(e["seq"] for e in _node_events(evs, claims["id"], "node.status"))
    assert claims_run > fu_done
    assert len(await versions(client, claims["id"])) == 1
    assert_fold_matches(snap, evs)


# ---------------------------------------------------------------- distrust + web cache


async def test_distrust_prosecutor_source_and_web_cache(client, fake, web):
    fake["scripts"]["prosecutor"] = lambda text, rnd: (
        [("web_search", {"query": "counter evidence on solar"})] if rnd == 0 else None
    )
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    pid = nodes["prosecutor"]["id"]
    assert "counter evidence on solar" in web.calls
    pv = snap["versions"][pid]
    web_rets = [c for c in pv["retrievals"] if c["via"] == "web"]
    assert web_rets and web_rets[0]["source_url"] == "https://web.test/counter-evidence-on-solar"
    r1_versions = len(await versions(client, nodes["r1"]["id"]))
    calls_before = len(web.calls)

    doc_id = web_rets[0]["document_id"]
    assert (await client.patch(f"/api/documents/{doc_id}", json={"trust": "distrusted"})).status_code == 200
    snap = await wait_run(client, run_id)
    pvs = await versions(client, pid)
    assert len(pvs) == 2 and pvs[-1]["reason"] == "recompute" and pvs[-1]["input_hash"] != pvs[0]["input_hash"]
    assert all(c["document_id"] != doc_id for c in pvs[-1]["retrievals"])
    assert len(await versions(client, nodes["r1"]["id"])) == r1_versions  # research never read it
    assert len(web.calls) == calls_before  # the recompute read the web cache
    cached = next(s for s in pvs[-1]["trace"] if s["tool"] == "web_search")
    assert "cached" in cached["summary"]

    # Rerun bypasses the cache
    assert (await client.post(f"/api/nodes/{nodes['r1']['id']}/rerun")).status_code == 202
    snap = await wait_run(client, run_id)
    assert len(web.calls) > calls_before
    assert_fold_matches(snap, await events(client, run_id))


# ---------------------------------------------------------------- AGENT_TOOLS kill switch


async def test_agent_tools_kill_switch_falls_back_to_single_shot(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "agent_tools", "")
    fake["thin"] = {"r1"}
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    assert all(n["status"] == "done" for n in snap["nodes"])
    assert nodes["r1_fu"]["created_by"] == "critic"  # engine-side follow-up from the critic's output
    r1v = snap["versions"][nodes["r1"]["id"]]
    assert r1v["trace"] == [] and r1v["retrievals"]
    evs = await events(client, run_id)
    agents = {e["payload"]["agent"] for e in evs if e["type"] == "tool.call"}
    assert agents == {"plan"}  # the planner's graph tools are always on
    assert not any(e["payload"]["tool"] == "list_documents" for e in evs if e["type"] == "tool.call")
    assert_fold_matches(snap, evs)
