"""Context store, history compaction and the cross-agent read tools (all offline, fake models)."""

from __future__ import annotations

import io
import re
import uuid
from types import SimpleNamespace
from uuid import UUID

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from sqlalchemy import select

from livegraph.agents import context
from livegraph.agents.deps import NodeDeps
from livegraph.agents.guard import ToolOut
from livegraph.agents.history import compact, estimate_tokens
from livegraph.agents.tools import context_tools
from livegraph.config import settings
from livegraph.db import base as db_base
from livegraph.db.models import ContextItem
from tests.util import assert_fold_matches, by_key, events, start_run, upload_docs, wait_run

REF_RE = re.compile(r"\[(ctx_[0-9a-f]{8})\]")
LONG_DOC = " ".join(
    f"Solar capacity growth fact {i}: installations of photovoltaic panels expanded in region {i} as module "
    f"prices fell and battery cost decline made storage cheap." for i in range(40)
)


async def _rows(run_id: str, **where) -> list[ContextItem]:
    async with db_base.SessionLocal() as s:
        q = select(ContextItem).where(ContextItem.run_id == UUID(run_id))
        for k, v in where.items():
            q = q.where(getattr(ContextItem, k) == v)
        return list((await s.execute(q.order_by(ContextItem.created_at))).scalars())


async def _done_run(client, long_doc: bool = False) -> tuple[str, dict]:
    await upload_docs(client)
    if long_doc:
        files = {"file": ("long.txt", io.BytesIO(LONG_DOC.encode()), "text/plain")}
        assert (await client.post("/api/documents", files=files)).status_code == 200
    run_id = await start_run(client)
    return run_id, await wait_run(client, run_id)


def _deps(run_id: str, node_id: str, key: str, readable: dict[str, str] | None = None, **kw) -> NodeDeps:
    return NodeDeps(
        run_id=UUID(run_id), node_id=UUID(node_id), node_key=key, node_type=kw.pop("node_type", "judge"),
        sched=None, attempt_id=kw.pop("attempt_id", uuid.uuid4().hex), me=None,
        readable={k: UUID(v) for k, v in (readable or {}).items()}, **kw,
    )


def _ctx(deps: NodeDeps) -> SimpleNamespace:
    return SimpleNamespace(deps=deps, tool_call_id="direct")  # all run_tool needs from a RunContext


# ---------------------------------------------------------------- compaction (pure)


def _history(long: str) -> list:
    return [
        ModelRequest(parts=[UserPromptPart(content="the prompt " * 50)]),
        ModelResponse(parts=[ToolCallPart("search_documents", {"query": "a"}, tool_call_id="c1"),
                             ToolCallPart("read_source", {"chunk_id": "x"}, tool_call_id="c2")]),
        ModelRequest(parts=[ToolReturnPart("search_documents", long + "A", tool_call_id="c1"),
                            ToolReturnPart("read_source", "short", tool_call_id="c2")]),
        ModelResponse(parts=[TextPart("thinking"), ToolCallPart("web_search", {"query": "b"}, tool_call_id="c3")]),
        ModelRequest(parts=[ToolReturnPart("web_search", long + "B", tool_call_id="c3"),
                            RetryPromptPart(content="fix the ids", tool_name="web_search", tool_call_id="c3")]),
        ModelResponse(parts=[ToolCallPart("search_documents", {"query": "c"}, tool_call_id="c4")]),
        ModelRequest(parts=[ToolReturnPart("search_documents", long + "C", tool_call_id="c4")]),
        ModelResponse(parts=[ToolCallPart("search_documents", {"query": "d"}, tool_call_id="c5")]),
        ModelRequest(parts=[ToolReturnPart("search_documents", long + "D", tool_call_id="c5")]),
    ]


def _stub_deps(msgs: list) -> SimpleNamespace:
    refs, stubs = {}, {}
    for i, m in enumerate(msgs):
        for p in m.parts:
            if isinstance(p, ToolReturnPart):
                ref = f"ctx_{i:04d}{p.tool_call_id[-1]:0>4}"
                refs[context.part_key(p.tool_call_id, p.content)] = ref
                stubs[ref] = context.stub_for(ref, f"{p.tool_name}(...) -> n chunks [chunk_id=abc]")
    return SimpleNamespace(refs=refs, stubs=stubs, recall_enabled=True)


def test_compaction_stubs_old_returns_keeps_pairing_and_is_idempotent():
    long = "x" * 3000
    msgs = _history(long)
    deps = _stub_deps(msgs)
    out = compact(msgs, deps, budget=500, keep_recent=2)

    assert len(out) == len(msgs)
    for a, b in zip(msgs, out, strict=True):
        assert type(a) is type(b) and [type(p) for p in a.parts] == [type(p) for p in b.parts]
        assert [getattr(p, "tool_call_id", None) for p in a.parts] == [getattr(p, "tool_call_id", None) for p in b.parts]
    # responses, user prompts and retry prompts are untouched; the last two requests stay verbatim
    for i in (0, 1, 3, 5, 6, 7, 8):
        assert out[i] is msgs[i]
    first, short = out[2].parts
    assert first.content.startswith("[ctx_") and "recall('ctx_" in first.content and "[chunk_id=abc]" in first.content
    assert first.metadata == {"lg_ref": REF_RE.match(first.content).group(1)}
    assert short.content == "short"  # below the size threshold
    web, retry = out[4].parts
    assert web.content.startswith("[ctx_") and retry is msgs[4].parts[1]
    assert msgs[2].parts[0].content == long + "A"  # input not mutated

    again = compact(out, deps, budget=500, keep_recent=2)
    assert again == out
    assert compact(out, deps, budget=10**6, keep_recent=2) is out  # under budget: nothing to do


def test_compaction_only_known_refs_and_respects_budget():
    long = "y" * 3000
    msgs = _history(long)
    assert compact(msgs, SimpleNamespace(refs={}, stubs={}, recall_enabled=True), budget=10, keep_recent=2) is msgs
    deps = _stub_deps(msgs)
    no_recall = SimpleNamespace(refs=deps.refs, stubs=deps.stubs, recall_enabled=False)  # e.g. the planner
    assert compact(msgs, no_recall, budget=10, keep_recent=2) is msgs
    # a budget that one compaction satisfies: only the oldest return is replaced
    out = compact(msgs, deps, budget=estimate_tokens(msgs) - 500, keep_recent=2)
    assert out[2].parts[0].content.startswith("[ctx_")
    assert out[4].parts[0].content == long + "B"


# ---------------------------------------------------------------- store rows, backfill, compaction in a run


async def test_store_rows_backfill_compaction_and_recall_in_agent_loop(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "history_budget_tokens", 300)
    seen: dict[int, str] = {}

    def script(text: str, rnd: int):
        if not re.search(r"## Research question\nsolar", text):
            return None  # only r1 (solar) exercises the loop
        seen[rnd] = text
        if rnd == 0:
            return [("search_documents", {"query": "solar capacity growth photovoltaic", "k": 8})]
        if rnd == 1:
            return [("search_documents", {"query": "battery cost decline storage", "k": 8})]
        if rnd == 2:
            return [("list_agent_runs", {})]
        if rnd == 3:
            refs = REF_RE.findall(text)
            return [("recall", {"ref": refs[0]})] if refs else None
        return None

    fake["scripts"]["research"] = script
    run_id, snap = await _done_run(client, long_doc=True)
    r1 = by_key(snap)["r1"]
    v = snap["versions"][r1["id"]]
    trace = v["trace"]
    assert [s["tool"] for s in trace] == ["search_documents", "search_documents", "list_agent_runs", "recall"]

    # the first search result was compacted out of the history before round 3, the stub carried its ref
    first_ref = trace[0]["ref"]
    assert first_ref and f"[{first_ref}]" in seen[3] and f"recall('{first_ref}')" in seen[3]
    assert f"[{first_ref}]" not in seen[2]  # still within the verbatim window then
    recall_step = trace[3]
    assert recall_step["ok"] and recall_step["ref"] == first_ref and recall_step["reads"] == []
    assert trace[2]["ref"] is None  # list_agent_runs is not stored

    rows = await _rows(run_id, node_id=UUID(r1["id"]))
    assert [r.ref for r in rows] == [trace[0]["ref"], trace[1]["ref"]]
    assert all(r.node_version_id == UUID(v["id"]) and r.kind == "tool_result" and r.agent == "research" for r in rows)
    assert rows[0].summary.startswith("search_documents(") and rows[0].chunk_ids == trace[0]["chunk_ids"]
    assert all(f"[chunk_id={c}]" in rows[0].summary for c in rows[0].chunk_ids)
    assert rows[0].tokens_est == len(rows[0].text) // 4

    evs = await events(client, run_id)
    res = next(e["payload"] for e in evs if e["type"] == "tool.result" and e["payload"]["call_id"] == trace[0]["call_id"])
    assert res["ref"] == first_ref and res["reads"] == []
    item = (await client.get(f"/api/runs/{run_id}/context/{first_ref}")).json()
    assert item["text"] == rows[0].text and item["node_version_id"] == v["id"]
    assert (await client.get(f"/api/runs/{run_id}/context/ctx_00000000")).status_code == 404
    assert_fold_matches(snap, evs)


async def test_budget_exhausted_not_stored_and_cached_repeat_reuses_ref(client, fake):
    def script(text: str, rnd: int):
        if rnd < 3:
            return [("web_search", {"query": f"angle {rnd}"})]
        if rnd in (3, 4):
            return [("search_documents", {"query": "solar capacity growth", "k": 3})]
        return None

    fake["scripts"]["research"] = script
    run_id, snap = await _done_run(client)
    r1 = by_key(snap)["r1"]
    trace = snap["versions"][r1["id"]]["trace"]
    assert [s["tool"] for s in trace] == ["web_search"] * 3 + ["search_documents"] * 2
    assert trace[2]["summary"] == "budget exhausted" and trace[2]["ref"] is None
    assert trace[0]["ref"] and trace[1]["ref"] and trace[0]["ref"] != trace[1]["ref"]
    assert trace[4]["summary"] == "repeat call (cached result)" and trace[4]["ref"] == trace[3]["ref"]
    rows = await _rows(run_id, node_id=UUID(r1["id"]))
    assert [r.tool for r in rows].count("web_search") == 2
    assert [r.tool for r in rows].count("search_documents") == 1
    assert_fold_matches(snap, await events(client, run_id))


# ---------------------------------------------------------------- read tools, scope rule, reads


async def test_judge_reads_upstream_and_non_ancestor_is_rejected(client, fake):
    def judge(text: str, rnd: int):
        if rnd == 0:
            return [
                ("read_agent_output", {"node_key": "prosecutor"}),
                ("read_agent_output", {"node_key": "editor"}),
                ("read_agent_trace", {"node_key": "prosecutor"}),
                ("search_context", {"query": "solar", "k": 5}),
                ("list_agent_runs", {}),
            ]
        if rnd == 1:
            refs = REF_RE.findall(text)
            return [("recall", {"ref": refs[0]})] if refs else None
        return None

    fake["scripts"]["judge"] = judge
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    trace = snap["versions"][nodes["judge"]["id"]]["trace"]
    read_p, read_e, read_t, search, listing, rec = trace
    assert read_p["ok"] and read_p["reads"] == [{"node_key": "prosecutor"}]
    assert not read_e["ok"] and read_e["summary"] == "not readable: editor is not upstream of you"
    assert read_e["reads"] == []
    assert read_t["ok"] and read_t["reads"] == [{"node_key": "prosecutor"}]
    assert search["ok"] and {"node_key": "prosecutor"} in search["reads"]
    assert listing["ok"] and listing["ref"] is None
    prosecutor_refs = {r.ref for r in await _rows(run_id, node_id=UUID(nodes["prosecutor"]["id"]))}
    assert rec["ok"] and rec["ref"] in prosecutor_refs and rec["reads"] == [{"node_key": "prosecutor"}]

    # the judge's prompt advertises the context tools and their budgets
    assert "list_agent_runs: at most 2 calls" in fake["prompts"]["judge"][0]
    assert "## Reading other agents' work" in fake["prompts"]["judge"][0]
    # trace (version) and snapshot/events fold agree on ref + reads
    assert_fold_matches(snap, await events(client, run_id))

    # a defender (upstream: claims, research, plan) cannot recall the prosecutor's results
    readable = {k: nodes[k]["current_version_id"] for k in ("claims", "r1", "r2", "plan")}
    deps = _deps(run_id, nodes["defender"]["id"], "defender", readable, node_type="defender")
    text = await context_tools.recall(_ctx(deps), next(iter(prosecutor_refs)))
    assert text == "not readable: prosecutor is not upstream of you"
    assert await context_tools.read_agent_output(_ctx(deps), "judge") == "not readable: judge is not upstream of you"
    assert (await context_tools.read_agent_output(_ctx(deps), "nope")).startswith("error: unknown node nope")
    out = await context_tools.read_agent_output(_ctx(deps), "claims")
    assert out.startswith("<source node=claims version=1 offset=0") and "C1" in out


async def test_superseded_and_old_version_rows_are_unreadable(client, fake):
    fake["scripts"]["prosecutor"] = lambda text, rnd: (
        [("search_documents", {"query": "evidence against solar", "k": 3})] if rnd == 0 else None
    )
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    pid = nodes["prosecutor"]["id"]
    judge_readable = {k: nodes[k]["current_version_id"] for k in ("prosecutor", "defender", "claims", "r1", "r2", "plan")}
    [live] = await _rows(run_id, node_id=UUID(pid))
    assert live.node_version_id == UUID(nodes["prosecutor"]["current_version_id"])

    # a row of an attempt that never produced a version (superseded mid-flight) has no version id
    dead = ToolOut(text="stale result", summary="1 chunk")
    ghost = _deps(run_id, pid, "prosecutor", node_type="prosecutor", attempt_id="superseded")
    assert await context.put(ghost, ref="ctx_deadbeef", tool="search_documents", args={"query": "q"}, out=dead)
    judge = _deps(run_id, nodes["judge"]["id"], "judge", judge_readable)
    assert await context_tools.recall(_ctx(judge), "ctx_deadbeef") == "not readable: prosecutor is not upstream of you"
    assert (await context_tools.recall(_ctx(judge), live.ref)).startswith(f"<source ref={live.ref} node=prosecutor")

    # after a steer the prosecutor has a new current version: rows of the old one are no longer readable
    assert (await client.post(f"/api/nodes/{pid}/steer", json={"text": "be harsher"})).status_code == 202
    snap = await wait_run(client, run_id)
    nodes = by_key(snap)
    judge = _deps(run_id, nodes["judge"]["id"], "judge", judge_readable)  # fresh repeat cache
    judge.readable["prosecutor"] = UUID(nodes["prosecutor"]["current_version_id"])
    assert await context_tools.recall(_ctx(judge), live.ref) == "not readable: prosecutor is not upstream of you"
    new = [r for r in await _rows(run_id, node_id=UUID(pid)) if r.node_version_id == judge.readable["prosecutor"]]
    assert new and (await context_tools.recall(_ctx(judge), new[0].ref)).startswith("<source ref=")


async def test_recall_paging(client):
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    deps = _deps(run_id, nodes["judge"]["id"], "judge")
    body = "".join(f"{i:04d}" for i in range(2250))  # 9000 chars
    assert await context.put(deps, ref="ctx_0000beef", tool="read_source", args={"chunk_id": "c"},
                             out=ToolOut(text=body, summary="read doc"))
    p1 = await context_tools.recall(_ctx(deps), "ctx_0000beef")
    assert p1.startswith("<source ref=ctx_0000beef node=judge agent=judge tool=read_source total=9000>\n")
    assert body[:4000] in p1 and body[4000:4004] not in p1.split("\n</source>")[0][-10:]
    assert p1.endswith("(more: recall('ctx_0000beef', offset=4000))")
    p3 = await context_tools.recall(_ctx(deps), "ctx_0000beef", offset=8000)
    assert body[8000:] in p3 and "more:" not in p3
    assert (await context_tools.recall(_ctx(deps), "ctx_nothere1")).startswith("error: unknown ref")
    # a page of a stored result is compactable back to that result's stub
    assert deps.stubs["ctx_0000beef"].startswith("[ctx_0000beef] read_source(c) -> read doc")


# ---------------------------------------------------------------- nested rebuttal: own refs, parent_ref


async def test_nested_rebuttal_refs_are_separate(client, fake):
    fake["scripts"]["judge"] = lambda text, rnd: (
        [("ask_rebuttal", {"claim_key": "C1", "side": "prosecutor", "question": "Which source undermines C1?"})]
        if rnd == 0 else None
    )
    run_id, snap = await _done_run(client)
    jid = by_key(snap)["judge"]["id"]
    v = snap["versions"][jid]
    ask = next(s for s in v["trace"] if s["tool"] == "ask_rebuttal")
    nested = [s for s in v["trace"] if s["parent_call_id"] == ask["call_id"]]
    assert nested and all(s["ref"] for s in nested) and ask["ref"]
    assert len({ask["ref"], *(s["ref"] for s in nested)}) == len(nested) + 1

    rows = {r.ref: r for r in await _rows(run_id, node_id=UUID(jid))}
    assert rows[ask["ref"]].kind == "rebuttal" and rows[ask["ref"]].agent == "judge"
    assert "The prosecutor answers on C1" in rows[ask["ref"]].text
    for s in nested:
        r = rows[s["ref"]]
        assert r.agent == "prosecutor" and r.parent_ref == ask["ref"] and r.node_version_id == UUID(v["id"])

    # child deps start with their own (empty) ref/stub maps; the parent's are untouched
    parent = _deps(run_id, jid, "judge")
    parent.refs["k"], parent.stubs["ctx_1"], parent.call_refs["call1"] = "ctx_1", "stub", "ctx_1"
    child = parent.child("prosecutor", "call1", {})
    assert child.refs == {} and child.stubs == {} and child.parent_ref == "ctx_1"
    child.refs["k2"] = "ctx_2"
    assert parent.refs == {"k": "ctx_1"}
    assert_fold_matches(snap, await events(client, run_id))


async def test_context_tools_switch_hides_tools(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "context_tools", False)
    fake["scripts"]["judge"] = lambda text, rnd: [("list_agent_runs", {})] if rnd == 0 else None
    run_id, snap = await _done_run(client)
    trace = snap["versions"][by_key(snap)["judge"]["id"]]["trace"]
    assert trace == []  # the fake only calls tools that are offered
    assert "Reading other agents' work" not in fake["prompts"]["judge"][0]
