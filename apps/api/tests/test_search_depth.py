"""Search depth: the research loop (reflect + stop rule), chained follow-ups, planner priority budgets, the
critic's search record, the editor's uncited-passage check and the agent recipe in the input hash."""

from __future__ import annotations

import re

from livegraph.agents import SPECS
from livegraph.agents import critic as critic_mod
from livegraph.agents import research as research_mod
from livegraph.agents.base import recipe
from livegraph.agents.validators import uncited_blocks
from livegraph.config import settings
from livegraph.engine.hashing import input_hash
from tests import fakes
from tests.util import (
    assert_fold_matches,
    by_key,
    events,
    start_run,
    upload_docs,
    wait_run,
)


async def _done_run(client) -> tuple[str, dict]:
    await upload_docs(client)
    run_id = await start_run(client)
    return run_id, await wait_run(client, run_id)


def _question(text: str) -> str:
    return re.search(r"## Research question\n(.*)", text).group(1)


# ---------------------------------------------------------------- research loop


async def test_reflect_stop_rule_withdraws_search_tools(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "research_min_new_docs", 50)  # round 2 always adds "too little"

    def script(text: str, rnd: int):
        q = _question(text)
        if rnd == 0:
            return [("web_search", {"query": q}), ("search_documents", {"query": f"{q} data", "k": 4})]
        if rnd == 1:
            return [("reflect", {"known": "some", "open_gaps": ["recent figures"], "next_queries": ["x"]})]
        if rnd == 2:
            return [("web_search", {"query": f"criticism of {q}", "kind": "challenge"}),
                    ("search_documents", {"query": f"{q} figures", "k": 4})]
        if rnd == 3:
            return [("reflect", {"known": "more", "open_gaps": ["recent figures"], "next_queries": ["y"]})]
        if rnd == 4:  # the search tools are gone: the fake drops unavailable tools, so this ends the run
            return [("search_documents", {"query": "after the stop", "k": 4})]
        return None

    fake["scripts"]["research"] = script
    run_id, snap = await _done_run(client)
    v = snap["versions"][by_key(snap)["r1"]["id"]]
    tools = [s["tool"] for s in v["trace"]]
    assert tools.count("reflect") == 2
    stop_at = max(i for i, t in enumerate(tools) if t == "reflect")
    assert "stop" in v["trace"][stop_at]["summary"]
    assert not any(t in ("web_search", "search_documents") for t in tools[stop_at + 1 :])
    assert "kind=challenge" in next(s["args_summary"] for s in v["trace"] if "criticism" in s["args_summary"])
    stats = v["output"]["search_stats"]
    assert stats["rounds"] == 2 and stats["stopped_by"] == "rule"
    assert stats["queries"] == {"explore": 1, "challenge": 1}
    assert stats["docs_seen"] >= stats["docs_cited"] >= 1
    assert_fold_matches(snap, await events(client, run_id))


async def test_reflect_continues_and_nudges_a_challenge_search(client, fake):
    seen: list[str] = []

    def script(text: str, rnd: int):
        seen.append(text)
        if rnd == 0:
            return [("search_documents", {"query": _question(text), "k": 4})]
        if rnd == 1:
            return [("reflect", {"known": "k", "open_gaps": ["g"], "next_queries": ["n"]})]
        return None

    fake["scripts"]["research"] = script
    _, snap = await _done_run(client)
    step = next(s for s in snap["versions"][by_key(snap)["r1"]["id"]]["trace"] if s["tool"] == "reflect")
    assert step["summary"].startswith("round 1:") and "stop" not in step["summary"]  # round 1 never stops
    assert any('kind="challenge"' in t for t in seen)


async def test_research_loop_off_restores_the_previous_agent(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "research_loop", False)
    lim = research_mod.limits({})
    assert lim.request_limit == research_mod.V1_REQUESTS and "reflect" not in lim.budgets
    assert lim.budgets["web_search"] == settings.web_search_max_per_node
    assert research_mod.agents.instructions(True).startswith(research_mod.SYSTEM_V1)
    _, snap = await _done_run(client)
    assert snap["versions"][by_key(snap)["r1"]["id"]]["output"]["search_stats"]["rounds"] == 0


# ---------------------------------------------------------------- budgets: priority x scale


def test_research_limits_scale_with_priority_and_budget_scale(monkeypatch):
    normal = research_mod.limits({})
    high = research_mod.limits({"params": {"priority": "high"}})
    low = research_mod.limits({"params": {"priority": "low"}})
    assert high.budgets["web_search"] > normal.budgets["web_search"] > low.budgets["web_search"] >= 1
    assert high.request_limit > normal.request_limit > low.request_limit
    assert high.budgets["reflect"] == normal.budgets["reflect"] == settings.research_max_rounds + 1  # never scaled
    monkeypatch.setattr(settings, "research_budget_scale", 2.0)
    assert research_mod.limits({}).budgets["web_search"] == 2 * settings.research_web_max


async def test_planner_priority_sets_the_research_budget(client, fake):
    plan = fakes._plan_script("", 0)
    plan = [
        ("add_node", {**a, "priority": "high"}) if n == "add_node" and a.get("key") == "r1" else (n, a) for n, a in plan
    ]
    fake["scripts"]["plan"] = lambda text, rnd: plan if rnd == 0 else None
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    assert nodes["r1"]["config"]["params"]["priority"] == "high"
    assert "priority" not in nodes["r2"]["config"]["params"]
    evs = await events(client, run_id)
    budgets = {
        e["node_id"]: e["payload"]["budget"] for e in evs
        if e["type"] == "tool.call" and e["payload"]["tool"] == "web_search"
    }
    assert budgets[nodes["r1"]["id"]] == research_mod.limits({"params": {"priority": "high"}}).budgets["web_search"]
    assert budgets[nodes["r2"]["id"]] == research_mod.limits({}).budgets["web_search"]


# ---------------------------------------------------------------- critic follow-ups chain to the max depth


async def test_follow_ups_chain_to_max_depth(client, fake):
    fake["thin"] = {"r1", "r1_fu", "r1_fu_fu"}
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    assert nodes["r1_fu"]["config"]["params"]["depth"] == 1 and "c_r1_fu" in nodes
    assert nodes["r1_fu_fu"]["config"]["params"]["depth"] == 2
    assert "c_r1_fu_fu" not in nodes  # at the max depth: no critic, so no deeper follow-up
    claims_parents = {
        e["source_node_id"] for e in snap["edges"] if e["target_node_id"] == nodes["claims"]["id"]
    }
    assert {nodes["r1_fu"]["id"], nodes["r1_fu_fu"]["id"]} <= claims_parents
    assert snap["run"]["status"] == "done"
    assert_fold_matches(snap, await events(client, run_id))


async def test_follow_up_depth_and_run_limits(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "max_follow_ups_per_run", 1)
    fake["thin"] = {"r1", "r2", "r1_fu"}
    _, snap = await _done_run(client)
    nodes = by_key(snap)
    follow_ups = [k for k, n in nodes.items() if n["type"] == "research" and k.endswith("_fu")]
    assert len(follow_ups) == 1  # the run cap
    steps = [
        s for k in ("c1", "c2", "c_r1_fu", "c_r2_fu") if k in nodes
        for s in snap["versions"][nodes[k]["id"]]["trace"] if s["tool"] == "request_follow_up_research"
    ]
    assert sorted(s["summary"] for s in steps if not s["ok"]) == ["follow-up limit"] * (len(steps) - 1)

    monkeypatch.setattr(settings, "max_follow_ups_per_run", 4)
    monkeypatch.setattr(settings, "follow_up_max_depth", 1)  # the previous rule: follow-ups are never followed up
    fake["thin"] = {"r1"}
    _, snap = await _done_run(client)
    nodes = by_key(snap)
    assert "r1_fu" in nodes and "c_r1_fu" not in nodes


# ---------------------------------------------------------------- critic sees the search record


def test_critic_prompt_shows_search_record():
    out = {"search_stats": {"rounds": 2, "stopped_by": "rule", "queries": {"explore": 2, "target": 1},
                            "docs_seen": 9, "docs_cited": 3, "domains_cited": ["nature.com", "who.int"]}}
    line = critic_mod.search_record(out)
    assert "2 reflection round(s)" in line and "explore 2, target 1" in line
    assert "3 of 9 documents" in line and "2 site(s): nature.com, who.int" in line and "added little" in line
    assert critic_mod.search_record({}) == ""


# ---------------------------------------------------------------- editor: passages that reference no claim


def test_uncited_blocks_flags_prose_without_claims_outside_open_questions():
    long = "x" * 130
    md = (
        f"# Brief\n\n{long} with no reference.\n\n{long} supported (C1).\n\n"
        f"- {long} bullet without a claim\n- short bullet\n- {long} bullet (C2, C3)\n\n"
        f"## Open questions\n\n{long} unresolved and fine to leave uncited."
    )
    flagged = uncited_blocks(md)
    assert len(flagged) == 2
    assert flagged[0].endswith("with no reference.") and "bullet without a claim" in flagged[1]


# ---------------------------------------------------------------- recipe: agent version in the input hash


def test_recipe_changes_only_for_the_changed_agent(monkeypatch):
    r_research = recipe(SPECS["research"], None, True)
    r_critic = recipe(SPECS["critic"], None, True)
    assert recipe(SPECS["research"], None, True) == r_research  # deterministic
    assert recipe(SPECS["research"], None, False) != r_research  # tools on/off changes the agent
    monkeypatch.setattr(settings, "research_loop", False)
    assert recipe(SPECS["research"], None, True) != r_research
    assert recipe(SPECS["critic"], None, True) == r_critic
    base = {"node_type": "research", "config": {}, "goal": "g", "parent_version_ids": [], "steer_texts": []}
    assert input_hash(**base, recipe="a") != input_hash(**base, recipe="b")
    assert input_hash(**base) == input_hash(**base, recipe=None)


async def test_thin_verdict_without_the_tool_call_still_gets_its_follow_up(client, fake):
    fake["thin"] = {"r1"}
    fake["scripts"]["critic"] = lambda text, rnd: None  # verdict `thin`, but never calls the tool
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    assert not any(s["tool"] == "request_follow_up_research" for s in snap["versions"][nodes["c1"]["id"]]["trace"])
    fu = nodes["r1_fu"]
    assert fu["config"]["question"] == "What else about r1?" and fu["status"] == "done"
    assert any(e["source_node_id"] == fu["id"] and e["target_node_id"] == nodes["claims"]["id"] for e in snap["edges"])
    assert "r2_fu" not in nodes  # `ok` verdicts add nothing
    assert_fold_matches(snap, await events(client, run_id))


# ---------------------------------------------------------------- research evidence reaching claims / debate


def test_select_evidence_takes_cited_chunks_round_robin_across_research_nodes():
    from livegraph.agents.base import select_evidence

    def c(cid, research, score, cited=True):
        return {"chunk_id": cid, "research": research, "score": score, "cited": cited, "text": "t"}

    # r1 cited many high-scoring chunks; r2 and r3 fewer and lower: they must still get in
    chunks = [c(f"a{i}", "r1", 0.9 - i / 100) for i in range(6)] + [c("b0", "r2", 0.5), c("b1", "r2", 0.4),
              c("c0", "r3", 0.3), c("u0", "r1", 0.95, cited=False)]
    picked = [x["chunk_id"] for x in select_evidence(chunks, 5)]
    assert picked == ["a0", "b0", "c0", "a1", "b1"]  # round-robin, best first within each node
    assert [x["chunk_id"] for x in select_evidence(chunks, 11)][-1] == "u0"  # uncited only after every cited chunk
    assert select_evidence([], 10) == []
