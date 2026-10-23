"""DISABLED_NODE_TYPES ablation knob (lgeval): disabled types are never added and the graph is rewired."""

from __future__ import annotations

from livegraph.config import disabled_types, settings
from tests.util import assert_fold_matches, by_key, events, start_run, upload_docs, wait_run


async def _done_run(client) -> tuple[str, dict]:
    await upload_docs(client)
    run_id = await start_run(client)
    return run_id, await wait_run(client, run_id)


def _parents(snap: dict, key: str) -> set[str]:
    ids = {n["id"]: n["key"] for n in snap["nodes"]}
    tgt = by_key(snap)[key]["id"]
    return {ids[e["source_node_id"]] for e in snap["edges"] if e["target_node_id"] == tgt}


def test_disabled_types_only_counts_disablable(monkeypatch):
    monkeypatch.setattr(settings, "disabled_node_types", " judge, editor ,bogus,critic")
    assert disabled_types() == {"judge", "critic"}
    monkeypatch.setattr(settings, "disabled_node_types", "")
    assert disabled_types() == set()


async def test_no_debate_wires_claims_to_editor_unjudged(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "disabled_node_types", "prosecutor,defender,judge")
    run_id, snap = await _done_run(client)
    types = {n["type"] for n in snap["nodes"]}
    assert not types & {"prosecutor", "defender", "judge"}
    assert {"claims", "editor", "critic"} <= types
    assert all(n["status"] == "done" for n in snap["nodes"])
    assert _parents(snap, "editor") == {"claims"}
    assert snap["claims"] and all(c["confidence"] is None for c in snap["claims"])
    assert "## Disabled node types" in fake["prompts"]["plan"][0]
    assert "not judged" in fake["prompts"]["editor"][-1]
    evs = await events(client, run_id)
    rejected = [e for e in evs if e["type"] == "tool.result" and not e["payload"]["ok"]]
    assert any("disabled" in (e["payload"]["error"] or "") for e in rejected)
    assert_fold_matches(snap, evs)


async def test_no_critic_has_no_critics_or_follow_ups(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "disabled_node_types", "critic")
    fake["thin"] = {"r1"}
    run_id, snap = await _done_run(client)
    nodes = by_key(snap)
    assert not any(n["type"] == "critic" for n in snap["nodes"]) and "r1_fu" not in nodes
    assert _parents(snap, "editor") == {"judge"}
    assert all(n["status"] == "done" for n in snap["nodes"])
    assert "## Disabled node types" in fake["prompts"]["plan"][0]
    assert "not judged" not in fake["prompts"]["editor"][-1]
    assert_fold_matches(snap, await events(client, run_id))


async def test_default_planner_prompt_has_no_disabled_section(client, fake):
    await _done_run(client)
    assert "Disabled node types" not in fake["prompts"]["plan"][0]
    assert "with judged confidence" in fake["prompts"]["editor"][-1]
