"""Deterministic offline stand-ins for every LLM node (pydantic-ai FunctionModel with streaming)."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from livegraph.agents.models import clear_overrides, set_override

# Scenario knobs, reset per test by the `fake_models` fixture.
FAKE: dict[str, Any] = {}


def reset() -> None:
    FAKE.clear()
    FAKE.update(
        {
            "thin": set(),  # research keys the critic calls thin
            "human_input": False,  # planner adds a human_input gate before the editor
            "slow": {},  # node type -> seconds of sleep per streamed chunk
            "prompts": {},  # node type -> list of prompts seen
            "fail": set(),  # node types that raise
        }
    )


reset()

UUID_RE = re.compile(r"chunk_id=([0-9a-f-]{36})")
CLAIM_RE = re.compile(r"^- (C\d+):", re.M)


def prompt_text(messages) -> str:
    parts = []
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    parts.append(p.content)
    return "\n".join(parts)


def steer_of(prompt: str) -> str:
    m = re.search(r"## User steering.*?\n(.*?)(?:\n\n|\Z)", prompt, re.S)
    return m.group(1).strip() if m else ""


async def _stream_output(info: AgentInfo, obj: dict, node_type: str):
    name = info.output_tools[0].name
    s = json.dumps(obj)
    step = 16
    delay = FAKE["slow"].get(node_type, 0)
    for i in range(0, len(s), step):
        if delay:
            await asyncio.sleep(delay)
        yield {0: DeltaToolCall(name=name if i == 0 else None, json_args=s[i : i + step], tool_call_id="out" if i == 0 else None)}


def _model(node_type: str, build) -> FunctionModel:
    async def stream(messages, info: AgentInfo):
        prompt = prompt_text(messages)
        FAKE["prompts"].setdefault(node_type, []).append(prompt)
        if node_type in FAKE["fail"]:
            raise RuntimeError(f"fake {node_type} failure")
        async for chunk in _stream_output(info, build(prompt), node_type):
            yield chunk

    return FunctionModel(stream_function=stream, model_name=f"fake-{node_type}")


def _research(prompt: str) -> dict:
    ids = UUID_RE.findall(prompt)
    q = re.search(r"## Research question\n(.*)", prompt).group(1)
    steer = steer_of(prompt)
    findings = [{"point": f"Finding about {q}" + (f" ({steer})" if steer else ""), "chunk_ids": ids[:2]}]
    if len(ids) > 2:
        findings.append({"point": "Secondary finding", "chunk_ids": ids[2:3] + ["not-a-real-id"]})
    return {"findings": findings, "gaps": [] if ids else ["no evidence"]}


def _claims(prompt: str) -> dict:
    ids = UUID_RE.findall(prompt)
    n_research = len(re.findall(r"^### ", prompt, re.M))
    claims = [
        {"key": "C1", "text": "Solar capacity grew rapidly.", "support_chunk_ids": ids[:1]},
        {"key": "C2", "text": "Battery costs fell sharply.", "support_chunk_ids": ids[1:2]},
    ]
    if n_research > 2:
        claims.append({"key": "C3", "text": f"Follow-up research covered {n_research} sources.", "support_chunk_ids": []})
    steer = steer_of(prompt)
    if steer:
        claims[0]["text"] += f" [{steer}]"
    return {"claims": claims}


def _keys(prompt: str) -> list[str]:
    return CLAIM_RE.findall(prompt) or ["C1"]


def _prosecutor(prompt: str) -> dict:
    return {"attacks": [{"claim_key": k, "argument": f"{k} overreaches.", "chunk_ids": []} for k in _keys(prompt)]}


def _defender(prompt: str) -> dict:
    return {"defenses": [{"claim_key": k, "argument": f"{k} is supported.", "chunk_ids": []} for k in _keys(prompt)]}


def _judge(prompt: str) -> dict:
    steer = steer_of(prompt)
    verdicts = []
    for i, k in enumerate(_keys(prompt)):
        conf = 0.8 if i == 0 else 0.3
        strike = f"STRIKE {k}" in steer
        verdicts.append({"claim_key": k, "confidence": conf, "rationale": f"weighed {k}", "strike": strike})
    return {"verdicts": verdicts}


def _editor(prompt: str) -> dict:
    claims = re.findall(r"^- (C\d+: .*)$", prompt, re.M)
    answer = re.search(r"### .*\(audience\)\n(.*)", prompt)
    body = "\n".join(f"- {c}" for c in claims if "[STRUCK]" not in c)
    extra = f"\n\nAudience: {answer.group(1)}" if answer else ""
    return {"markdown": f"# Brief\n\n{body}{extra}"}


def _critic(prompt: str) -> dict:
    m = re.search(r"^### .* \((\w+)\)$", prompt, re.M)
    key = m.group(1) if m else ""
    if key in FAKE["thin"]:
        return {"verdict": "thin", "follow_up_question": f"What else about {key}?", "reason": "few sources"}
    return {"verdict": "ok", "follow_up_question": None, "reason": "fine"}


def _planner_model() -> FunctionModel:
    async def stream(messages, info: AgentInfo):
        prompt = prompt_text(messages)
        FAKE["prompts"].setdefault("plan", []).append(prompt)
        returned = [p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returned:
            calls: list[tuple[str, dict]] = [
                ("add_node", {"key": "r1", "type": "research", "title": "Solar", "question": "solar capacity growth"}),
                ("add_node", {"key": "r2", "type": "research", "title": "Batteries", "question": "battery cost decline"}),
                ("add_edge", {"source_key": "plan", "target_key": "r1"}),
                ("add_edge", {"source_key": "plan", "target_key": "r2"}),
                ("add_node", {"key": "claims", "type": "claims", "title": "Claims"}),
                ("add_edge", {"source_key": "r1", "target_key": "claims"}),
                ("add_edge", {"source_key": "r2", "target_key": "claims"}),
                ("add_node", {"key": "prosecutor", "type": "prosecutor", "title": "Prosecutor"}),
                ("add_node", {"key": "defender", "type": "defender", "title": "Defender"}),
                ("add_node", {"key": "judge", "type": "judge", "title": "Judge"}),
                ("add_node", {"key": "editor", "type": "editor", "title": "Editor"}),
                ("add_edge", {"source_key": "claims", "target_key": "prosecutor"}),
                ("add_edge", {"source_key": "claims", "target_key": "defender"}),
                ("add_edge", {"source_key": "claims", "target_key": "judge"}),
                ("add_edge", {"source_key": "prosecutor", "target_key": "judge"}),
                ("add_edge", {"source_key": "defender", "target_key": "judge"}),
                ("add_edge", {"source_key": "judge", "target_key": "editor"}),
                # deliberately invalid: cycle -> tool returns an error string, run continues
                ("add_edge", {"source_key": "editor", "target_key": "plan"}),
            ]
            if FAKE["human_input"]:
                calls += [
                    ("add_node", {"key": "audience", "type": "human_input", "title": "Audience",
                                  "instructions": "Who is the brief for?"}),
                    ("add_edge", {"source_key": "plan", "target_key": "audience"}),
                    ("add_edge", {"source_key": "audience", "target_key": "editor"}),
                ]
            calls.append(("finish_plan", {}))
            # critics are intentionally omitted: the engine's ensure_default_shape adds c1/c2
            for i, (name, args) in enumerate(calls):
                yield {i: DeltaToolCall(name=name, json_args=json.dumps(args), tool_call_id=f"t{i}")}
            return
        async for chunk in _stream_output(
            info, {"subquestions": ["solar capacity growth", "battery cost decline"], "rationale": "Two angles."}, "plan"
        ):
            yield chunk

    return FunctionModel(stream_function=stream, model_name="fake-plan")


def install() -> None:
    clear_overrides()
    set_override("plan", _planner_model())
    for t, b in {
        "research": _research,
        "claims": _claims,
        "prosecutor": _prosecutor,
        "defender": _defender,
        "judge": _judge,
        "editor": _editor,
        "critic": _critic,
    }.items():
        set_override(t, _model(t, b))
