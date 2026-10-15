"""Judge agent: weighs attacks vs defenses and sets per-claim confidence (and may strike claims).

Tools: read_source (check a cited chunk itself) and ask_rebuttal (the prosecutor/defender agent answers one
pointed question; agent-as-tool). The model returns ``JudgeVerdicts``; the engine stores ``JudgeOutput`` =
verdicts + the rebuttal exchanges, rendered after the verdicts (so the live buffer stays append-only).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from livegraph.agents import validators
from livegraph.agents.base import AgentContext, AgentFactory, AgentLimits, format_claims, format_instructions, format_steer
from livegraph.agents.tools import context_tools, debate_tools, graph_tools, rag_tools
from livegraph.config import settings

TYPE = "judge"
TIER = "smart"


class Verdict(BaseModel):
    claim_key: str
    confidence: float = Field(ge=0.0, le=1.0, description="Probability the claim is true and well supported, 0..1")
    rationale: str = Field(default="", description="One or two sentences explaining the confidence.")
    strike: bool = Field(default=False, description="True if the claim should be removed from the brief.")


class JudgeVerdicts(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


class Rebuttal(BaseModel):
    claim_key: str
    side: str
    question: str
    answer: str
    chunk_ids: list[str] = Field(default_factory=list)


class JudgeOutput(JudgeVerdicts):
    rebuttals: list[Rebuttal] = Field(default_factory=list)


Output = JudgeOutput

SYSTEM = """You are an impartial Judge. For every claim, weigh the Prosecutor's attack against the Defender's
defense and the cited evidence, then set a calibrated `confidence` between 0 and 1 (0.5 = coin flip).
Strike (`strike: true`) claims that are unsupported or refuted. Give a short `rationale` that references the
decisive argument. Output exactly one verdict per claim key."""

TOOL_SYSTEM = """

Tools: `read_source(chunk_id)` reads a cited chunk in full so you can check what it actually says;
`ask_rebuttal(claim_key, side, question)` puts one pointed question to the prosecutor or the defender
(use it only when the debate leaves a decisive point unresolved; at most once per claim and side).
If an upstream input is missing something decisive, `request_upstream_work(target_node_key, instruction)` asks
that agent to redo its work after you return. Then return your verdicts."""

LIMITS = AgentLimits(
    # the judge's own requests plus up to judge_max_rebuttals nested runs (their usage rolls up here)
    request_limit=10 + settings.judge_max_rebuttals * debate_tools.REBUTTAL_REQUESTS,
    timeout=480,
    budgets={"read_source": 6, "ask_rebuttal": settings.judge_max_rebuttals},
)

agents = AgentFactory(
    "judge", JudgeVerdicts, SYSTEM, tool_system=TOOL_SYSTEM,
    tools=[rag_tools.READ, debate_tools.ASK_REBUTTAL, *context_tools.TOOLS, graph_tools.UPSTREAM],
    validator=validators.judge,
)
agent = agents.get()


def build_prompt(ctx: AgentContext) -> str:
    debate = "\n\n".join(
        f"### {p.type.title()} ({p.key})\n{p.output_text}" for p in ctx.parents if p.type in ("prosecutor", "defender")
    )
    others = "\n\n".join(
        f"### {p.title} ({p.key})\n{p.output_text}"
        for p in ctx.parents
        if p.type not in ("prosecutor", "defender", "claims")
    )
    return (
        f"## Goal\n{ctx.goal}\n\n## Claims\n{format_claims(ctx.claims)}\n\n## Debate\n{debate or '(no debate)'}"
        + (f"\n\n## Other inputs\n{others}" if others else "")
        + format_instructions(ctx)
        + format_steer(ctx)
    )


def finalize(out: JudgeVerdicts, deps: Any) -> JudgeOutput:
    rebuttals = [Rebuttal.model_validate(r) for r in (getattr(deps, "rebuttals", None) or [])]
    return JudgeOutput(verdicts=out.verdicts, rebuttals=rebuttals)


def render(out: JudgeVerdicts) -> str:
    lines = []
    for v in out.verdicts:
        mark = " — **struck**" if v.strike else ""
        lines.append(f"- **{v.claim_key}** confidence {v.confidence:.2f}{mark}: {v.rationale}")
    rebuttals = getattr(out, "rebuttals", None) or []
    if rebuttals:
        lines.append("\n**Rebuttals**\n")
        for r in rebuttals:
            lines.append(f"- **{r.claim_key}** asked the {r.side}: {r.question}\n  > {r.answer}")
    return "\n".join(lines)


def parse_edit(text: str, prev: dict) -> dict:
    return {**prev, "edited_markdown": text}
