"""Judge: weighs attacks vs defenses and sets per-claim confidence (and may strike claims)."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from livegraph.agents.base import AgentContext, format_claims, format_instructions, format_steer

TYPE = "judge"
TIER = "smart"


class Verdict(BaseModel):
    claim_key: str
    confidence: float = Field(ge=0.0, le=1.0, description="Probability the claim is true and well supported, 0..1")
    rationale: str = Field(default="", description="One or two sentences explaining the confidence.")
    strike: bool = Field(default=False, description="True if the claim should be removed from the brief.")


class JudgeOutput(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


Output = JudgeOutput

SYSTEM = """You are an impartial Judge. For every claim, weigh the Prosecutor's attack against the Defender's
defense and the cited evidence, then set a calibrated `confidence` between 0 and 1 (0.5 = coin flip).
Strike (`strike: true`) claims that are unsupported or refuted. Give a short `rationale` that references the
decisive argument. Output exactly one verdict per claim key."""

agent: Agent[None, JudgeOutput] = Agent(output_type=JudgeOutput, instructions=SYSTEM, retries=2, name="judge")


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


def render(out: JudgeOutput) -> str:
    lines = []
    for v in out.verdicts:
        mark = " — **struck**" if v.strike else ""
        lines.append(f"- **{v.claim_key}** confidence {v.confidence:.2f}{mark}: {v.rationale}")
    return "\n".join(lines)


def parse_edit(text: str, prev: dict) -> dict:
    return {**prev, "edited_markdown": text}
