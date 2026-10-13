"""Shared prompt builder for prosecutor/defender."""

from __future__ import annotations

from livegraph.agents.base import AgentContext, format_claims, format_evidence, format_instructions, format_steer


def build_debate_prompt(ctx: AgentContext) -> str:
    return (
        f"## Goal\n{ctx.goal}\n\n## Claims\n{format_claims(ctx.claims)}\n\n## Evidence chunks\n"
        f"{format_evidence(ctx.evidence)}{format_instructions(ctx)}{format_steer(ctx)}"
    )
