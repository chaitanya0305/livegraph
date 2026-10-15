"""Defender: argues for each claim using the evidence."""

from __future__ import annotations

from pydantic import BaseModel, Field

from livegraph.agents import debate, validators
from livegraph.agents.base import AgentContext, AgentFactory, cite
from livegraph.agents.debate import build_debate_prompt

TYPE = "defender"
TIER = "fast"


class Defense(BaseModel):
    claim_key: str
    argument: str = Field(description="The strongest evidence-based defense of this claim.")
    chunk_ids: list[str] = Field(default_factory=list, description="chunk_id values backing the defense.")


class DefenderOutput(BaseModel):
    defenses: list[Defense] = Field(default_factory=list)


Output = DefenderOutput

SYSTEM = """You are the Defender in a structured debate. For each claim, make the strongest honest case FOR
it using the evidence. Cite exact `chunk_id`s (verbatim UUIDs from the evidence list) in `chunk_ids`.
Never invent chunk ids. If a claim is genuinely indefensible, say so plainly. One defense per claim,
referencing it by `claim_key` (C1, C2, ...)."""

LIMITS = debate.LIMITS

agents = AgentFactory(
    "defender", DefenderOutput, SYSTEM, tool_system=debate.TOOL_SYSTEM, tools=debate.TOOLS,
    validator=validators.defender,
)
agent = agents.get()


def build_prompt(ctx: AgentContext) -> str:
    return build_debate_prompt(ctx)


def render(out: DefenderOutput) -> str:
    return "\n".join(f"- **{d.claim_key}** {d.argument}{cite(d.chunk_ids)}" for d in out.defenses)


def parse_edit(text: str, prev: dict) -> dict:
    return {**prev, "edited_markdown": text}
