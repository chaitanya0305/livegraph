"""Prosecutor: attacks each claim's weakest points."""

from __future__ import annotations

from pydantic import BaseModel, Field

from livegraph.agents import debate, validators
from livegraph.agents.base import AgentContext, AgentFactory, cite
from livegraph.agents.debate import build_debate_prompt

TYPE = "prosecutor"
TIER = "fast"


class Attack(BaseModel):
    claim_key: str
    argument: str = Field(description="The strongest specific objection to this claim.")
    chunk_ids: list[str] = Field(default_factory=list, description="chunk_id values backing the objection.")


class ProsecutorOutput(BaseModel):
    attacks: list[Attack] = Field(default_factory=list)


Output = ProsecutorOutput

SYSTEM = """You are the Prosecutor in a structured debate. For each claim, make the strongest honest case
AGAINST it: overreach, weak or outdated sources, missing context, contradicting evidence. Be specific and
cite exact `chunk_id`s (verbatim UUIDs from the evidence list) in `chunk_ids` when evidence supports your
objection. Never invent chunk ids. One attack per claim, referencing it by `claim_key` (C1, C2, ...)."""

LIMITS = debate.LIMITS

agents = AgentFactory(
    "prosecutor", ProsecutorOutput, SYSTEM, tool_system=debate.TOOL_SYSTEM, tools=debate.TOOLS,
    validator=validators.prosecutor,
)
agent = agents.get()


def build_prompt(ctx: AgentContext) -> str:
    return build_debate_prompt(ctx)


def render(out: ProsecutorOutput) -> str:
    return "\n".join(f"- **{a.claim_key}** {a.argument}{cite(a.chunk_ids)}" for a in out.attacks)


def parse_edit(text: str, prev: dict) -> dict:
    return {**prev, "edited_markdown": text}
