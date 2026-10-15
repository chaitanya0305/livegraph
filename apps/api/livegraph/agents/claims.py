"""Claims node: distills research into numbered claims (C1..Cn) with supporting chunk ids."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from livegraph.agents import validators
from livegraph.agents.base import (
    AgentContext,
    AgentFactory,
    AgentLimits,
    cite,
    format_evidence,
    format_instructions,
    format_steer,
)
from livegraph.agents.tools import context_tools

TYPE = "claims"
TIER = "fast"


class ClaimItem(BaseModel):
    key: str = Field(description="C1, C2, ... in order")
    text: str = Field(description="One crisp, falsifiable claim answering part of the goal.")
    support_chunk_ids: list[str] = Field(default_factory=list, description="chunk_id values supporting the claim.")


class ClaimsOutput(BaseModel):
    claims: list[ClaimItem] = Field(default_factory=list)


Output = ClaimsOutput

SYSTEM = """You turn research findings into a short list of clear, falsifiable claims that together answer
the user's goal. Number them C1, C2, ... (keys). Each claim must be supported by evidence: list the exact
`chunk_id`s (verbatim UUIDs from the evidence list) in `support_chunk_ids`. Do not invent chunk ids.
Prefer 4-8 claims. Merge duplicates; drop findings with no support."""

LIMITS = AgentLimits(request_limit=3, timeout=180)

# context tools only run with CONTEXT_TOOLS_SINGLE_SHOT=true (otherwise this stays a validated single shot)
agents = AgentFactory(
    "claims", ClaimsOutput, SYSTEM, tools=context_tools.TOOLS, validator=validators.claims
)
agent = agents.get()


def build_prompt(ctx: AgentContext) -> str:
    research = "\n\n".join(f"### {p.title} ({p.key})\n{p.output_text}" for p in ctx.parents) or "(none)"
    return (
        f"## Goal\n{ctx.goal}\n\n## Research results\n{research}\n\n## Evidence chunks\n"
        f"{format_evidence(ctx.evidence)}{format_instructions(ctx)}{format_steer(ctx)}"
    )


def render(out: ClaimsOutput) -> str:
    return "\n".join(f"- **{c.key}** {c.text}{cite(c.support_chunk_ids)}" for c in out.claims)


def sanitize(out: ClaimsOutput, valid_ids: set[str]) -> ClaimsOutput:
    for i, c in enumerate(out.claims, 1):
        c.key = c.key.strip() or f"C{i}"
        c.support_chunk_ids = [x for x in c.support_chunk_ids if x in valid_ids]
    return out


_LINE = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*\**\s*(C\d+)\s*\**\s*[:.)\-–]?\s*(.+?)\s*$")
_CITE = re.compile(r"\s*\[[0-9a-fA-F-]{4,36}\]")


def parse_edit(text: str, prev: dict) -> dict:
    """Re-parse an edited markdown claim list ("- **C1** text"). Keeps support ids by key."""
    prev_support = {c.get("key"): c.get("support_chunk_ids", []) for c in prev.get("claims", [])}
    claims = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        key, body = m.group(1), _CITE.sub("", m.group(2)).strip()
        if body:
            claims.append({"key": key, "text": body, "support_chunk_ids": prev_support.get(key, [])})
    if not claims:
        return {**prev, "edited_markdown": text}
    return {"claims": claims}
