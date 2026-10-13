"""human_input: no LLM; waits for the user's answer."""

from __future__ import annotations

from pydantic import BaseModel

TYPE = "human_input"
TIER = None


class HumanInputOutput(BaseModel):
    answer: str


Output = HumanInputOutput


def prompt_for(config: dict, title: str) -> str:
    return (config or {}).get("prompt") or (config or {}).get("instructions") or (config or {}).get("question") or title


def render(out: HumanInputOutput) -> str:
    return out.answer


def parse_edit(text: str, prev: dict) -> dict:
    return {"answer": text}
