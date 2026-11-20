"""Named LiveGraph system configs for the Claim A matrix: each is a set of env overrides for the API server.

Every config pins all ablation knobs (BASE), so a stray value in apps/api/.env cannot leak into a run and
two configs differ only in what they ablate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_AGENT_TOOLS = "research,prosecutor,defender,judge,critic,plan"

# models of the study: every role on DeepSeek v4.1 Flash (~$0.011/run vs ~$0.13 with gpt-4.1 as smart model).
# Providers: the cheapest fp8 endpoints in price order (Oct 2026: baidu $0.0105/run, morph $0.0164, deepinfra
# flex $0.0193, deepinfra $0.0241); OpenRouter never leaves the list, so no run lands on a pricier provider or
# on fp4 weights. Pinned here, not taken from apps/api/.env, so the dev app's models never leak into results.
STUDY_PROVIDERS = "baidu/fp8,morph/fp8,deepinfra/fp8/flex,deepinfra/fp8"
STUDY_MODELS: dict[str, str] = {
    "MODEL_SMART": "deepseek/deepseek-v4.1-flash",
    "PROVIDER_SMART": STUDY_PROVIDERS,
    "REASONING_SMART": "false",
    "MODEL_FAST": "deepseek/deepseek-v4.1-flash",
    "PROVIDER_FAST": STUDY_PROVIDERS,
}

# the knobs the configs below vary, at their LiveGraph defaults
BASE: dict[str, str] = {
    **STUDY_MODELS,
    "DISABLED_NODE_TYPES": "",
    "AGENT_TOOLS": DEFAULT_AGENT_TOOLS,
    "CONTEXT_TOOLS": "true",
    # search depth (livegraph/config.py): the research loop, full web pages, chained follow-ups, planner scouting
    "RESEARCH_LOOP": "true",
    "WEB_PAGE_TEXT": "true",
    "FOLLOW_UP_MAX_DEPTH": "2",
    "PLAN_WEB_SEARCH": "true",
    "RESEARCH_BUDGET_SCALE": "1.0",
    "REASONING_NODES": "prosecutor,defender",
    "EVIDENCE_PROMPT_CHUNKS": "100",
    "EVIDENCE_PROMPT_CHARS": "600",
}


@dataclass(frozen=True)
class SystemConfig:
    name: str
    ablates: str  # one line: what this config removes relative to `full`
    overrides: dict[str, str] = field(default_factory=dict)

    @property
    def env(self) -> dict[str, str]:
        return {**BASE, **self.overrides}


CONFIGS: dict[str, SystemConfig] = {
    c.name: c
    for c in (
        SystemConfig("full", "nothing: the default Living Brief graph with every agent and tool"),
        SystemConfig(
            "no_debate",
            "prosecutor, defender and judge: claims feed the editor directly and stay unjudged (no confidence)",
            {"DISABLED_NODE_TYPES": "prosecutor,defender,judge"},
        ),
        SystemConfig(
            "no_critic",
            "the research critics and the follow-up research they add",
            {"DISABLED_NODE_TYPES": "critic"},
        ),
        SystemConfig(
            "single_shot",
            "agent tool loops: every node is one LLM call (research gets engine-side search; planner keeps graph tools)",
            {"AGENT_TOOLS": ""},
        ),
        SystemConfig(
            "old_search",
            "the search-depth changes: single-pass research on highlight snippets, unchained follow-ups, no planner "
            "scouting (the pre-loop agent)",
            {"RESEARCH_LOOP": "false", "WEB_PAGE_TEXT": "false", "FOLLOW_UP_MAX_DEPTH": "1", "PLAN_WEB_SEARCH": "false"},
        ),
        SystemConfig(
            "research_reasoning",
            "nothing removed: research agents also get the model's hidden reasoning (vs. reflect alone)",
            {"REASONING_NODES": "prosecutor,defender,research"},
        ),
        SystemConfig(
            "budget_x0.5",
            "half the research search budget and request cap (search-depth sweep)",
            {"RESEARCH_BUDGET_SCALE": "0.5"},
        ),
        SystemConfig(
            "budget_x2",
            "nothing removed: twice the research search budget and request cap (search-depth sweep)",
            {"RESEARCH_BUDGET_SCALE": "2.0"},
        ),
        SystemConfig(
            "no_context_tools",
            "cross-agent read tools (list_agent_runs, read_agent_output, read_agent_trace, recall, search_context)",
            {"CONTEXT_TOOLS": "false"},
        ),
    )
}


def parse_names(names: str | list[str]) -> list[str]:
    """`"full,no_debate"` or a list -> config names, validated, order kept, duplicates dropped."""
    items = names.split(",") if isinstance(names, str) else list(names)
    out = list(dict.fromkeys(n.strip() for n in items if n.strip()))
    unknown = [n for n in out if n not in CONFIGS]
    if unknown:
        raise ValueError(f"unknown config(s) {', '.join(unknown)}; expected one of {', '.join(CONFIGS)}")
    return out


def resolve(names: str | list[str]) -> dict[str, dict[str, str]]:
    """Config name -> env overrides (the shape of `Manifest.configs`)."""
    return {n: CONFIGS[n].env for n in parse_names(names)}


def describe() -> str:
    return "\n".join(f"  {c.name:<17} ablates {c.ablates}" for c in CONFIGS.values())
