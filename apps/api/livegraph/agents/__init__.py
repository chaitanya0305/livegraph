"""Pydantic AI agents, one module per node type. ``SPECS[type]`` is the registry used by the engine."""

from __future__ import annotations

from types import ModuleType

from livegraph.agents import claims, critic, defender, editor, human_input, judge, plan, prosecutor, research

SPECS: dict[str, ModuleType] = {
    m.TYPE: m for m in (plan, research, claims, prosecutor, defender, judge, human_input, editor, critic)
}

NODE_TYPES = tuple(SPECS)
