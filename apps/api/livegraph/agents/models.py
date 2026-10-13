"""OpenRouter model factory with a test override hook.

``smart()`` / ``fast()`` return Pydantic AI models for ``settings.model_smart`` / ``settings.model_fast``.
Tests call ``set_override(...)`` to swap in ``FunctionModel``/``TestModel`` (per tier or per node type).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai.models import Model  # noqa: E402

from livegraph.config import settings  # noqa: E402

Tier = Literal["smart", "fast"]

# key: a tier ("smart"/"fast") or a node type ("research", "judge", ...); node type wins.
_overrides: dict[str, Model] = {}


def set_override(key: str, model: Model | None) -> None:
    if model is None:
        _overrides.pop(key, None)
    else:
        _overrides[key] = model


def clear_overrides() -> None:
    _overrides.clear()


@lru_cache(maxsize=8)
def _openrouter(model_name: str) -> Model:
    from openai import AsyncOpenAI
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    client = AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "missing-openrouter-key",
        default_headers={"X-Title": "LiveGraph"},
    )
    return OpenRouterModel(model_name, provider=OpenRouterProvider(openai_client=client))


def smart() -> Model:
    return _overrides.get("smart") or _openrouter(settings.model_smart)


def fast() -> Model:
    return _overrides.get("fast") or _openrouter(settings.model_fast)


def model_for(node_type: str, tier: Tier) -> Model:
    if node_type in _overrides:
        return _overrides[node_type]
    return smart() if tier == "smart" else fast()
