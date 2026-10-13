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


def provider_routing(slug: str) -> dict | None:
    """OpenRouter provider routing: prefer ``slug``, fall back to others if it is down."""
    return {"order": [slug], "allow_fallbacks": True} if slug else None


@lru_cache(maxsize=8)
def _openrouter(model_name: str, provider_slug: str = "", reasoning: bool | None = None) -> Model:
    from openai import AsyncOpenAI
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    client = AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "missing-openrouter-key",
        default_headers={"X-Title": "LiveGraph"},
    )
    model_settings = OpenRouterModelSettings()
    if routing := provider_routing(provider_slug):
        model_settings["openrouter_provider"] = routing
    if reasoning is not None:
        model_settings["openrouter_reasoning"] = {"enabled": reasoning}
    return OpenRouterModel(
        model_name, provider=OpenRouterProvider(openai_client=client), settings=model_settings or None
    )


_prompted_models: set[int] = set()


def uses_prompted_output(model: Model) -> bool:
    """True for real fast-tier models configured to return structured output as prompted JSON text."""
    return id(model) in _prompted_models


def smart() -> Model:
    return _overrides.get("smart") or _openrouter(settings.model_smart, settings.provider_smart)


def fast() -> Model:
    if "fast" in _overrides:
        return _overrides["fast"]
    model = _openrouter(settings.model_fast, settings.provider_fast, settings.reasoning_fast)
    if settings.output_mode_fast == "prompted":
        _prompted_models.add(id(model))
    return model


def model_for(node_type: str, tier: Tier) -> Model:
    if node_type in _overrides:
        return _overrides[node_type]
    return smart() if tier == "smart" else fast()
