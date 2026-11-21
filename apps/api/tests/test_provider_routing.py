"""OpenRouter provider routing from PROVIDER_SMART / PROVIDER_FAST."""

from livegraph.agents.models import provider_routing


def test_single_provider_prefers_and_falls_back() -> None:
    assert provider_routing("baidu/fp8") == {"order": ["baidu/fp8"], "allow_fallbacks": True}


def test_provider_list_is_ordered_and_strict() -> None:
    assert provider_routing("baidu/fp8, morph/fp8,") == {"order": ["baidu/fp8", "morph/fp8"], "allow_fallbacks": False}


def test_empty_means_openrouter_default() -> None:
    assert provider_routing("") is None and provider_routing(" , ") is None
