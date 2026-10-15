"""Per-node-type model settings (no network: only inspects the configured models)."""

from livegraph.agents import models
from livegraph.config import settings


def test_reasoning_only_for_configured_node_types(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(settings, "reasoning_fast", False)
    monkeypatch.setattr(settings, "reasoning_nodes", "prosecutor,defender")
    monkeypatch.setattr(settings, "reasoning_smart", None)
    models.clear_overrides()
    models._openrouter.cache_clear()

    def reasoning_of(node_type: str):
        return models.model_for(node_type, "fast").settings.get("openrouter_reasoning")

    assert reasoning_of("prosecutor") == {"enabled": True}
    assert reasoning_of("defender") == {"enabled": True}
    assert reasoning_of("research") == {"enabled": False}
    assert reasoning_of("claims") == {"enabled": False}
    assert models.model_for("prosecutor", "fast") is not models.model_for("research", "fast")
    assert models.uses_prompted_output(models.model_for("prosecutor", "fast"))
    assert models.model_for("judge", "smart").settings is None or "openrouter_reasoning" not in models.model_for("judge", "smart").settings
    models._openrouter.cache_clear()
    monkeypatch.setattr(settings, "reasoning_smart", False)
    assert models.model_for("judge", "smart").settings.get("openrouter_reasoning") == {"enabled": False}
    models._openrouter.cache_clear()
