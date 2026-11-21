from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from lgeval import costs
from lgeval.costs import (
    EST_RUN_USAGE,
    FALLBACK_PRICING,
    Budget,
    Pricing,
    cost_usd,
    est_run_usd,
    estimate_runs_usd,
    load_pricing,
    parse_models,
    resolve_model,
)
from lgeval.schema import TokenUsage

MODELS_BODY = {"data": [
    {"id": "openai/gpt-4.1", "pricing": {"prompt": "0.000002", "completion": "0.000008"}},
    {"id": "acme/cheap", "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}},
    {"id": "broken/model", "pricing": {"prompt": "n/a"}},
]}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> dict:
        raise AssertionError("network used in a test")

    monkeypatch.setattr(costs, "_fetch", boom)
    costs.UNPRICED.clear()


def test_parse_models_converts_to_per_million() -> None:
    table = parse_models(MODELS_BODY)
    assert table["openai/gpt-4.1"] == Pricing(in_per_m=2.0, out_per_m=8.0)
    assert table["acme/cheap"].in_per_m == pytest.approx(0.1)
    assert "broken/model" not in table


def test_offline_uses_fallback() -> None:
    p = load_pricing(["deepseek/deepseek-v4.1-flash", "openai/gpt-4.1-mini"], offline=True)
    assert p["deepseek/deepseek-v4.1-flash"] == Pricing(in_per_m=0.054, out_per_m=0.215)
    assert p["openai/gpt-4.1-mini"].out_per_m == 1.60


def test_fresh_cache_is_used(tmp_path: Path) -> None:
    cache = tmp_path / ".pricing.json"
    cache.write_text(json.dumps({"fetched_at": time.time(), "models": {"acme/cheap": {"in_per_m": 0.1, "out_per_m": 0.2}}}))
    p = load_pricing(["acme/cheap", "openai/gpt-4.1"], cache=cache)
    assert p["acme/cheap"].out_per_m == 0.2
    assert p["openai/gpt-4.1"].in_per_m == 2.0  # fallback fills gaps


def test_stale_cache_refetches_and_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "p.json"
    cache.write_text(json.dumps({"fetched_at": 0, "models": {"acme/cheap": {"in_per_m": 9, "out_per_m": 9}}}))
    monkeypatch.setattr(costs, "_fetch", lambda: parse_models(MODELS_BODY))
    p = load_pricing(["acme/cheap"], cache=cache)
    assert p["acme/cheap"].in_per_m == pytest.approx(0.1)
    assert json.loads(cache.read_text())["fetched_at"] > 0


def test_fetch_failure_falls_back_to_stale_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    cache = tmp_path / "p.json"
    cache.write_text(json.dumps({"fetched_at": 0, "models": {"acme/cheap": {"in_per_m": 9, "out_per_m": 9}}}))

    def fail() -> dict:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(costs, "_fetch", fail)
    assert load_pricing(["acme/cheap"], cache=cache)["acme/cheap"].in_per_m == 9


def test_unknown_model_warns_once_and_costs_zero(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="lgeval.costs")
    assert load_pricing(["nobody/model"], offline=True) == {}
    usage = {"nobody/model": TokenUsage(tokens_in=10**6, tokens_out=10**6)}
    assert cost_usd(usage, {}) == 0.0
    assert cost_usd(usage, {}) == 0.0
    assert costs.UNPRICED == {"nobody/model"}
    assert sum("nobody/model" in r.message for r in caplog.records) == 1


def test_resolve_model_variants() -> None:
    table = {m: Pricing(in_per_m=i, out_per_m=o) for m, (i, o) in FALLBACK_PRICING.items()}
    assert resolve_model("openai/gpt-4.1-2025-04-14", table) == "openai/gpt-4.1"
    assert resolve_model("deepseek/deepseek-v4.1-flash:free", table) == "deepseek/deepseek-v4.1-flash"
    assert resolve_model("gpt-4.1-mini", table) == "openai/gpt-4.1-mini"
    assert resolve_model("nope", table) is None


def test_cost_and_estimates() -> None:
    p = load_pricing([*EST_RUN_USAGE, "openai/gpt-4.1"], offline=True)
    usage = {"openai/gpt-4.1": TokenUsage(tokens_in=1_000_000, tokens_out=500_000)}
    assert cost_usd(usage, p) == pytest.approx(2.0 + 4.0)
    one = est_run_usd(p)
    assert 0.025 < one < 0.04  # ~$0.03 per run: every role on deepseek-v4.1-flash via baidu/fp8
    assert estimate_runs_usd(10, p) == pytest.approx(10 * one)


def test_budget() -> None:
    b = Budget(1.0)
    assert not b.exhausted()
    b.add(0.6)
    assert b.remaining == pytest.approx(0.4)
    b.add(0.4)
    assert b.exhausted() and b.spent == pytest.approx(1.0)
    unlimited = Budget(None)
    unlimited.add(1e9)
    assert not unlimited.exhausted() and unlimited.remaining is None


def test_parse_endpoint_picks_the_pinned_provider() -> None:
    from lgeval.costs import parse_endpoint

    body = {"data": {"endpoints": [
        {"tag": "together", "pricing": {"prompt": "0.0000003", "completion": "0.0000012"}},
        {"tag": "baidu/fp8", "pricing": {"prompt": "0.000000054", "completion": "0.000000215"}},
    ]}}
    p = parse_endpoint(body, "baidu/fp8")
    assert p is not None and p.in_per_m == pytest.approx(0.054) and p.out_per_m == pytest.approx(0.215)
    assert parse_endpoint(body, "missing") is None
