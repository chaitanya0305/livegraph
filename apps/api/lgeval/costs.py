"""OpenRouter pricing (fetch + 24h cache + fallback table), token usage -> USD, and a spending guard."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import httpx
from pydantic import BaseModel

from lgeval.schema import TokenUsage
from lgeval.store import DEFAULT_ROOT

log = logging.getLogger("lgeval.costs")

MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_S = 24 * 3600
DEFAULT_CACHE = DEFAULT_ROOT / ".pricing.json"


class Pricing(BaseModel):
    in_per_m: float  # USD per million input tokens
    out_per_m: float  # USD per million output tokens


# $/M (input, output), checked 2026-10-22
FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v4.1-flash": (0.054, 0.215),  # baidu/fp8 (list price 0.30 / 1.20)
    "openai/gpt-4.1": (2.00, 8.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "deepseek/deepseek-v4-pro": (0.44, 0.89),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "xiaomi/mimo-v2.6-pro": (0.387, 0.783),  # deepinfra/fp8 (list price 0.435 / 0.87)
}

# Measured on the study setup (every role on deepseek-v4.1-flash, multi-agent nodes; 2 complete smoke runs,
# 2026-10-23): 270K/32K and 641K/66K tokens -> ~$0.02-0.05/run. Research nodes are ~55% of the input tokens.
EST_RUN_USAGE: dict[str, TokenUsage] = {
    "deepseek/deepseek-v4.1-flash": TokenUsage(tokens_in=400_000, tokens_out=45_000),
}

# Rough cost of other systems/configs relative to one full run, for --dry-run estimates
CONFIG_COST_FACTOR: dict[str, float] = {
    "full": 1.0,
    "no_debate": 0.6,
    "no_critic": 0.9,
    "no_context_tools": 0.9,
    "single_shot": 0.6,
    "single_agent": 0.3,
}
RECOMPUTE_FACTOR = 0.35  # one incremental update after a mutation, relative to a full run
SCORE_USD_PER_BRIEF = 0.03  # LLM scoring of one brief (all metrics) with xiaomi/mimo-v2.6-pro, low reasoning

UNPRICED: set[str] = set()  # models seen without a price (counted as $0)


def _fallback() -> dict[str, Pricing]:
    return {m: Pricing(in_per_m=i, out_per_m=o) for m, (i, o) in FALLBACK_PRICING.items()}


def parse_models(payload: dict) -> dict[str, Pricing]:
    """OpenRouter /models body ($/token strings) -> {model id: Pricing in $/M}."""
    out: dict[str, Pricing] = {}
    for m in payload.get("data", []):
        p = m.get("pricing") or {}
        try:
            out[m["id"]] = Pricing(in_per_m=float(p["prompt"]) * 1e6, out_per_m=float(p["completion"]) * 1e6)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _read_cache(cache: Path, max_age: float | None) -> dict[str, Pricing] | None:
    try:
        raw = json.loads(cache.read_text())
    except (OSError, ValueError):
        return None
    if max_age is not None and time.time() - float(raw.get("fetched_at", 0)) > max_age:
        return None
    return {k: Pricing.model_validate(v) for k, v in raw.get("models", {}).items()}


# OpenRouter provider the study prefers per model (runners.configs.STUDY_MODELS, metrics.llm judge): these are
# priced at that provider's endpoint, not the model's list price, which can be several times higher. A request
# that falls back to a later provider in STUDY_PROVIDERS costs up to ~2.3x more (deepinfra/fp8), not counted here.
PROVIDER_PINS: dict[str, str] = {
    "deepseek/deepseek-v4.1-flash": "baidu/fp8",
    "xiaomi/mimo-v2.6-pro": "deepinfra/fp8",
}


def parse_endpoint(payload: dict, tag: str) -> Pricing | None:
    """OpenRouter /models/{id}/endpoints body -> the price of the endpoint with `tag` (e.g. "baidu/fp8")."""
    for e in (payload.get("data") or {}).get("endpoints", []):
        if e.get("tag") == tag:
            p = e.get("pricing") or {}
            try:
                return Pricing(in_per_m=float(p["prompt"]) * 1e6, out_per_m=float(p["completion"]) * 1e6)
            except (KeyError, TypeError, ValueError):
                return None
    return None


def _fetch(timeout: float = 15.0) -> dict[str, Pricing]:
    r = httpx.get(MODELS_URL, timeout=timeout)
    r.raise_for_status()
    table = parse_models(r.json())
    for model, tag in PROVIDER_PINS.items():
        try:
            e = httpx.get(f"{MODELS_URL}/{model}/endpoints", timeout=timeout)
            e.raise_for_status()
            if (price := parse_endpoint(e.json(), tag)) is not None:
                table[model] = price
            else:
                log.warning("no %s endpoint for %s; using its list price", tag, model)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("endpoint pricing for %s failed (%s); using its list price", model, exc)
    return table


def pricing_table(*, cache: Path | None = None, offline: bool = False) -> dict[str, Pricing]:
    """All known prices: fresh cache, else OpenRouter (then cached), else stale cache; fallback fills gaps."""
    cache = cache or DEFAULT_CACHE
    table = None if offline else _read_cache(cache, CACHE_TTL_S)
    if table is None and not offline:
        try:
            table = _fetch()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"fetched_at": time.time(),
                                         "models": {k: v.model_dump() for k, v in table.items()}}))
        except (httpx.HTTPError, OSError, ValueError) as e:
            log.warning("OpenRouter pricing fetch failed (%s); using cache/fallback", e)
            table = _read_cache(cache, None)
    return {**_fallback(), **(table or {})}


def resolve_model(name: str, pricing: dict[str, Pricing]) -> str | None:
    """Pricing key for a model name as reported by a provider (date suffixes, ':variant', no vendor)."""
    if name in pricing:
        return name
    cands = [name.split(":")[0]]
    cands.append(re.sub(r"-(\d{4}-\d{2}-\d{2}|\d{8}|\d{4})$", "", cands[0]))
    for c in cands:
        if c in pricing:
            return c
    tail = cands[-1].split("/")[-1]
    hits = [k for k in pricing if k.split("/")[-1] == tail]
    return hits[0] if len(hits) == 1 else None


def _warn_unpriced(model: str) -> None:
    if model not in UNPRICED:
        UNPRICED.add(model)
        log.warning("no price for model %r; counting it as $0", model)


def load_pricing(models: list[str], *, cache: Path | None = None, offline: bool = False) -> dict[str, Pricing]:
    """Prices for `models` (OpenRouter, cached 24h; offline or failed -> FALLBACK_PRICING).
    Unknown models are left out (warned once)."""
    table = pricing_table(cache=cache, offline=offline)
    out: dict[str, Pricing] = {}
    for m in models:
        key = resolve_model(m, table)
        if key is None:
            _warn_unpriced(m)
        else:
            out[m] = table[key]
    return out


def cost_usd(usage: dict[str, TokenUsage], pricing: dict[str, Pricing]) -> float:
    total = 0.0
    for model, u in usage.items():
        key = resolve_model(model, pricing)
        if key is None:
            if u.tokens_in or u.tokens_out:
                _warn_unpriced(model)
            continue
        p = pricing[key]
        total += u.tokens_in / 1e6 * p.in_per_m + u.tokens_out / 1e6 * p.out_per_m
    return total


def est_run_usd(pricing: dict[str, Pricing] | None = None) -> float:
    """Estimated USD of one full LiveGraph run (EST_RUN_USAGE)."""
    return cost_usd(EST_RUN_USAGE, pricing if pricing is not None else _fallback())


def estimate_runs_usd(n_runs: float, pricing: dict[str, Pricing] | None = None) -> float:
    return n_runs * est_run_usd(pricing)


def pricing_manifest(pricing: dict[str, Pricing]) -> dict[str, dict[str, float]]:
    """Shape for Manifest.pricing."""
    return {k: v.model_dump() for k, v in pricing.items()}


class Budget:
    """Spending guard: callers add measured spend and stop starting new work once `exhausted()`."""

    def __init__(self, max_usd: float | None) -> None:
        self.max_usd = max_usd
        self.spent = 0.0

    def add(self, usd: float) -> None:
        self.spent += usd

    def exhausted(self) -> bool:
        return self.max_usd is not None and self.spent >= self.max_usd

    @property
    def remaining(self) -> float | None:
        return None if self.max_usd is None else max(0.0, self.max_usd - self.spent)

    def __repr__(self) -> str:
        cap = "∞" if self.max_usd is None else f"${self.max_usd:.2f}"
        return f"Budget(spent=${self.spent:.4f} of {cap})"
