"""Judge LLM: an `LLM` callable over OpenRouter with a disk cache, retries and budget accounting.

Every scorer takes `llm: LLM = (system, user) -> Awaitable[str]`, so tests pass fakes. The default judge is a
model from a different family than the systems under test (LiveGraph uses gpt-4.1 + deepseek) to limit
self-preference bias (Zheng et al., 2023; Panickssery et al., 2024); the tradeoff is that a smaller judge is a
weaker verifier, so judge agreement with human labels should be spot-checked (see `calibration.load_human_labels`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from lgeval.metrics import prompts

LLM = Callable[[str, str], Awaitable[str]]

# different model family from the system under test (DeepSeek in the study), to limit self-preference bias;
# the strongest model under ~$1/M on the Artificial Analysis Intelligence Index (46, Oct 2026)
DEFAULT_JUDGE_MODEL = "xiaomi/mimo-v2.6-pro"
# cheapest fp8 endpoints in price order (deepinfra 0.387/0.783, then xiaomi and novita 0.435/0.87); priced at
# the first via costs.PROVIDER_PINS; OpenRouter never leaves the list
DEFAULT_JUDGE_PROVIDER = "deepinfra/fp8,xiaomi/fp8,novita/fp8"
DEFAULT_JUDGE_REASONING = "low"  # OpenRouter reasoning effort: verification benefits, cost stays bounded
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class BudgetExhausted(RuntimeError):
    """Raised before a judge call once the spending cap is reached."""


class JudgeError(RuntimeError):
    """The judge reply could not be parsed into the required JSON shape (after one retry)."""


def default_root() -> Path:
    try:
        from lgeval.store import DEFAULT_ROOT

        return Path(DEFAULT_ROOT)
    except Exception:
        return Path(__file__).resolve().parents[4] / "eval-results"


def _settings() -> tuple[str, str]:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    base = os.environ.get("OPENROUTER_BASE_URL", "")
    if not key or not base:
        try:
            from livegraph.config import get_settings

            s = get_settings()
            key = key or s.openrouter_api_key
            base = base or s.openrouter_base_url
        except Exception:
            pass
    return key, (base or "https://openrouter.ai/api/v1").rstrip("/")


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    errors: int = 0


@dataclass
class OpenRouterLLM:
    """Callable judge. `stats` counts real calls, cache hits, tokens and USD of this instance."""

    model: str
    budget: Any = None  # costs.Budget
    temperature: float = 0.0
    provider: str = ""  # OpenRouter provider tag(s): one = preferred, comma list = only these, in order
    reasoning_effort: str | None = None  # "low" | "medium" | "high"; None = model default
    json_mode: bool = True
    cache_dir: Path | None = None
    max_retries: int = 5
    timeout: float = 120.0
    transport: httpx.AsyncBaseTransport | None = None
    pricing: dict[str, Any] | None = None  # model -> costs.Pricing; loaded lazily
    stats: LLMStats = field(default_factory=LLMStats)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    def _cache_path(self, system: str, user: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps([self.model, self.temperature, system, user], ensure_ascii=False)
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / re.sub(r"[^A-Za-z0-9._-]", "_", self.model) / f"{digest}.json"

    def _price(self, tokens_in: int, tokens_out: int) -> float:
        try:
            from lgeval.costs import cost_usd, load_pricing
            from lgeval.schema import TokenUsage

            if self.pricing is None:
                self.pricing = load_pricing([self.model])
            return cost_usd({self.model: TokenUsage(tokens_in=tokens_in, tokens_out=tokens_out)}, self.pricing)
        except Exception:
            return 0.0

    async def __call__(self, system: str, user: str) -> str:
        path = self._cache_path(system, user)
        if path is not None and path.exists():
            try:
                cached = json.loads(path.read_text())["text"]
                self.stats.cache_hits += 1
                return cached
            except (OSError, ValueError, KeyError):
                pass
        if self.budget is not None and self.budget.exhausted():
            raise BudgetExhausted(f"judge budget exhausted (${self.budget.spent:.2f} spent)")
        text, usage = await self._complete(system, user)
        tin, tout = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        usd = self._price(tin, tout)
        self.stats.calls += 1
        self.stats.tokens_in += tin
        self.stats.tokens_out += tout
        self.stats.usd += usd
        if self.budget is not None:
            self.budget.add(usd)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"model": self.model, "text": text, "usage": usage}, ensure_ascii=False))
            tmp.replace(path)
        return text

    async def _complete(self, system: str, user: str) -> tuple[str, dict]:
        key, base = _settings()
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self.transport, timeout=self.timeout)
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if order := [p.strip() for p in self.provider.split(",") if p.strip()]:
            body["provider"] = {"order": order, "allow_fallbacks": len(order) == 1}
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        json_mode = self.json_mode
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self.max_retries):
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            else:
                body.pop("response_format", None)
            try:
                r = await self._client.post(
                    f"{base}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {key}", "X-Title": "lgeval"},
                )
                if r.status_code == 400 and json_mode and "response_format" in r.text:
                    json_mode = False  # model does not support JSON mode; prompts still demand JSON
                    continue
                if r.status_code in RETRY_STATUS:
                    raise httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                if "error" in data and not data.get("choices"):
                    raise httpx.HTTPError(str(data["error"]))
                text = data["choices"][0]["message"].get("content") or ""
                return text, data.get("usage") or {}
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, httpx.HTTPError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in RETRY_STATUS:
                    raise
                last = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
        self.stats.errors += 1
        raise RuntimeError(f"judge call failed after {self.max_retries} attempts: {last}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def make_llm(
    model: str = DEFAULT_JUDGE_MODEL,
    *,
    budget: Any = None,
    temperature: float = 0,
    cache_dir: Path | None | bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
    provider: str | None = None,
    reasoning_effort: str | None = None,
) -> OpenRouterLLM:
    """Real judge over OpenRouter. `cache_dir=True` -> eval-results/.llm_cache/, False/None disables it."""
    if cache_dir is True:
        cache = default_root() / ".llm_cache"
    elif cache_dir:
        cache = Path(cache_dir)
    else:
        cache = None
    # the default judge gets its pinned provider and reasoning effort; other models run as OpenRouter routes them
    default = model == DEFAULT_JUDGE_MODEL
    return OpenRouterLLM(
        model=model, budget=budget, temperature=float(temperature), cache_dir=cache, transport=transport,
        provider=provider if provider is not None else (DEFAULT_JUDGE_PROVIDER if default else ""),
        reasoning_effort=reasoning_effort if reasoning_effort is not None else (DEFAULT_JUDGE_REASONING if default else None),
    )


# --- JSON parsing ------------------------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> Any:
    """Parse a JSON object/array from a model reply, tolerating code fences and prose around it."""
    text = (text or "").strip()
    candidates = [text]
    candidates += [m.group(1).strip() for m in _FENCE.finditer(text)]
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = text.find(open_), text.rfind(close)
        if 0 <= i < j:
            candidates.append(text[i : j + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except ValueError:
            continue
    raise ValueError(f"no JSON in reply: {text[:200]!r}")


async def ask_json(llm: LLM, system: str, user: str, validate: Callable[[Any], Any] | None = None) -> Any:
    """Call the judge and parse JSON; `validate` may transform or raise. One retry, then JudgeError."""
    last: Exception | None = None
    for attempt in range(2):
        reply = await llm(system, user if attempt == 0 else user + prompts.JSON_RETRY_SUFFIX)
        try:
            data = parse_json(reply)
            return validate(data) if validate else data
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            last = exc
    raise JudgeError(str(last))
