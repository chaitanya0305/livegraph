"""Embedding backends: OpenRouter (OpenAI-compatible /embeddings) and an offline FakeEmbedder."""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol, runtime_checkable

from livegraph.config import settings

EMBED_BATCH = 64


@runtime_checkable
class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenRouterEmbedder:
    """Calls OpenRouter's OpenAI-compatible embeddings endpoint in batches of <= 64."""

    def __init__(self, *, model: str | None = None, dim: int | None = None, client=None) -> None:
        self.model = model or settings.embed_model
        self.dim = dim or settings.embed_dim
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key)
        self.client = client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = [t if t.strip() else " " for t in texts[i : i + EMBED_BATCH]]
            resp = await self.client.embeddings.create(model=self.model, input=batch)
            data = sorted(resp.data, key=lambda d: d.index)
            out.extend(list(d.embedding) for d in data)
        return out


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FakeEmbedder:
    """Deterministic hashing-trick bag-of-words embedder (unigrams + bigrams), L2-normalized.

    Texts sharing words get high cosine similarity, which is enough for offline tests.
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embed_dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = _tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = hashlib.blake2b(f.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            v[idx] += sign
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            # Avoid zero vectors (cosine distance undefined): fixed unit vector.
            v[0] = 1.0
            return v
        return [x / norm for x in v]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


_override: Embedder | None = None
_default: Embedder | None = None


def set_embedder(embedder: Embedder | None) -> None:
    """Override the process-wide embedder (tests). Pass None to clear."""
    global _override
    _override = embedder


def get_embedder() -> Embedder:
    global _default
    if _override is not None:
        return _override
    if _default is None:
        if os.environ.get("LIVEGRAPH_FAKE_EMBED") == "1" or not settings.openrouter_api_key:
            _default = FakeEmbedder()
        else:
            _default = OpenRouterEmbedder()
    return _default
