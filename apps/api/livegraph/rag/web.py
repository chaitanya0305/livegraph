"""Web research: Exa's search API or OpenRouter's web search plugin (``WEB_PROVIDER``).

Exa: ``POST {EXA_BASE_URL}/search`` with header ``x-api-key`` and body
``{"query", "type": "auto", "numResults": N, "contents": {"highlights": true}}``. Response:
``results: [{url, title, highlights: [str], publishedDate?, author?}]``; the highlights (query-relevant
page excerpts) become the citation content. No LLM call is involved.

OpenRouter request: chat.completions with ``extra_body={"plugins": [{"id": "web", "max_results": N}]}``
(equivalent to the ``:online`` model suffix). Response: the assistant message carries
``annotations: [{"type": "url_citation", "url_citation": {url, title, content?, start_index, end_index}}]``
where ``content`` (page excerpt/highlights) is added by OpenRouter when available.
"""

from __future__ import annotations

import logging

import httpx
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from livegraph.config import settings
from livegraph.rag.models import Document
from livegraph.rag.store import ingest_text

log = logging.getLogger(__name__)

_client: Any | None = None
_exa_transport: httpx.AsyncBaseTransport | None = None


def set_web_client(client: Any | None) -> None:
    """Override the OpenAI-compatible client used for web research (tests). None clears."""
    global _client
    _client = client


def set_exa_transport(transport: httpx.AsyncBaseTransport | None) -> None:
    """Override the httpx transport used for Exa requests (tests, e.g. ``httpx.MockTransport``). None clears."""
    global _exa_transport
    _exa_transport = transport


def web_provider() -> str:
    """The backend ``web_research`` uses: "exa" or "openrouter".

    ``auto`` picks a test override first (Exa transport, then OpenRouter client), then Exa when
    EXA_API_KEY is set, else OpenRouter.
    """
    p = settings.web_provider.strip().lower()
    if p in ("exa", "openrouter"):
        return p
    if _exa_transport is not None:
        return "exa"
    if _client is not None:
        return "openrouter"
    return "exa" if settings.exa_api_key else "openrouter"


def _get_client() -> Any:
    if _client is not None:
        return _client
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key)


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(getattr(obj, "__dict__", {}))


def parse_citations(message: Any) -> list[dict]:
    """Extract ``{url, title, content}`` from url_citation annotations, in order, deduped by url."""
    msg = _as_dict(message)
    body: str = msg.get("content") or ""
    seen: set[str] = set()
    out: list[dict] = []
    for ann in msg.get("annotations") or []:
        ann = _as_dict(ann)
        if ann.get("type") != "url_citation":
            continue
        uc = _as_dict(ann.get("url_citation")) or ann  # tolerate flattened shape
        url = (uc.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        content = (uc.get("content") or "").strip()
        if not content and body:
            s, e = uc.get("start_index"), uc.get("end_index")
            if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(body):
                # Widen to the surrounding paragraph for some context.
                ps = body.rfind("\n\n", 0, s)
                pe = body.find("\n\n", e)
                content = body[ps + 2 if ps >= 0 else 0 : pe if pe >= 0 else len(body)].strip()
            else:
                content = body.strip()
        out.append({"url": url, "title": (uc.get("title") or url).strip(), "content": content})
    return out


def _web_body(max_results: int) -> dict:
    body: dict = {"plugins": [{"id": "web", "max_results": max_results}]}
    if settings.provider_fast:
        body["provider"] = {"order": [settings.provider_fast], "allow_fallbacks": True}
    if not settings.reasoning_fast:
        body["reasoning"] = {"enabled": False}
    return body


def parse_exa_results(data: Any) -> list[dict]:
    """``[{url, title, content}]`` from an Exa /search response, in order, deduped by url.

    Content is the result's highlights joined by blank lines (falling back to ``text`` if present).
    """
    seen: set[str] = set()
    out: list[dict] = []
    for r in _as_dict(data).get("results") or []:
        r = _as_dict(r)
        url = (r.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        highlights = [h.strip() for h in r.get("highlights") or [] if isinstance(h, str) and h.strip()]
        content = "\n\n".join(highlights) or (r.get("text") or "").strip()
        out.append({"url": url, "title": (r.get("title") or url).strip(), "content": content})
    return out


async def _exa_citations(query: str, max_results: int) -> list[dict]:
    body = {"query": query, "type": "auto", "numResults": max_results, "contents": {"highlights": True}}
    async with httpx.AsyncClient(
        base_url=settings.exa_base_url, timeout=settings.exa_timeout, transport=_exa_transport
    ) as client:
        resp = await client.post("/search", json=body, headers={"x-api-key": settings.exa_api_key})
        resp.raise_for_status()
        return parse_exa_results(resp.json())[:max_results]


async def _openrouter_citations(query: str, max_results: int) -> list[dict]:
    resp = await _get_client().chat.completions.create(
        model=settings.model_fast,
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant. Search the web and summarize the most relevant, "
                "factual sources for the query. Cite every source.",
            },
            {"role": "user", "content": query},
        ],
        extra_body=_web_body(max_results),
    )
    choices = getattr(resp, "choices", None) or _as_dict(resp).get("choices") or []
    if not choices:
        return []
    first = choices[0]
    message = getattr(first, "message", None) or _as_dict(first).get("message")
    return parse_citations(message)[:max_results]


async def fetch_citations(query: str, max_results: int = 5) -> list[dict]:
    """Network half of ``web_research``: ``[{url, title, content}]`` for the cited pages (no DB access).

    [] when WEB_SEARCH is off, the chosen provider has no API key (and no test override), or on any error.
    """
    if not settings.web_search or not query.strip():
        return []
    provider = web_provider()
    if provider == "exa":
        if _exa_transport is None and not settings.exa_api_key:
            return []
        fetch = _exa_citations
    else:
        if _client is None and not settings.openrouter_api_key:
            return []
        fetch = _openrouter_citations
    try:
        return await fetch(query, max_results)
    except Exception:  # noqa: BLE001 -- network/API failures must never break a research node
        log.exception("web_research (%s) failed for query %r", provider, query)
        return []


async def ingest_citations(session: AsyncSession, *, run_id: UUID, citations: list[dict]) -> list[Document]:
    """DB half: ingest each citation as Document(origin="web"), deduped by (run, source_url).

    Concurrent callers must serialize this (the engine holds a per-run lock until commit); the partial
    unique index on documents(run_id, source_url) backs that up.
    """
    if not citations:
        return []
    urls = [c["url"] for c in citations]
    existing = {
        d.source_url: d
        for d in (
            await session.execute(select(Document).where(Document.run_id == run_id, Document.source_url.in_(urls)))
        ).scalars()
    }
    docs: list[Document] = []
    for c in citations:
        doc = existing.get(c["url"])
        if doc is None:
            text = f"{c['title']}\n\n{c['content']}".strip() if c["content"] else c["title"]
            doc = await ingest_text(
                session, run_id=run_id, title=c["title"], text=text, source_url=c["url"], origin="web"
            )
            existing[c["url"]] = doc
        docs.append(doc)
    return docs


async def web_research(session: AsyncSession, *, run_id: UUID, query: str, max_results: int = 5) -> list[Document]:
    """Search the web (Exa or OpenRouter, see ``web_provider``) and ingest each cited page as Document(origin="web").

    Returns the documents for the cited URLs (newly ingested ones plus any already ingested
    for this run with the same source_url). No-op ([]) when WEB_SEARCH is off, no API key is
    configured (and no client override), or on any network/API error.
    """
    citations = await fetch_citations(query, max_results)
    return await ingest_citations(session, run_id=run_id, citations=citations)
