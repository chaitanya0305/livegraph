"""Web research via OpenRouter's web search plugin.

Request: chat.completions with ``extra_body={"plugins": [{"id": "web", "max_results": N}]}``
(equivalent to the ``:online`` model suffix). Response: the assistant message carries
``annotations: [{"type": "url_citation", "url_citation": {url, title, content?, start_index, end_index}}]``
where ``content`` (page excerpt/highlights) is added by OpenRouter when available.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from livegraph.config import settings
from livegraph.rag.models import Document
from livegraph.rag.store import ingest_text

log = logging.getLogger(__name__)

_client: Any | None = None


def set_web_client(client: Any | None) -> None:
    """Override the OpenAI-compatible client used for web research (tests). None clears."""
    global _client
    _client = client


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


async def web_research(session: AsyncSession, *, run_id: UUID, query: str, max_results: int = 5) -> list[Document]:
    """Search the web through OpenRouter and ingest each cited page as Document(origin="web").

    Returns the documents for the cited URLs (newly ingested ones plus any already ingested
    for this run with the same source_url). No-op ([]) when WEB_SEARCH is off, no API key is
    configured (and no client override), or on any network/API error.
    """
    if not settings.web_search or not query.strip():
        return []
    if _client is None and not settings.openrouter_api_key:
        return []
    try:
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
            extra_body={"plugins": [{"id": "web", "max_results": max_results}]},
        )
        choices = getattr(resp, "choices", None) or _as_dict(resp).get("choices") or []
        if not choices:
            return []
        first = choices[0]
        message = getattr(first, "message", None) or _as_dict(first).get("message")
        citations = parse_citations(message)[:max_results]
    except Exception:  # noqa: BLE001 -- network/API failures must never break a research node
        log.exception("web_research failed for query %r", query)
        return []

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
