"""Evidence tools: search_documents, read_source, web_search, fetch_page (opt-in).

Each call opens its own session. Everything an agent sees from the corpus or the web is wrapped as
untrusted ``<source>`` blocks, and every chunk it saw is accumulated in ``deps.retrieved`` (recorded as the
version's retrievals, which is what lets distrust/pin recompute the node). Web results are cached per run
by normalized query: recomputes read the cache and never hit the network; only a first run, a steer or a
Rerun (``deps.force``) may.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import UUID

from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from livegraph import rag
from livegraph.agents.deps import NodeDeps
from livegraph.agents.guard import ToolOut, run_tool, wrap_chunks
from livegraph.config import settings
from livegraph.rag.models import Document, WebCache
from livegraph.rag.schemas import RetrievedChunk

_fallback_lock = asyncio.Lock()  # deps without a scheduler (smoke script)


def _web_lock(deps: NodeDeps) -> asyncio.Lock:
    return getattr(deps.sched, "web_lock", None) or _fallback_lock


WEB_KINDS = ("explore", "target", "challenge")
WEB_CATEGORIES = ("research paper", "news", "pdf", "company", "financial report")
WEB_K = 5  # chunks web_search returns
PER_DOC, PER_DOMAIN = 2, 2  # at most this many of them from one page / one site (WEB_PAGE_TEXT)


def norm_query(q: str, *, since_year: int | None = None, category: str | None = None) -> str:
    """Web-cache key: the normalized query, plus its filters when set (plain queries keep their old keys)."""
    key = " ".join(q.lower().split())
    if since_year:
        key += f"|since={int(since_year)}"
    if category:
        key += f"|cat={category}"
    return key


def domain_of(url: str | None) -> str:
    host = (urlparse(url or "").hostname or "").lower()
    return host.removeprefix("www.")


def diversify(chunks: list[RetrievedChunk], k: int, per_doc: int = PER_DOC, per_domain: int = PER_DOMAIN) -> list[RetrievedChunk]:
    """The best ``k`` chunks (input is best-first) with at most ``per_doc`` per document and ``per_domain`` per
    site, so one long page cannot fill the result; backfills from the rest if the caps leave slots empty."""
    picked: list[RetrievedChunk] = []
    docs: dict[str, int] = {}
    sites: dict[str, int] = {}
    for c in chunks:
        d, site = str(c.document_id), domain_of(c.source_url) or str(c.document_id)
        if docs.get(d, 0) >= per_doc or sites.get(site, 0) >= per_domain:
            continue
        picked.append(c)
        docs[d] = docs.get(d, 0) + 1
        sites[site] = sites.get(site, 0) + 1
        if len(picked) == k:
            return picked
    rest = [c for c in chunks if c not in picked]
    return picked + rest[: k - len(picked)]


async def web_doc_ids(
    deps: NodeDeps, query: str, *, since_year: int | None = None, category: str | None = None
) -> tuple[list[UUID], str]:
    """Document ids for a web query: cache first (unless forced), else the network (if allowed).

    Returns (doc_ids, source) with source in cache|web|offline|none.
    """
    qn = norm_query(query, since_year=since_year, category=category)
    async with deps.session() as s:
        if not deps.force:
            row = await s.scalar(select(WebCache).where(WebCache.run_id == deps.run_id, WebCache.query_norm == qn))
            if row is not None:
                return [UUID(x) for x in row.doc_ids], "cache"
    if not (deps.force or deps.web_live):
        return [], "offline"
    citations = await rag.fetch_citations(query, since_year=since_year, category=category)
    if not citations:
        return [], "none"
    deps.guard()
    async with _web_lock(deps):  # serialize check-then-insert of web docs per run
        async with deps.session() as s:
            docs = await rag.ingest_citations(s, run_id=deps.run_id, citations=citations)
            ids = [str(d.id) for d in docs]
            await s.execute(
                pg_insert(WebCache)
                .values(run_id=deps.run_id, query_norm=qn, doc_ids=ids)
                .on_conflict_do_update(constraint="uq_web_cache_run_query", set_={"doc_ids": ids})
            )
            await s.commit()
    return [d.id for d in docs], "web"


async def search_documents(ctx: RunContext[NodeDeps], query: str, k: int = 6) -> str:
    """Semantic search over this run's documents (uploads, the global library and pages found on the web).

    Args:
        query: what to look for, phrased as a specific question or statement.
        k: how many chunks to return (1-8).
    """

    async def impl() -> ToolOut:
        kk = max(1, min(int(k or 6), settings.search_k_max))
        async with ctx.deps.session() as s:
            chunks = await rag.search(s, run_id=ctx.deps.run_id, query=query, k=kk)
        ctx.deps.add_chunks(chunks, "search")
        if not chunks:
            return ToolOut(text="no matching chunks", summary="0 chunks")
        return ToolOut(text=wrap_chunks(chunks), summary=f"{len(chunks)} chunks", chunks=chunks)

    return await run_tool(ctx, "search_documents", {"query": query, "k": k}, impl)


async def read_source(ctx: RunContext[NodeDeps], chunk_id: str) -> str:
    """Read a chunk in full together with its neighbouring chunks from the same document.

    Args:
        chunk_id: the chunk_id of a <source> you have seen (full UUID, or its first 8 characters).
    """

    async def impl() -> ToolOut:
        deps = ctx.deps
        async with deps.session() as s:
            cid = await rag.resolve_chunk_id(s, run_id=deps.run_id, ref=chunk_id)
            window = await rag.get_chunk_window(s, run_id=deps.run_id, chunk_id=cid) if cid else []
        if not window:
            return ToolOut(
                text=f"error: chunk {chunk_id} is unknown, out of scope or distrusted",
                summary="chunk not available", ok=False, error="chunk not available",
            )
        for c in window:  # keep the ranking score of chunks the agent already found
            prev = deps.retrieved.get(str(c.chunk_id))
            if prev is not None:
                c.score = prev.score
        deps.add_chunks(window, "read")
        main = next((c for c in window if c.chunk_id == cid), window[0])
        extra = f" (+{len(window) - 1} neighbours)" if len(window) > 1 else ""
        return ToolOut(text=wrap_chunks(window), summary=f"read {main.document_title}{extra}", chunks=window)

    return await run_tool(ctx, "read_source", {"chunk_id": chunk_id}, impl)


async def web_search(
    ctx: RunContext[NodeDeps],
    query: str,
    kind: Literal["explore", "target", "challenge"] = "explore",
    since_year: int | None = None,
    category: Literal["research paper", "news", "pdf", "company", "financial report"] | None = None,
) -> str:
    """Search the web; the pages found are added to the run's documents and their most relevant passages returned.

    Args:
        query: a short natural-language question or description of the page you want (not a keyword list).
        kind: explore (map the topic), target (fill one specific gap) or challenge (look for criticism,
            limitations or evidence against the current answer).
        since_year: only pages published in or after this year (for recent developments).
        category: restrict to one kind of source, e.g. "research paper" or "news".
    """
    kind = kind if kind in WEB_KINDS else "explore"
    category = category if category in WEB_CATEGORIES else None
    since_year = int(since_year) if since_year else None

    async def impl() -> ToolOut:
        deps = ctx.deps
        deps.query_kinds[kind] = deps.query_kinds.get(kind, 0) + 1
        doc_ids, source = await web_doc_ids(deps, query, since_year=since_year, category=category)
        if not doc_ids:
            reason = {
                "offline": "web search is only available on a fresh run or a rerun; use search_documents",
                "none": "no web results (web search may be disabled); use search_documents",
            }.get(source, "no web results")
            return ToolOut(text=reason, summary="no web results")
        async with deps.session() as s:
            if settings.web_page_text:  # whole pages: rank across them, then spread over pages and sites
                found = await rag.search(s, run_id=deps.run_id, query=query, k=WEB_K * 3, document_ids=doc_ids)
                chunks = diversify(found, WEB_K)
            else:
                chunks = await rag.search(s, run_id=deps.run_id, query=query, k=WEB_K, document_ids=doc_ids)
        deps.add_chunks(chunks, "web")
        tag = " (cached)" if source == "cache" else ""
        if not chunks:
            return ToolOut(text="the web pages found are not usable (distrusted or empty)", summary=f"0 chunks{tag}")
        return ToolOut(
            text=wrap_chunks(chunks), summary=f"{len(doc_ids)} pages, {len(chunks)} chunks{tag}", chunks=chunks
        )

    args: dict[str, Any] = {"query": query}
    if kind != "explore":  # only non-defaults: a plain query keeps its one-line trace summary
        args["kind"] = kind
    if since_year:
        args["since_year"] = since_year
    if category:
        args["category"] = category
    return await run_tool(ctx, "web_search", args, impl)


# ---------------------------------------------------------------- fetch_page (FETCH_PAGE=true only)

_TAGS = re.compile(r"<[^>]+>")
_DROP = re.compile(r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>", re.S | re.I)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_BLOCK = re.compile(r"</?(p|div|br|li|h[1-6]|tr|section|article)[^>]*>", re.I)


def html_to_text(raw: str) -> tuple[str, str]:
    m = _TITLE.search(raw)
    title = html.unescape(" ".join(m.group(1).split())) if m else ""
    body = _BLOCK.sub("\n\n", _DROP.sub(" ", raw))
    text = html.unescape(_TAGS.sub(" ", body))
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return title, "\n\n".join(p for p in paras if p)


def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


async def _download(url: str) -> tuple[str, str]:
    import httpx

    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("only http(s) URLs can be fetched")
    if not await asyncio.to_thread(_public_host, p.hostname):
        raise ValueError("URL host is not public")
    async with httpx.AsyncClient(timeout=settings.fetch_timeout, follow_redirects=False) as client:
        async with client.stream("GET", url, headers={"User-Agent": "LiveGraph/0.1"}) as resp:
            if resp.is_redirect:
                raise ValueError(f"redirected to {resp.headers.get('location')}; fetch that URL instead")
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "").lower()
            if not any(t in ctype for t in ("text/html", "text/plain", "text/markdown", "application/xhtml")):
                raise ValueError(f"unsupported content type {ctype or 'unknown'}")
            buf = bytearray()
            async for part in resp.aiter_bytes():
                buf.extend(part)
                if len(buf) >= settings.fetch_max_bytes:
                    break
    raw = bytes(buf[: settings.fetch_max_bytes]).decode(resp.encoding or "utf-8", errors="replace")
    if "html" in ctype:
        return html_to_text(raw)
    return "", raw


async def fetch_page(ctx: RunContext[NodeDeps], url: str) -> str:
    """Fetch a web page, add it to the run's documents and return its most relevant chunks.

    Args:
        url: an http(s) URL, typically one cited by a web_search result.
    """

    async def impl() -> ToolOut:
        deps = ctx.deps
        async with deps.session() as s:
            doc = await s.scalar(select(Document).where(Document.run_id == deps.run_id, Document.source_url == url))
        if doc is None or deps.force:
            if doc is None and not (deps.force or deps.web_live):
                return ToolOut(text="pages can only be fetched on a fresh run or a rerun", summary="offline")
            if doc is None:
                title, text = await _download(url)
                if not text.strip():
                    return ToolOut(text="error: the page has no extractable text", summary="empty page", ok=False,
                                   error="empty page")
                deps.guard()
                async with _web_lock(deps):
                    async with deps.session() as s:
                        doc = await s.scalar(
                            select(Document).where(Document.run_id == deps.run_id, Document.source_url == url)
                        )
                        if doc is None:
                            doc = await rag.ingest_text(
                                s, run_id=deps.run_id, title=title or url, text=text, source_url=url, origin="web"
                            )
                            await s.commit()
        query = (deps.ctx.config.get("question") if deps.ctx else None) or (deps.ctx.goal if deps.ctx else url)
        async with deps.session() as s:
            chunks = await rag.search(s, run_id=deps.run_id, query=query, k=4, document_ids=[doc.id])
        deps.add_chunks(chunks, "web")
        return ToolOut(text=wrap_chunks(chunks) or "the page is distrusted", summary=f"{len(chunks)} chunks",
                       chunks=chunks)

    return await run_tool(ctx, "fetch_page", {"url": url}, impl)


async def _only_if_fetch_enabled(ctx: RunContext[NodeDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    return tool_def if settings.fetch_page else None


async def _only_if_plan_web(ctx: RunContext[NodeDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    return tool_def if settings.plan_web_search else None


SEARCH = Tool(search_documents, takes_ctx=True)
READ = Tool(read_source, takes_ctx=True)
WEB = Tool(web_search, takes_ctx=True)
PLAN_WEB = Tool(web_search, takes_ctx=True, prepare=_only_if_plan_web)  # the planner's scouting (PLAN_WEB_SEARCH)
FETCH = Tool(fetch_page, takes_ctx=True, prepare=_only_if_fetch_enabled)
