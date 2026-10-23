"""Brief text helpers: sentence splitting, citation markers, lexical evidence ranking and URL page fetching."""

from __future__ import annotations

import hashlib
import html
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import httpx

from lgeval.schema import Citation, Evidence, RunRecord

CLAIM_MARK = re.compile(r"\(\s*(C\d+(?:\s*[,;]\s*C\d+)*)\s*\)")
CLAIM_KEY = re.compile(r"C\d+")
URL = re.compile(r"https?://[^\s<>\")\]]+")
MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
NUM_MARK = re.compile(r"\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]")
FOOTNOTE = re.compile(r"\[\^[^\]]+\]")
SKIP_SECTION = re.compile(r"^(open questions?|references|sources|citations|bibliography|further reading)\b", re.I)
REFERENCE_SECTION = re.compile(r"^(references|sources|citations|bibliography|works cited)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[*])")
_WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    "a an the and or of to in on for with by at from as is are was were be been being it its this that these those "
    "which who whom what when where how why than then there their they them he she his her we our you your not no "
    "but if so such can could may might will would should has have had do does did into over under about also more "
    "most other some any all each per via".split()
)


def norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def tokens(s: str) -> list[str]:
    return [w for w in _WORD.findall(s.lower()) if w not in STOP and len(w) > 1]


def brief_sentences(brief: str) -> list[str]:
    """Content sentences of a markdown brief: headings, tables, rules and the "Open questions"/references
    sections are skipped; list items and paragraphs are split into sentences (>= 3 words)."""
    out: list[str] = []
    skipping = False
    para: list[str] = []

    def flush() -> None:
        if para:
            text = " ".join(para)
            for s in _SENT_SPLIT.split(text):
                s = s.strip()
                if len(norm(CLAIM_MARK.sub("", s)).split()) >= 3:
                    out.append(s)
            para.clear()

    for raw in brief.splitlines():
        line = raw.strip()
        heading = re.match(r"^#{1,6}\s*(.*)$", line)
        bold_heading = re.match(r"^\*\*([^*]+)\*\*:?$", line)
        if heading or bold_heading:
            flush()
            title = (heading or bold_heading).group(1).strip().strip("*: ")
            skipping = bool(SKIP_SECTION.match(title))
            continue
        if skipping:
            continue
        if not line or re.match(r"^(\||[-*_]{3,}$)", line):
            flush()
            continue
        item = re.match(r"^(?:[-*+]|\d+[.)])\s+(.*)$", line)
        if item:
            flush()
            para.append(item.group(1).replace("**", ""))
            flush()
            continue
        para.append(line.replace("**", ""))
    flush()
    return out


def claim_keys_in(sentence: str) -> list[str]:
    keys: list[str] = []
    for m in CLAIM_MARK.finditer(sentence):
        keys += [k for k in CLAIM_KEY.findall(m.group(1)) if k not in keys]
    return keys


def urls_in(sentence: str) -> list[str]:
    found = [m.group(2) for m in MD_LINK.finditer(sentence)] + URL.findall(MD_LINK.sub("", sentence))
    return list(dict.fromkeys(u.rstrip(".,;") for u in found))


def strip_markers(text: str) -> str:
    """Normalize every citation style to "[ref]" and drop a trailing reference list, so formatting does not
    reveal which system wrote a brief (blind pairwise)."""
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        title = re.sub(r"^#{1,6}\s*|\*\*|:$", "", raw.strip()).strip()
        if i > 0 and REFERENCE_SECTION.match(title) and len(title) < 40:
            lines = lines[:i]
            break
    text = "\n".join(lines)
    text = MD_LINK.sub(lambda m: f"{m.group(1)} [ref]", text)
    text = URL.sub("[ref]", text)
    text = CLAIM_MARK.sub("[ref]", text)
    text = NUM_MARK.sub("[ref]", text)
    text = FOOTNOTE.sub("[ref]", text)
    text = re.sub(r"(\[ref\][\s,;]*){2,}", "[ref] ", text)
    return text.strip()


def resolve_citations(run: RunRecord) -> list[Citation]:
    """One Citation per content sentence of the brief (uncited ones have no chunk_ids/urls).

    Uses the runner's `RunRecord.citations` when a sentence matches one, else parses inline markers: `(C1)` ->
    the claim's support_chunk_ids, URLs/markdown links -> urls. With no brief, returns run.citations as is.
    """
    if not run.brief.strip():
        return list(run.citations)
    by_norm = {norm(c.sentence): c for c in run.citations if c.sentence.strip()}
    claims = {c.key: c for c in run.claims}
    out: list[Citation] = []
    for s in brief_sentences(run.brief):
        ns = norm(s)
        match = by_norm.get(ns)
        if match is None and ns:
            match = next((c for n, c in by_norm.items() if n and len(n) > 15 and (n in ns or ns in n)), None)
        if match is not None and (match.chunk_ids or match.urls):
            out.append(match.model_copy(update={"sentence": s}))
            continue
        keys = claim_keys_in(s)
        chunk_ids: list[str] = []
        for k in keys:
            for cid in claims[k].support_chunk_ids if k in claims else []:
                if cid not in chunk_ids:
                    chunk_ids.append(cid)
        out.append(Citation(sentence=s, claim_keys=keys, chunk_ids=chunk_ids, urls=urls_in(s)))
    return out


def evidence_lookup(evidence: Iterable[Evidence]) -> dict[str, Evidence]:
    return {e.chunk_id: e for e in evidence}


def find_evidence(chunk_id: str, by_id: dict[str, Evidence]) -> Evidence | None:
    """Exact chunk id, else a unique 8+ char prefix match (agents sometimes cite short ids)."""
    if chunk_id in by_id:
        return by_id[chunk_id]
    if len(chunk_id) >= 8:
        hits = [e for k, e in by_id.items() if k.startswith(chunk_id) or chunk_id.startswith(k)]
        if len(hits) == 1:
            return hits[0]
    return None


def rank_passages(query: str, docs: list[tuple[str, str]], k: int) -> list[tuple[str, str]]:
    """Top-k (id, text) by BM25 against `query` (stdlib; used to keep judge prompts small)."""
    if len(docs) <= k:
        return list(docs)
    q = set(tokens(query))
    toks = [tokens(t) for _, t in docs]
    n = len(docs)
    avg = sum(len(t) for t in toks) / n or 1.0
    df = Counter(w for t in toks for w in set(t))
    scores = []
    for i, t in enumerate(toks):
        tf = Counter(t)
        s = 0.0
        for w in q:
            if tf[w]:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(t) / avg))
        scores.append((s, -i))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in sorted(order)]


def best_window(text: str, query: str, size: int = 1500) -> str:
    """The `size`-char window of a long page with the most query-token overlap."""
    if len(text) <= size:
        return text
    q = set(tokens(query))
    best, best_i = -1, 0
    step = size // 3
    for i in range(0, max(1, len(text) - size + step), step):
        score = sum(1 for w in tokens(text[i : i + size]) if w in q)
        if score > best:
            best, best_i = score, i
    return text[best_i : best_i + size]


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


class PageFetcher:
    """Fetches cited URL pages as plain text (opt-in `--fetch-urls`), cached on disk and size-capped."""

    def __init__(self, cache_dir: Path | None, *, max_bytes: int = 500_000, max_chars: int = 60_000,
                 timeout: float = 15.0, transport: httpx.AsyncBaseTransport | None = None):
        self.cache_dir = cache_dir
        self.max_bytes, self.max_chars, self.timeout = max_bytes, max_chars, timeout
        self.transport = transport
        self._mem: dict[str, str] = {}

    async def fetch(self, url: str) -> str:
        """Page text ('' on any failure or a non-text page)."""
        if url in self._mem:
            return self._mem[url]
        path = None
        if self.cache_dir is not None:
            path = self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.txt"
            if path.exists():
                self._mem[url] = path.read_text()
                return self._mem[url]
        text = ""
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=True,
                                         headers={"User-Agent": "lgeval/0.1 (citation check)"}) as client:
                async with client.stream("GET", url) as r:
                    ctype = r.headers.get("content-type", "")
                    if r.status_code < 400 and ("html" in ctype or "text" in ctype or not ctype):
                        buf = bytearray()
                        async for chunk in r.aiter_bytes():
                            buf += chunk
                            if len(buf) >= self.max_bytes:
                                break
                        raw = bytes(buf[: self.max_bytes]).decode(r.encoding or "utf-8", errors="replace")
                        text = (html_to_text(raw) if "html" in ctype or "<html" in raw[:500].lower() else raw)
                        text = text[: self.max_chars]
        except (httpx.HTTPError, UnicodeError, ValueError):
            text = ""
        self._mem[url] = text
        if path is not None and text:  # failures are not cached on disk (retried next session)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return text
