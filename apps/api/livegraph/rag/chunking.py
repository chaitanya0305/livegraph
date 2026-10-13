"""Text extraction and paragraph/sentence-aware chunking."""

from __future__ import annotations

import io
import re

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, size: int) -> list[str]:
    """Split a long paragraph into sentence-ish pieces each <= size."""
    pieces: list[str] = []
    for sent in _SENT_RE.split(text):
        sent = sent.strip()
        if not sent:
            continue
        while len(sent) > size:
            cut = sent.rfind(" ", 0, size)
            if cut <= size // 2:
                cut = size
            pieces.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            pieces.append(sent)
    return pieces


def _tail(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    t = text[-n:]
    sp = t.find(" ")
    return t[sp + 1 :] if 0 <= sp < n // 2 else t


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    units: list[str] = []
    for p in paragraphs:
        if not p:
            continue
        units.extend([p] if len(p) <= size else _split_long(p, size))

    chunks: list[str] = []
    cur = ""
    for u in units:
        if not cur:
            cur = u
        elif len(cur) + 2 + len(u) <= size:
            cur = f"{cur}\n\n{u}"
        else:
            chunks.append(cur)
            prefix = _tail(cur, overlap) if overlap else ""
            cur = f"{prefix} {u}".strip() if prefix and len(prefix) + 1 + len(u) <= size else u
    if cur:
        chunks.append(cur)
    return chunks


def extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)


def decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        return extract_pdf_text(data)
    return decode_text(data)
