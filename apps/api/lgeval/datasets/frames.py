"""FRAMES (google/frames-benchmark) via the Hugging Face datasets-server rows API, cached as Question JSONL."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import httpx

from lgeval.schema import Question
from lgeval.store import DEFAULT_ROOT

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "google/frames-benchmark"
PAGE = 100
CACHE = DEFAULT_ROOT / ".datasets" / "frames.jsonl"


def _links(value: Any) -> list[str]:
    """wiki_links arrives as a list or as its Python repr ("['https://...', ...]")."""
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [x.strip(" '\"") for x in value.strip("[]").split(",") if x.strip(" '\"")]
        return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]
    return []


def row_to_question(row_idx: int, row: dict) -> Question:
    meta = {k: row[k] for k in ("reasoning_types",) if row.get(k) is not None}
    return Question(
        id=f"frames-{row_idx:04d}",
        question=str(row["Prompt"]),
        dataset="frames",
        gold_answer=str(row["Answer"]) if row.get("Answer") is not None else None,
        sources=_links(row.get("wiki_links")),
        meta=meta,
    )


def fetch_rows(client: httpx.Client) -> list[Question]:
    out: list[Question] = []
    offset, total = 0, None
    while total is None or offset < total:
        r = client.get(ROWS_URL, params={"dataset": DATASET, "config": "default", "split": "test",
                                         "offset": offset, "length": PAGE})
        r.raise_for_status()
        body = r.json()
        total = int(body.get("num_rows_total", 0))
        rows = body.get("rows", [])
        if not rows:
            break
        out += [row_to_question(int(x["row_idx"]), x["row"]) for x in rows]
        offset += len(rows)
    return out


def load_frames(
    *, cache: Path | None = None, offline: bool = False, transport: httpx.BaseTransport | None = None
) -> list[Question]:
    """All FRAMES test questions; downloaded once and cached under eval-results/.datasets/."""
    cache = cache or CACHE
    if cache.exists():
        with cache.open(encoding="utf-8") as f:
            return [Question.model_validate(json.loads(line)) for line in f if line.strip()]
    if offline:
        raise FileNotFoundError(f"FRAMES not cached at {cache} and offline=True")
    with httpx.Client(timeout=30.0, transport=transport) as client:
        qs = fetch_rows(client)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text("".join(q.model_dump_json() + "\n" for q in qs), encoding="utf-8")
    tmp.replace(cache)
    return qs
