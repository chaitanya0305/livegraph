from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from lgeval.datasets import load_questions
from lgeval.datasets.frames import _links, load_frames


def test_custom_set() -> None:
    qs = load_questions("custom")
    assert len(qs) >= 25
    assert len({q.id for q in qs}) == len(qs)
    assert all(q.dataset == "custom" and q.question.endswith("?") and q.gold_facts == [] for q in qs)


def test_sampling_is_seeded_and_ordered() -> None:
    a = load_questions("custom", limit=10, seed=0)
    assert [q.id for q in a] == [q.id for q in load_questions("custom", limit=10, seed=0)]
    assert [q.id for q in a] == sorted(q.id for q in a) and len(a) == 10
    assert [q.id for q in a] != [q.id for q in load_questions("custom", limit=10, seed=1)]
    assert len(load_questions("custom", limit=1000)) == len(load_questions("custom"))


def test_unknown_dataset() -> None:
    with pytest.raises(ValueError):
        load_questions("nope")


def _rows_handler(total: int, page_sizes: list[int]):
    calls: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        off = int(req.url.params["offset"])
        n = min(int(req.url.params["length"]), total - off)
        rows = [{"row_idx": i, "row": {
            "Prompt": f"Question {i}?", "Answer": f"Answer {i}", "reasoning_types": "Numerical reasoning",
            "wiki_links": f"['https://en.wikipedia.org/wiki/A{i}', 'https://en.wikipedia.org/wiki/B{i}']",
        }} for i in range(off, off + n)]
        return httpx.Response(200, json={"rows": rows, "num_rows_total": total})

    return httpx.MockTransport(handler), calls


def test_frames_fetch_and_cache(tmp_path: Path) -> None:
    transport, calls = _rows_handler(total=150, page_sizes=[100, 50])
    cache = tmp_path / ".datasets" / "frames.jsonl"
    qs = load_frames(cache=cache, transport=transport)
    assert len(qs) == 150 and len(calls) == 2
    q = qs[7]
    assert q.id == "frames-0007" and q.question == "Question 7?" and q.gold_answer == "Answer 7"
    assert q.sources == ["https://en.wikipedia.org/wiki/A7", "https://en.wikipedia.org/wiki/B7"]
    assert q.dataset == "frames" and q.meta["reasoning_types"] == "Numerical reasoning"
    # second load is served from the cache, even offline
    assert len(load_frames(cache=cache, offline=True)) == 150 and len(calls) == 2


def test_frames_offline_without_cache(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_frames(cache=tmp_path / "none.jsonl", offline=True)


def test_links_formats() -> None:
    assert _links(["a", "b"]) == ["a", "b"]
    assert _links("['a', 'b']") == ["a", "b"]
    assert _links("[a, b") == ["a", "b"]
    assert _links(None) == []
