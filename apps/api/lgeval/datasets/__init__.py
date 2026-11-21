"""Question sets: "custom" (datasets/custom.jsonl) and "frames" (google/frames-benchmark, cached)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from lgeval.schema import Question

DATA_DIR = Path(__file__).resolve().parent
CUSTOM_PATH = DATA_DIR / "custom.jsonl"
DATASETS = ("custom", "frames")


def read_jsonl(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return [Question.model_validate(json.loads(line)) for line in f if line.strip()]


def sample(questions: list[Question], limit: int | None, seed: int = 0) -> list[Question]:
    """`limit` questions drawn with Random(seed), kept in dataset order; all when limit is None."""
    if limit is None or limit >= len(questions):
        return list(questions)
    idx = sorted(random.Random(seed).sample(range(len(questions)), limit))
    return [questions[i] for i in idx]


def load_questions(name: str, *, limit: int | None = None, seed: int = 0, offline: bool = False) -> list[Question]:
    """Load a question set by name ("custom", "frames", or a path to a Question JSONL file)."""
    if name == "custom":
        qs = read_jsonl(CUSTOM_PATH)
    elif name == "frames":
        from lgeval.datasets.frames import load_frames

        qs = load_frames(offline=offline)
    elif Path(name).suffix == ".jsonl" and Path(name).exists():
        qs = read_jsonl(Path(name))
    else:
        raise ValueError(f"unknown dataset {name!r} (expected one of {', '.join(DATASETS)} or a .jsonl path)")
    return sample(qs, limit, seed)
