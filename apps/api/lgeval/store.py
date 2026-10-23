"""Results directory layout: <root>/<experiment>/<run_tag>/{manifest.json, <name>.jsonl, snapshots/<run_id>.json}."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from lgeval.schema import Manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path(os.environ.get("LGEVAL_RESULTS", REPO_ROOT / "eval-results"))  # gitignored

T = TypeVar("T", bound=BaseModel)


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def new_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class ResultStore:
    """One experiment invocation's output directory. Pass an existing `run_tag` (or `open`) to resume."""

    def __init__(self, root: Path, experiment: str, *, run_tag: str | None = None) -> None:
        self.root = Path(root)
        self.experiment = experiment
        self.run_tag = run_tag or new_run_tag()
        self.dir = self.root / experiment / self.run_tag
        self.dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def open(cls, path: Path) -> ResultStore:
        """Store for an existing results dir `<root>/<experiment>/<run_tag>`."""
        path = Path(path)
        return cls(path.parent.parent, path.parent.name, run_tag=path.name)

    # ------------------------------------------------------------ manifest

    def write_manifest(self, manifest: Manifest) -> None:
        if manifest.git_sha is None:
            manifest = manifest.model_copy(update={"git_sha": git_sha()})
        (self.dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    def read_manifest(self) -> Manifest | None:
        p = self.dir / "manifest.json"
        return Manifest.model_validate_json(p.read_text()) if p.exists() else None

    # ------------------------------------------------------------ jsonl

    def path(self, name: str) -> Path:
        return self.dir / (name if name.endswith(".jsonl") else f"{name}.jsonl")

    def append(self, name: str, record: BaseModel) -> None:
        with self.path(name).open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def rows(self, name: str) -> Iterator[dict]:
        """Raw JSON objects of `name`.jsonl (a truncated last line from a crash is skipped)."""
        p = self.path(name)
        if not p.exists():
            return
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def read(self, name: str, model: type[T]) -> list[T]:
        return [model.model_validate(r) for r in self.rows(name)]

    def done_keys(self, name: str, key: Callable[[dict], str]) -> set[str]:
        """Keys of the records already written, for resuming: skip work whose key is in the set."""
        return {key(r) for r in self.rows(name)}

    # ------------------------------------------------------------ snapshots

    def snapshot_path(self, run_id: str) -> Path:
        return self.dir / "snapshots" / f"{run_id}.json"

    def save_snapshot(self, run_id: str, snapshot: dict) -> Path:
        p = self.snapshot_path(run_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snapshot, default=str))
        return p

    def load_snapshot(self, run_id: str) -> dict:
        return json.loads(self.snapshot_path(run_id).read_text())
