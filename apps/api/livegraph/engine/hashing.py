"""input_hash: sha256 over canonical JSON of everything that determines a node's output."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def input_hash(
    *,
    node_type: str,
    config: dict[str, Any],
    goal: str,
    parent_version_ids: list[str],
    steer_texts: list[str],
    documents: list[tuple[str, str]] | None = None,
) -> str:
    payload = {
        "type": node_type,
        "config": config or {},
        "goal": goal,
        "parents": sorted(parent_version_ids),
        "steer": list(steer_texts),
    }
    if documents is not None:
        payload["documents"] = sorted([list(d) for d in documents])
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()
