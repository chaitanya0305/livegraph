"""Hashes of the incremental engine.

- ``input_hash``: sha256 over canonical JSON of everything that determines a node's output (reuse key).
- ``output_hash``: sha256 over a version's content; downstream input hashes reference their inputs by it
  (``"<key>:<output_hash or version id>"``), so a recompute that produced the same content stops the
  propagation there (early cutoff), and a restore of an equal version reuses everything below.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def input_ref(key: str, version_id: Any, output_hash: str | None) -> str:
    """How a node's input hash names one of its inputs: by content when known, else by version id."""
    return f"{key}:{output_hash or version_id}"


def input_hash(
    *,
    node_type: str,
    config: dict[str, Any],
    goal: str,
    parent_version_ids: list[str],
    steer_texts: list[str],
    documents: list[tuple[str, str]] | None = None,
    title: str | None = None,
    question: str | None = None,
    ancestors: list[str] | None = None,
) -> str:
    """``parent_version_ids`` are ``input_ref`` strings of the direct parents; ``ancestors`` the ``input_ref``s of
    the transitive ancestors the node can read beyond them (research/claims evidence, context tools)."""
    payload: dict[str, Any] = {
        "type": node_type,
        "config": config or {},
        "goal": goal,
        "parents": sorted(parent_version_ids),
        "steer": list(steer_texts),
    }
    if documents is not None:
        payload["documents"] = sorted([list(d) for d in documents])
    if title is not None:
        payload["title"] = title
    if question is not None:
        payload["question"] = question
    if ancestors:
        payload["ancestors"] = sorted(set(ancestors))
    return _sha(payload)


def normalize_text(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in (text or "").strip().splitlines())


def output_hash(output: dict[str, Any], output_text: str, retrievals: list[tuple[str, bool]] | None = None) -> str:
    """Content hash of a version: output JSON, whitespace-normalized text, sorted (chunk_id, cited) retrievals."""
    return _sha({
        "output": output or {},
        "text": normalize_text(output_text),
        "retrievals": sorted([str(c), bool(cited)] for c, cited in (retrievals or [])),
    })
