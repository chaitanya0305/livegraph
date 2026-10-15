"""Context store: every stored tool result of an agent attempt is a ``context_items`` row addressed by ``ref``.

- ``put`` (called by ``guard.run_tool`` in its own session) writes the full text plus a deterministic one-line
  ``summary`` (``tool(args) -> summary [chunk_id=...]``) and registers the compaction stub for the call on the
  deps (``deps.refs`` / ``deps.stubs``), so the history processor never touches the database.
- ``backfill`` stamps the attempt's rows with its version id once that version persists (engine phase 4).
  Rows of superseded attempts keep ``node_version_id`` NULL and are unreadable to everyone else.
- ``readable_versions`` / ``load_readable`` apply the scope rule: an agent may read its own attempt and the
  current versions of its transitive active ancestors (``deps.readable``), never anything downstream or
  sideways, so what it reads is always covered by its input hash.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from livegraph.db.models import ContextItem, Node
from livegraph.trace import summarize_args, truncate

log = logging.getLogger(__name__)

PAGE_CHARS = 4000  # recall / read_agent_output page size


def new_ref() -> str:
    return "ctx_" + uuid.uuid4().hex[:8]


def part_key(tool_call_id: str | None, text: str) -> str:
    """Key of one tool return in a message history: the provider call id + a digest of the returned text
    (provider ids are not unique across rounds, the pair is)."""
    return f"{tool_call_id or ''}:{hashlib.sha1(text.encode()).hexdigest()[:16]}"


def summary_line(tool: str, args: dict[str, Any], summary: str, chunk_ids: list[str]) -> str:
    cites = "".join(f" [chunk_id={c}]" for c in chunk_ids)  # full ids: still citable after compaction
    return f"{tool}({summarize_args(args)}) -> {truncate(summary, 200)}{cites}"


def stub_for(ref: str, summary: str) -> str:
    return f"[{ref}] {summary} (recall('{ref}') for full text)"


async def put(deps: Any, *, ref: str, tool: str, args: dict[str, Any], out: Any, kind: str = "tool_result") -> bool:
    """Store one tool result; False (and nothing stored) when the write fails (never fails the tool call)."""
    chunk_ids = [str(c.chunk_id) for c in out.chunks]
    summary = summary_line(tool, args, out.summary, chunk_ids)
    try:
        async with deps.session() as s:
            s.add(ContextItem(
                run_id=deps.run_id, ref=ref, node_id=deps.node_id, attempt_id=deps.attempt_id, agent=deps.agent,
                parent_ref=deps.parent_ref, kind=kind, tool=tool, args=_jsonable(args), ok=out.ok, text=out.text,
                summary=summary, chunk_ids=chunk_ids, tokens_est=len(out.text) // 4,
            ))
            await s.commit()
    except Exception:  # noqa: BLE001 - the store is best-effort; the model already has the text
        log.warning("context_items write failed for %s/%s", deps.node_key, tool, exc_info=True)
        return False
    deps.stubs[ref] = stub_for(ref, summary)
    return True


async def backfill(s: AsyncSession, run_id: UUID, attempt_id: str, version_id: UUID) -> None:
    """The attempt's rows now belong to its persisted version (call before the version becomes visible)."""
    await s.execute(
        update(ContextItem)
        .where(ContextItem.run_id == run_id, ContextItem.attempt_id == attempt_id)
        .values(node_version_id=version_id)
    )
    await s.commit()


def scope_filter(deps: Any):
    """SQL condition: rows this attempt may read (its own + its readable ancestors' current versions)."""
    versions = list(deps.readable.values())
    cond = ContextItem.attempt_id == deps.attempt_id
    return or_(cond, ContextItem.node_version_id.in_(versions)) if versions else cond


async def load_readable(s: AsyncSession, deps: Any, ref: str) -> tuple[ContextItem | None, str | None, str]:
    """(row, owner node key, error). ``error`` is "" when the row is readable."""
    row = await s.get(ContextItem, (deps.run_id, ref.strip()))
    if row is None:
        return None, None, f"error: unknown ref {ref}"
    key = deps.node_key if row.node_id == deps.node_id else await s.scalar(select(Node.key).where(Node.id == row.node_id))
    if row.attempt_id == deps.attempt_id or (row.node_version_id is not None and row.node_version_id in deps.readable.values()):
        return row, key, ""
    return None, key, f"not readable: {key} is not upstream of you"


def page(text: str, offset: int) -> tuple[str, int | None]:
    """One PAGE_CHARS page of ``text`` from ``offset`` and the next offset (None at the end)."""
    offset = max(0, int(offset or 0))
    chunk = text[offset : offset + PAGE_CHARS]
    nxt = offset + PAGE_CHARS
    return chunk, (nxt if nxt < len(text) else None)


def _jsonable(args: dict[str, Any]) -> dict[str, Any]:
    return {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) for k, v in args.items()}
