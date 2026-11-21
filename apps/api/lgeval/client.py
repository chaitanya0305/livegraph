"""Async REST client for the LiveGraph API (docs/CONTRACT.md) plus helpers that read a RunSnapshot."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx

from lgeval.schema import Citation, ClaimOut, Evidence, TokenUsage

SETTLED_RUN = {"done", "failed", "cancelled", "waiting_input"}
BUSY_NODE = {"running", "pending", "stale"}


class LiveGraphClient:
    """Thin wrapper over `/api/...`. Every method raises `httpx.HTTPStatusError` on a non-2xx response."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=f"{self.base_url}/api", transport=transport, timeout=timeout)

    async def __aenter__(self) -> LiveGraphClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _req(self, method: str, path: str, **kw: Any) -> Any:
        r = await self._http.request(method, path, **kw)
        r.raise_for_status()
        return r.json() if r.content else None

    # ------------------------------------------------------------ reads

    async def health(self) -> bool:
        try:
            r = await self._http.get("/health")
        except httpx.HTTPError:
            return False
        return r.status_code == 200 and bool(r.json().get("ok"))

    async def create_run(self, goal: str, document_ids: list[str] | None = None) -> dict:
        return await self._req("POST", "/runs", json={"goal": goal, "document_ids": document_ids or []})

    async def snapshot(self, run_id: str) -> dict:
        return await self._req("GET", f"/runs/{run_id}")

    async def events(self, run_id: str, since: int = 0) -> list[dict]:
        return await self._req("GET", f"/runs/{run_id}/events", params={"since": since})

    async def versions(self, node_id: str) -> list[dict]:
        return await self._req("GET", f"/nodes/{node_id}/versions")

    async def all_versions(self, snapshot: dict) -> dict[str, list[dict]]:
        """Every version (with retrievals) of every node in the snapshot, keyed by node id."""
        ids = [n["id"] for n in snapshot.get("nodes", [])]
        lists = await asyncio.gather(*(self.versions(i) for i in ids))
        return dict(zip(ids, lists, strict=True))

    async def impact(self, node_id: str, action: str) -> dict:
        return await self._req("GET", f"/nodes/{node_id}/impact", params={"action": action})

    # ------------------------------------------------------------ mutations

    async def steer(self, node_id: str, text: str) -> None:
        await self._req("POST", f"/nodes/{node_id}/steer", json={"text": text})

    async def edit(self, node_id: str, output_text: str) -> dict:
        return await self._req("POST", f"/nodes/{node_id}/edit", json={"output_text": output_text})

    async def rerun(self, node_id: str) -> None:
        await self._req("POST", f"/nodes/{node_id}/rerun")

    async def restore(self, node_id: str, version_id: str) -> None:
        await self._req("POST", f"/nodes/{node_id}/restore", json={"version_id": version_id})

    async def veto(self, node_id: str) -> None:
        await self._req("DELETE", f"/nodes/{node_id}")

    async def patch_node(self, node_id: str, **fields: Any) -> dict:
        return await self._req("PATCH", f"/nodes/{node_id}", json=fields)

    async def set_trust(self, document_id: str, trust: str) -> dict:
        return await self._req("PATCH", f"/documents/{document_id}", json={"trust": trust})

    async def upload(self, filename: str, data: bytes, run_id: str | None = None) -> dict:
        form = {"run_id": run_id} if run_id else None
        return await self._req("POST", "/documents", files={"file": (filename, data)}, data=form)

    async def answer(self, node_id: str, answer: str) -> None:
        await self._req("POST", f"/nodes/{node_id}/input", json={"answer": answer})

    async def cancel(self, run_id: str) -> None:
        await self._req("POST", f"/runs/{run_id}/cancel")

    # ------------------------------------------------------------ waiting

    async def wait_settled(
        self, run_id: str, *, timeout: float = 900, poll: float = 2.0, since_seq: int = 0
    ) -> dict:
        """Poll until the run is quiet and return its RunSnapshot.

        Quiet = run.status in {done, failed, cancelled, waiting_input}, no live node is running/pending/stale,
        and last_seq did not move across one poll. With `since_seq` (the last_seq seen before a mutation)
        the run must also have logged an event after it, unless nothing happens within a grace period of
        max(10s, 3 polls) — some mutations (an identical rerun hitting the cache) may emit next to nothing.
        Raises TimeoutError after `timeout` seconds.
        """
        start = time.monotonic()
        grace = max(10.0, 3 * poll)
        prev_seq: int | None = None
        while True:
            snap = await self.snapshot(run_id)
            seq = int(snap.get("last_seq", 0))
            moved = seq > since_seq or time.monotonic() - start >= grace
            if moved and is_quiet(snap) and seq == prev_seq:
                return snap
            prev_seq = seq
            if time.monotonic() - start >= timeout:
                raise TimeoutError(f"run {run_id} not settled after {timeout:.0f}s (status={snap['run']['status']})")
            await asyncio.sleep(poll)


def is_quiet(snapshot: dict) -> bool:
    """Run status is terminal-ish and no live node has work left.

    Nodes downstream of a failed node or an unanswered human_input gate stay `pending` forever, so for a
    failed/cancelled/waiting_input run only running/stale nodes count as work left."""
    status = snapshot["run"]["status"]
    if status not in SETTLED_RUN:
        return False
    busy = BUSY_NODE if status == "done" else BUSY_NODE - {"pending"}
    return not any(n["status"] in busy for n in snapshot.get("nodes", []) if not n.get("removed"))


# ---------------------------------------------------------------- snapshot helpers


def _live_nodes(snapshot: dict, node_type: str) -> list[dict]:
    return [n for n in snapshot.get("nodes", []) if n["type"] == node_type and not n.get("removed")]


def _current_version(snapshot: dict, node_type: str) -> dict | None:
    """Current version of the last live node of `node_type` that has one."""
    versions = snapshot.get("versions", {})
    for n in reversed(_live_nodes(snapshot, node_type)):
        v = versions.get(n["id"])
        if v is not None:
            return v
    return None


def brief_of(snapshot: dict) -> str:
    """The editor node's current output_text ("" if none)."""
    v = _current_version(snapshot, "editor")
    if v is None:
        return ""
    return v.get("output_text") or (v.get("output") or {}).get("markdown", "") or ""


def usage_of(
    versions_by_node: dict[str, list[dict]], *, after_seq_versions: set[str] | None = None
) -> dict[str, TokenUsage]:
    """Sum tokens_in/out per model over ALL given versions (not only current ones).

    `after_seq_versions`: when given, only versions whose id is in this set count (e.g. the versions a
    mutation created). Versions without tokens (edits, restores) are skipped; tokens without a model are
    booked under "unknown". Reused versions are not new versions, so they never appear twice here.
    """
    out: dict[str, TokenUsage] = {}
    for versions in versions_by_node.values():
        for v in versions:
            if after_seq_versions is not None and str(v.get("id")) not in after_seq_versions:
                continue
            tin, tout = int(v.get("tokens_in") or 0), int(v.get("tokens_out") or 0)
            if not tin and not tout:
                continue
            model = v.get("model") or "unknown"
            out[model] = out.get(model, TokenUsage()) + TokenUsage(tokens_in=tin, tokens_out=tout)
    return out


_REF = re.compile(r"\bC(\d+)\b")
_PAREN = re.compile(r"\(([^()]*\bC\d+\b[^()]*)\)")
_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BULLET = re.compile(r"^\s*(?:[-*+>]|\d+[.)])\s+")
_MD = re.compile(r"\*\*|__|`")
_ONLY_REFS = re.compile(r"^[\s(),;C\d\[\].]*$")
_ABBREV = re.compile(r"(?:\b(?:al|e\.g|i\.e|vs|etc|approx|Dr|Mr|Ms|Inc|Fig|No|St)|\b[A-Z](?:\.[A-Z])*)\.$")


def claim_refs(text: str) -> list[str]:
    """Claim keys cited in `text` as (C1) / (C1, C3), in order of first appearance."""
    keys: list[str] = []
    for m in _PAREN.finditer(text):
        for n in _REF.findall(m.group(1)):
            if f"C{n}" not in keys:
                keys.append(f"C{n}")
    return keys


def brief_sentences(markdown: str) -> list[str]:
    """Content sentences of a markdown brief: headings and blank lines dropped, bullets and emphasis
    stripped, lines split at sentence ends. A fragment that is only claim refs ("(C2).") is merged into
    the sentence before it. A table row is one unit (cells joined by " | "), and a cell holding just a
    claim key is rewritten as "(Cn)" so it counts as a citation."""
    out: list[str] = []
    lines = [x.strip() for x in markdown.splitlines()]
    for i, line in enumerate(lines):
        if not line or line.startswith("#") or set(line) <= set("-*_=|: "):
            continue
        if line.startswith("|"):
            # table row = one unit; header rows (followed by |---|) dropped; a bare "C3" cell is a citation
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt.startswith("|") and set(nxt) <= set("-|: "):
                continue
            cells = [_MD.sub("", c).strip() for c in line.strip("|").split("|")]
            cells = [f"({c})" if re.fullmatch(r"C\d+", c) else c for c in cells if c]
            if cells:
                out.append(" | ".join(cells))
            continue
        line = _MD.sub("", _BULLET.sub("", line)).strip()
        start = len(out)
        for part in _SPLIT.split(line):
            part = part.strip()
            if not part:
                continue
            if len(out) > start and _ABBREV.search(out[-1]):
                out[-1] = f"{out[-1]} {part}"
            elif out and _ONLY_REFS.match(part) and _REF.search(part):
                out[-1] = f"{out[-1]} {part}"
            else:
                out.append(part)
    return out


def _chunk_index(snapshot: dict, versions_by_node: dict[str, list[dict]]) -> tuple[dict[str, dict], list[str]]:
    """chunk_id -> retrieval row from every version; and the ids seen on current versions (in order)."""
    index: dict[str, dict] = {}
    current: list[str] = []
    for v in snapshot.get("versions", {}).values():
        for r in v.get("retrievals") or []:
            cid = str(r["chunk_id"])
            index.setdefault(cid, r)
            if cid not in current:
                current.append(cid)
    for versions in versions_by_node.values():
        for v in versions:
            for r in v.get("retrievals") or []:
                index.setdefault(str(r["chunk_id"]), r)
    return index, current


def extract_record_parts(
    snapshot: dict, versions_by_node: dict[str, list[dict]] | None = None
) -> tuple[list[ClaimOut], list[Citation], list[Evidence]]:
    """Build the brief → claims → chunks citation chain of a finished LiveGraph run.

    - claims: the `claims` node's current `output.claims[*]` (key, text, support_chunk_ids), joined by key
      with `snapshot.claims` for the judged confidence and active/struck status. Snapshot claims missing
      from the node output (e.g. added later) are appended with no support.
    - citations: one per content sentence of the brief (`brief_sentences`), with the `(Cn)` keys it cites
      and the union of those claims' support_chunk_ids. Sentences that cite nothing are kept (empty lists),
      so citation recall can count them.
    - evidence: every chunk retrieved by a current version, plus any chunk a claim cites that only an older
      version retrieved (looked up in `versions_by_node`, node id -> NodeVersion[] from GET /versions).
    """
    versions_by_node = versions_by_node or {}
    judged = {c["key"]: c for c in snapshot.get("claims", [])}
    claims: list[ClaimOut] = []
    v = _current_version(snapshot, "claims")
    for item in ((v or {}).get("output") or {}).get("claims", []):
        key = str(item.get("key", "")).strip()
        j = judged.get(key, {})
        claims.append(
            ClaimOut(
                key=key,
                text=item.get("text", ""),
                confidence=j.get("confidence"),
                status=j.get("status", "active"),
                support_chunk_ids=[str(x) for x in item.get("support_chunk_ids", [])],
            )
        )
    seen = {c.key for c in claims}
    for key, j in judged.items():
        if key not in seen:
            claims.append(ClaimOut(key=key, text=j.get("text", ""), confidence=j.get("confidence"),
                                   status=j.get("status", "active")))

    support = {c.key: c.support_chunk_ids for c in claims}
    citations: list[Citation] = []
    for sent in brief_sentences(brief_of(snapshot)):
        keys = claim_refs(sent)
        chunks: list[str] = []
        for k in keys:
            chunks += [x for x in support.get(k, []) if x not in chunks]
        citations.append(Citation(sentence=sent, claim_keys=keys, chunk_ids=chunks))

    index, current = _chunk_index(snapshot, versions_by_node)
    wanted = list(current)
    for c in claims:
        wanted += [x for x in c.support_chunk_ids if x not in wanted]
    evidence = [
        Evidence(
            chunk_id=cid,
            title=index[cid].get("document_title", ""),
            url=index[cid].get("source_url"),
            text=index[cid].get("text", ""),
        )
        for cid in wanted
        if cid in index
    ]
    return claims, citations, evidence
