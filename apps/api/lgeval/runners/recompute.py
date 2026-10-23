"""Claim C benchmark: after a change, LiveGraph updates the brief at a fraction of the cost/time of starting over.

Methodology (cited by docs/EVAL.md and the paper)
-------------------------------------------------
Subjects. Finished `full`-config runs from a quality results dir (`runs.jsonl`, status `done`) or explicit
run ids. Runs are processed one at a time, so wall times are not skewed by contention between runs.

Mutations (all target choices are deterministic functions of the run snapshot, so a re-run of the benchmark
on the same runs picks the same targets):
- `identical`: Rerun of the `claims` node with nothing changed (no web access: a single-shot node). The
  engine forces a new version; if its output hash equals the old one, early cutoff stops propagation there,
  otherwise (LLM sampling noise) the descendants recompute. `meta.output_unchanged` says which happened.
- `edit`: the claims node's markdown is re-submitted without one claim (the active claim with the lowest
  judged confidence; ties -> highest key), in the `- **Ck** text` format `claims.parse_edit` reads (support
  ids are kept by key). The edit itself is not an LLM call; only its descendants are.
- `patch_question`: `PATCH /nodes/{id}` with a reworded equivalent of a research node's effective question
  ("Based on the available evidence, <question>"). The target is the research node whose current version
  cites the most chunks (ties -> key). The patched node may search the web live once (engine semantics).
- `distrust`: the document most cited by the brief: the chunk ids supporting the claims the brief cites
  `(Cn)` are mapped to documents via current-version retrievals (fallback: retrievals with `cited=true`);
  most chunks wins (ties -> document id). `PATCH /documents/{id} {trust: distrusted}`.
- `steer`: the same research node as `patch_question`, steered with a fixed template:
  `Focus specifically on the most recent quantitative evidence (figures, dates, named sources) for: "<q>";
  prefer primary sources over secondary commentary.`

Isolation. Mutations of one run interact (a distrust changes what a later steer sees), and a fresh full run
per mutation would multiply spend for no extra information. We apply the mutations sequentially to the same
run and *undo* each before the next: (1) revert the input (trust back to `normal`, PATCH the old question
back), (2) veto nodes the mutation added (e.g. a critic's follow-up research), (3) `restore` the pre-mutation
current version of every node whose current version differs, most-upstream first, until every node is back
on its pre-mutation version (≤3 passes). Because the engine reuses versions by input hash, the undo makes no
LLM calls in the normal case; its spend is still measured (`meta.undo_cost_usd`) and charged to the budget.
`meta.reset_ok` records whether the run was verifiably back (same current versions and same brief), and
the next record's `meta.clean_start` whether it started from the original state. A steer cannot be fully
undone (persisted steer texts stay in the node's input hash), so `steer` is always applied last; the
execution order is fixed: identical, edit, patch_question, distrust, steer.

Measurement. For each mutation: `nodes_total` = active nodes before; `nodes_impacted` = |union of
GET /impact node_ids| (action rerun/edit/config/steer; for distrust, `rerun` of every corpus-reading node
whose current retrievals touch the document) — an upper bound. We record `last_seq`, apply the mutation and
`wait_settled(since_seq=last_seq)`; then, over events with seq > last_seq: `nodes_recomputed` = distinct
nodes with a `node.version {reused: false}` whose version made an LLM call (model/tokens set; the edit
version itself does not count), `nodes_reused` = distinct other nodes with a `node.version {reused: true}`
(hash hit or early cutoff). Usage = tokens of exactly those new versions (per model), priced by `costs.py`.
`wall_s` = time between the first and the last event after the mark (server timestamps; no polling slack;
`meta.poll_wall_s` keeps the client-side elapsed time including wait_settled's polling).

Baselines ("starting over").
- `original_run` (default, zero extra spend): the original run's own cost and time, measured with the same
  event accounting (versions with `reused: false` up to the first `run.status done`; wall = first event to
  that event). `meta.record_cost_usd/record_wall_s` keep the RunRecord's figures for cross-checking.
- `fresh_run` (stronger evidence, costs one full run per source run): a new run of the same goal (global
  uploads re-attached; run-scoped uploads cannot be, `meta.baseline_dropped_uploads`), measured the same way.
  Its brief is stored as `baseline_brief` so metrics can test "no quality loss" (pairwise / factual parity of
  `brief_after` vs `baseline_brief`). The change itself is not expressed in the fresh run (a steer/edit has no
  honest up-front equivalent); the claim is about the cost of producing an updated brief either way.
Both sides count only persisted versions' tokens: superseded/cancelled attempts and embedding calls are not
priced on either side.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from lgeval.client import LiveGraphClient, brief_of, claim_refs, usage_of
from lgeval.schema import Manifest, RecomputeRecord, RunRecord, TokenUsage

MUTATIONS: tuple[str, ...] = ("identical", "edit", "patch_question", "distrust", "steer")  # execution order
DEFAULT_MUTATIONS = "distrust,steer,edit,identical"
CORPUS_TYPES = ("research", "prosecutor", "defender", "judge")
# rough share of a full run's spend per mutation, for --dry-run estimates only
EST_SHARE = {"identical": 0.6, "edit": 0.5, "patch_question": 0.7, "distrust": 0.8, "steer": 0.7}

CostFn = Callable[[dict[str, TokenUsage]], float]
Sink = Callable[[RecomputeRecord], None]


def log(msg: str) -> None:
    print(f"[recompute] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- snapshot helpers


def active_nodes(snap: dict) -> list[dict]:
    return [n for n in snap.get("nodes", []) if not n.get("removed")]


def nodes_of(snap: dict, node_type: str) -> list[dict]:
    return sorted((n for n in active_nodes(snap) if n["type"] == node_type), key=lambda n: n["key"])


def current(snap: dict, node_id: str) -> dict:
    return (snap.get("versions") or {}).get(node_id) or {}


def question_of(node: dict) -> str:
    return ((node.get("config") or {}).get("question") or node.get("title") or "").strip()


def _ts(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s)) if s else None
    except ValueError:
        return None


def _parents(snap: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for e in snap.get("edges", []):
        out.setdefault(e["target_node_id"], set()).add(e["source_node_id"])
    return out


def ancestors(snap: dict, node_id: str) -> set[str]:
    parents, seen, stack = _parents(snap), set(), [node_id]
    while stack:
        for p in parents.get(stack.pop(), ()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


# ---------------------------------------------------------------- event accounting


@dataclass
class Delta:
    """What happened in a window of events."""

    recomputed: list[str] = field(default_factory=list)  # node ids with a new LLM version
    reused: list[str] = field(default_factory=list)  # node ids reused (and not recomputed)
    edited: list[str] = field(default_factory=list)  # node ids with a new non-LLM version (edit/answer)
    added: list[str] = field(default_factory=list)  # node ids added
    new_versions: list[dict] = field(default_factory=list)
    usage: dict[str, TokenUsage] = field(default_factory=dict)
    wall_s: float | None = None  # first -> last event of the window (server clock)
    n_events: int = 0
    last_seq: int = 0


def _is_llm(v: dict) -> bool:
    return bool(v.get("model") or v.get("tokens_in") or v.get("tokens_out"))


def account(events: Iterable[dict], *, after_seq: int = 0, upto_seq: int | None = None) -> Delta:
    """Fold the persisted events with after_seq < seq <= upto_seq into a Delta."""
    d = Delta()
    recomputed: dict[str, None] = {}
    reused: dict[str, None] = {}
    edited: dict[str, None] = {}
    first = last = None
    for e in sorted(events, key=lambda e: e.get("seq") or 0):
        seq = e.get("seq")
        if seq is None or seq <= after_seq or (upto_seq is not None and seq > upto_seq):
            continue
        d.n_events += 1
        d.last_seq = seq
        t = _ts(e.get("ts"))
        if t is not None:
            first = first or t
            last = t
        nid = e.get("node_id")
        if e["type"] == "node.added":
            d.added.append(str((e.get("payload") or {}).get("node", {}).get("id") or nid))
        if e["type"] != "node.version" or nid is None:
            continue
        p = e.get("payload") or {}
        v = p.get("version") or {}
        if p.get("reused"):
            reused[nid] = None
        elif _is_llm(v):
            recomputed[nid] = None
            d.new_versions.append(v)
        else:
            edited[nid] = None
    d.recomputed = list(recomputed)
    d.reused = [n for n in reused if n not in recomputed]
    d.edited = [n for n in edited if n not in recomputed]
    d.usage = usage_of({"_": d.new_versions})
    if first is not None and last is not None:
        d.wall_s = max(0.0, (last - first).total_seconds())
    return d


def first_done_seq(events: Iterable[dict]) -> int | None:
    for e in sorted(events, key=lambda e: e.get("seq") or 0):
        if e["type"] == "run.status" and (e.get("payload") or {}).get("status") == "done":
            return e["seq"]
    return None


# ---------------------------------------------------------------- target selection (pure)


def _resolve_chunk(cid: str, chunk_docs: dict[str, str]) -> str | None:
    if cid in chunk_docs:
        return chunk_docs[cid]
    hits = {d for c, d in chunk_docs.items() if len(cid) >= 8 and c.startswith(cid)}
    return hits.pop() if len(hits) == 1 else None


def pick_distrust_doc(snap: dict) -> tuple[str, str, int] | None:
    """(document_id, title, n_supporting_chunks) of the document most cited by the brief's claims."""
    chunk_docs: dict[str, str] = {}
    titles: dict[str, str] = {}
    cited_counts: dict[str, int] = {}
    for v in (snap.get("versions") or {}).values():
        for r in v.get("retrievals") or []:
            if r.get("trust") == "distrusted":
                continue
            doc = str(r["document_id"])
            chunk_docs.setdefault(str(r["chunk_id"]), doc)
            titles.setdefault(doc, r.get("document_title") or "")
            if r.get("cited"):
                cited_counts[doc] = cited_counts.get(doc, 0) + 1
    claims = (current(snap, nodes_of(snap, "claims")[-1]["id"]).get("output") or {}).get("claims", []) \
        if nodes_of(snap, "claims") else []
    keys = set(claim_refs(brief_of(snap))) or {c.get("key") for c in claims}
    counts: dict[str, int] = {}
    for c in claims:
        if c.get("key") not in keys:
            continue
        for cid in c.get("support_chunk_ids") or []:
            doc = _resolve_chunk(str(cid), chunk_docs)
            if doc is not None:
                counts[doc] = counts.get(doc, 0) + 1
    counts = counts or cited_counts
    if not counts:
        return None
    doc = min(counts, key=lambda d: (-counts[d], d))
    return doc, titles.get(doc, ""), counts[doc]


def pick_research_node(snap: dict) -> dict | None:
    """The research node whose current version cites the most chunks (ties -> findings, then key)."""
    best, score = None, None
    for n in nodes_of(snap, "research"):
        v = current(snap, n["id"])
        if not v:
            continue
        s = (
            sum(1 for r in v.get("retrievals") or [] if r.get("cited")),
            len((v.get("output") or {}).get("findings") or []),
        )
        if score is None or s > score:
            best, score = n, s
    return best


def steer_text(node: dict) -> str:
    return (
        "Focus specifically on the most recent quantitative evidence (figures, dates, named sources) for: "
        f'"{question_of(node)}"; prefer primary sources over secondary commentary.'
    )


def reworded_question(q: str) -> str:
    q = q.strip()
    if not q:
        return q
    return f"Based on the available evidence, {q[0].lower() + q[1:] if not q[:2].isupper() else q}"


def _key_num(key: str) -> int:
    digits = "".join(ch for ch in key if ch.isdigit())
    return int(digits) if digits else 0


def drop_claim_edit(snap: dict) -> tuple[str, str] | None:
    """(edited claims markdown, dropped key): drop the active claim with the lowest judged confidence."""
    if not nodes_of(snap, "claims"):
        return None
    claims = (current(snap, nodes_of(snap, "claims")[-1]["id"]).get("output") or {}).get("claims") or []
    if len(claims) < 2:
        return None
    judged = {c["key"]: c for c in snap.get("claims", [])}

    def conf(c: dict) -> float:
        j = judged.get(c.get("key"), {})
        return 0.5 if j.get("confidence") is None else float(j["confidence"])

    active = [c for c in claims if judged.get(c.get("key"), {}).get("status", "active") == "active"] or claims
    drop = min(active, key=lambda c: (conf(c), -_key_num(str(c.get("key")))))
    lines = [f"- **{c['key']}** {c.get('text', '').strip()}" for c in claims if c is not drop]
    return "\n".join(lines), str(drop.get("key"))


# ---------------------------------------------------------------- mutation plans


@dataclass
class Plan:
    mutation: str
    target: str  # node key or document id
    detail: str
    impact: list[tuple[str, str]]  # (node_id, impact action) whose node_ids are unioned
    apply: Callable[[], Awaitable[Any]]
    revert: Callable[[], Awaitable[Any]] | None = None
    target_node_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def make_plan(client: LiveGraphClient, snap: dict, mutation: str) -> Plan | None:
    """Deterministic plan for `mutation` on this snapshot, or None when the run has no suitable target."""
    if mutation == "identical":
        n = (nodes_of(snap, "claims") or [None])[-1]
        if n is None or not current(snap, n["id"]):
            return None
        return Plan("identical", n["key"], "rerun with nothing changed", [(n["id"], "rerun")],
                    lambda: client.rerun(n["id"]), target_node_id=n["id"],
                    meta={"output_hash_before": current(snap, n["id"]).get("output_hash")})
    if mutation == "edit":
        n = (nodes_of(snap, "claims") or [None])[-1]
        e = drop_claim_edit(snap)
        if n is None or e is None:
            return None
        text, key = e
        return Plan("edit", n["key"], f"dropped claim {key}", [(n["id"], "edit")],
                    lambda: client.edit(n["id"], text), target_node_id=n["id"], meta={"dropped_claim": key})
    if mutation in ("steer", "patch_question"):
        n = pick_research_node(snap)
        if n is None:
            return None
        if mutation == "steer":
            text = steer_text(n)
            return Plan("steer", n["key"], text, [(n["id"], "steer")], lambda: client.steer(n["id"], text),
                        target_node_id=n["id"])
        old = (n.get("config") or {}).get("question")
        new = reworded_question(question_of(n))
        return Plan("patch_question", n["key"], new, [(n["id"], "config")],
                    lambda: client.patch_node(n["id"], question=new),
                    revert=lambda: client.patch_node(n["id"], question=old or ""),
                    target_node_id=n["id"], meta={"question_before": question_of(n)})
    if mutation == "distrust":
        pick = pick_distrust_doc(snap)
        if pick is None:
            return None
        doc, title, n_chunks = pick
        touching = [
            n["id"] for n in active_nodes(snap)
            if n["type"] in CORPUS_TYPES
            and any(str(r.get("document_id")) == doc for r in current(snap, n["id"]).get("retrievals") or [])
        ]
        return Plan("distrust", doc, title, [(nid, "rerun") for nid in touching],
                    lambda: client.set_trust(doc, "distrusted"), revert=lambda: client.set_trust(doc, "normal"),
                    meta={"supporting_chunks": n_chunks, "nodes_touching_doc": len(touching)})
    raise ValueError(f"unknown mutation {mutation!r}")


# ---------------------------------------------------------------- measuring one mutation


@dataclass
class Measurement:
    before: dict
    after: dict
    delta: Delta
    impacted: list[str]
    poll_wall_s: float


async def measure(client: LiveGraphClient, run_id: str, before: dict, plan: Plan, *,
                  timeout: float, poll: float) -> Measurement:
    impacted: dict[str, None] = {}
    for nid, action in plan.impact:
        for x in (await client.impact(nid, action)).get("node_ids", []):
            impacted[str(x)] = None
    mark = int(before.get("last_seq", 0))
    t0 = time.monotonic()
    await plan.apply()
    after = await client.wait_settled(run_id, timeout=timeout, poll=poll, since_seq=mark)
    poll_wall = time.monotonic() - t0
    delta = account(await client.events(run_id, since=mark), after_seq=mark)
    return Measurement(before, after, delta, list(impacted), poll_wall)


async def reset(client: LiveGraphClient, run_id: str, before: dict, plan: Plan, *, cost_fn: CostFn,
                timeout: float, poll: float, passes: int = 3) -> dict[str, Any]:
    """Bring the run back to `before` (see module docstring); returns reset_ok + undo spend."""
    snap = await client.snapshot(run_id)
    mark = int(snap.get("last_seq", 0))

    async def settle(fn: Callable[[], Awaitable[Any]]) -> dict:
        s0 = int((await client.snapshot(run_id)).get("last_seq", 0))
        await fn()
        return await client.wait_settled(run_id, timeout=timeout, poll=poll, since_seq=s0)

    orig = {n["id"] for n in active_nodes(before)}
    added = [n["id"] for n in active_nodes(snap) if n["id"] not in orig]
    vids = {nid: current(before, nid).get("id") for nid in orig}

    def changed(snap: dict) -> set[str]:
        live = {n["id"] for n in active_nodes(snap)}
        return {nid for nid in orig & live if vids[nid] and current(snap, nid).get("id") != vids[nid]}

    # one batch (revert, vetoes, restores) before waiting, so the scheduler never recomputes a half-undone mix
    diff = changed(snap)
    for i in range(passes):
        if i and not diff:
            break
        roots = sorted(nid for nid in diff if not (ancestors(snap, nid) & diff))

        async def undo(first: bool = not i, roots: list[str] = roots) -> None:
            if first and plan.revert is not None:
                await plan.revert()
            for nid in added if first else ():
                await client.veto(nid)
            for nid in roots:
                await client.restore(nid, vids[nid])

        snap = await settle(undo)
        diff = changed(snap)
    d = account(await client.events(run_id, since=mark), after_seq=mark)
    return {
        "reset_ok": not diff and brief_of(snap) == brief_of(before),
        "undo_cost_usd": cost_fn(d.usage),
        "undo_recomputed": len(d.recomputed),
        "undo_vetoed": len(added),
        "undo_unrestored": len(diff),
    }


# ---------------------------------------------------------------- baselines


@dataclass
class Baseline:
    kind: str  # fresh_run | original_run
    wall_s: float
    cost_usd: float
    brief: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


async def original_baseline(client: LiveGraphClient, run_id: str, cost_fn: CostFn,
                            record: RunRecord | None = None) -> Baseline:
    events = await client.events(run_id, since=0)
    done = first_done_seq(events)
    d = account(events, after_seq=0, upto_seq=done)
    meta: dict[str, Any] = {"baseline_run_id": run_id, "baseline_done_seq": done}
    if record is not None:
        meta.update(record_cost_usd=record.cost_usd, record_wall_s=record.wall_s)
    if done is None or d.wall_s is None:  # no finished window in the log: fall back to the RunRecord
        if record is None or record.wall_s is None or record.cost_usd is None:
            raise ValueError(f"run {run_id}: no completed run in the event log and no RunRecord figures")
        return Baseline("original_run", record.wall_s, record.cost_usd, meta={**meta, "source": "run_record"})
    return Baseline("original_run", d.wall_s, cost_fn(d.usage), meta={**meta, "source": "events",
                                                                       "baseline_recomputed": len(d.recomputed)})


async def fresh_baseline(client: LiveGraphClient, snap: dict, cost_fn: CostFn, *,
                         timeout: float, poll: float) -> Baseline:
    run = snap["run"]
    docs = [d for d in snap.get("documents", []) if d.get("origin") == "upload"]
    global_docs = [str(d["id"]) for d in docs if not d.get("run_id")]
    created = await client.create_run(run["goal"], global_docs or None)
    new_id = created["run"]["id"]
    t0 = time.monotonic()
    after = await client.wait_settled(new_id, timeout=timeout, poll=poll)
    d = account(await client.events(new_id, since=0))
    return Baseline(
        "fresh_run",
        d.wall_s if d.wall_s is not None else time.monotonic() - t0,
        cost_fn(d.usage),
        brief_of(after),
        meta={"baseline_run_id": new_id, "baseline_status": after["run"]["status"],
              "baseline_dropped_uploads": len(docs) - len(global_docs)},
    )


# ---------------------------------------------------------------- driver


@dataclass
class Target:
    run_id: str
    question_id: str
    record: RunRecord | None = None


async def bench_run(
    client: LiveGraphClient,
    target: Target,
    mutations: list[str],
    *,
    baseline: str,
    cost_fn: CostFn,
    sink: Sink,
    budget: Any | None = None,
    timeout: float = 900,
    poll: float = 2.0,
) -> list[RecomputeRecord]:
    """Apply each mutation to one run (in MUTATIONS order, undoing in between) and emit RecomputeRecords."""
    order = [m for m in MUTATIONS if m in mutations]
    out: list[RecomputeRecord] = []
    run_id = target.run_id
    snap = await client.wait_settled(run_id, timeout=timeout, poll=poll)
    if snap["run"]["status"] != "done":
        log(f"{run_id}: status {snap['run']['status']}, skipped")
        return out
    if baseline == "fresh_run":
        base = await fresh_baseline(client, snap, cost_fn, timeout=timeout, poll=poll)
        if budget is not None:
            budget.add(base.cost_usd)
    else:
        base = await original_baseline(client, run_id, cost_fn, target.record)
    clean = True
    for m in order:
        if budget is not None and budget.exhausted():
            log(f"budget exhausted ({budget!r}); stopping")
            break
        before = await client.snapshot(run_id)
        plan = make_plan(client, before, m)
        if plan is None:
            log(f"{run_id}: no target for {m}, skipped")
            continue
        try:
            meas = await measure(client, run_id, before, plan, timeout=timeout, poll=poll)
        except TimeoutError as e:
            log(f"{run_id}: {m} did not settle ({e}); remaining mutations of this run skipped")
            break
        except httpx.HTTPStatusError as e:
            log(f"{run_id}: {m} rejected: {e.response.status_code} {e.response.text[:200]}")
            continue
        d = meas.delta
        cost = cost_fn(d.usage)
        if budget is not None:
            budget.add(cost)
        meta: dict[str, Any] = {**plan.meta, **base.meta, "clean_start": clean, "poll_wall_s": meas.poll_wall_s,
                                "nodes_added": len(d.added), "nodes_edited": len(d.edited), "events": d.n_events,
                                "status_after": meas.after["run"]["status"]}
        if m == "identical" and plan.target_node_id:
            meta["output_unchanged"] = (
                current(meas.after, plan.target_node_id).get("output_hash") == plan.meta.get("output_hash_before")
            )
        try:
            info = await reset(client, run_id, before, plan, cost_fn=cost_fn, timeout=timeout, poll=poll)
        except (TimeoutError, httpx.HTTPStatusError) as e:
            info = {"reset_ok": False, "reset_error": str(e)[:300]}
        if budget is not None:
            budget.add(info.get("undo_cost_usd", 0.0))
        meta.update(info)
        rec = RecomputeRecord(
            question_id=target.question_id,
            run_id=run_id,
            mutation=m,  # type: ignore[arg-type]
            target=plan.target,
            detail=plan.detail,
            wall_s=d.wall_s if d.wall_s is not None else meas.poll_wall_s,
            usage=d.usage,
            cost_usd=cost,
            nodes_total=len(active_nodes(before)),
            nodes_impacted=len(meas.impacted),
            nodes_recomputed=len(d.recomputed),
            nodes_reused=len(d.reused),
            baseline=base.kind,  # type: ignore[arg-type]
            baseline_wall_s=base.wall_s,
            baseline_cost_usd=base.cost_usd,
            brief_before=brief_of(before),
            brief_after=brief_of(meas.after),
            baseline_brief=base.brief,
            meta=meta,
        )
        sink(rec)
        out.append(rec)
        log(f"{run_id} {m}: ${cost:.4f} vs ${base.cost_usd:.4f}, {rec.wall_s:.0f}s vs {base.wall_s:.0f}s, "
            f"recomputed {rec.nodes_recomputed}/{rec.nodes_total} reused {rec.nodes_reused}"
            f"{'' if info.get('reset_ok') else ' (RESET FAILED)'}")
        if not info.get("reset_ok"):
            clean = False
    return out


# ---------------------------------------------------------------- summary


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _group(recs: list[RecomputeRecord]) -> dict[str, Any]:
    cost_r = [r.cost_usd / r.baseline_cost_usd for r in recs if r.baseline_cost_usd > 0]
    time_r = [r.wall_s / r.baseline_wall_s for r in recs if r.baseline_wall_s > 0]
    cr = _median(cost_r)
    kinds = {r.baseline for r in recs}
    return {
        "n": len(recs),
        "n_runs": len({r.run_id for r in recs}),
        "baseline": kinds.pop() if len(kinds) == 1 else "mixed",
        "cost_ratio_median": cr,
        "time_ratio_median": _median(time_r),
        "cost_savings_pct_median": None if cr is None else 100.0 * (1.0 - cr),
        "cost_usd_median": _median([r.cost_usd for r in recs]),
        "wall_s_median": _median([r.wall_s for r in recs]),
        "baseline_cost_usd_median": _median([r.baseline_cost_usd for r in recs]),
        "baseline_wall_s_median": _median([r.baseline_wall_s for r in recs]),
        "nodes_total_mean": _mean([float(r.nodes_total) for r in recs]),
        "nodes_impacted_mean": _mean([float(r.nodes_impacted) for r in recs]),
        "nodes_recomputed_mean": _mean([float(r.nodes_recomputed) for r in recs]),
        "nodes_reused_mean": _mean([float(r.nodes_reused) for r in recs]),
        "recompute_frac_mean": _mean([r.nodes_recomputed / r.nodes_total for r in recs if r.nodes_total]),
        "brief_changed_frac": _mean([float(r.brief_after.strip() != r.brief_before.strip()) for r in recs]),
        "clean_frac": _mean([float(r.meta.get("clean_start", True) and r.meta.get("reset_ok", True))
                             for r in recs]),
    }


def summarize(records: Iterable[RecomputeRecord | dict]) -> dict[str, dict[str, Any]]:
    """Per-mutation summary of RecomputeRecords (dicts are validated first) for report.py.

    Returns ``{mutation: stats, ..., "all": stats}`` — mutation keys in MUTATIONS order, only those present;
    "all" pools every record (present iff there is at least one). ``stats`` is::

        {
          "n": int,                               # records
          "n_runs": int,                          # distinct run ids
          "baseline": "original_run" | "fresh_run" | "mixed",
          "cost_ratio_median": float | None,      # median of cost_usd / baseline_cost_usd (baseline > 0)
          "time_ratio_median": float | None,      # median of wall_s / baseline_wall_s (baseline > 0)
          "cost_savings_pct_median": float | None,  # 100 * (1 - cost_ratio_median)
          "cost_usd_median": float, "wall_s_median": float,
          "baseline_cost_usd_median": float, "baseline_wall_s_median": float,
          "nodes_total_mean": float, "nodes_impacted_mean": float,
          "nodes_recomputed_mean": float, "nodes_reused_mean": float,
          "recompute_frac_mean": float | None,    # mean of nodes_recomputed / nodes_total
          "brief_changed_frac": float,            # share of records whose brief text changed
          "clean_frac": float,                    # share started from the original state and reset cleanly
        }

    Ratios are medians of per-record ratios (not ratios of medians); < 1 means cheaper/faster than
    starting over. Quality parity (brief_after vs baseline_brief) is scored by metrics/, not here.
    """
    recs = [r if isinstance(r, RecomputeRecord) else RecomputeRecord.model_validate(r) for r in records]
    out: dict[str, dict[str, Any]] = {}
    for m in MUTATIONS:
        group = [r for r in recs if r.mutation == m]
        if group:
            out[m] = _group(group)
    if recs:
        out["all"] = _group(recs)
    return out


# ---------------------------------------------------------------- inputs + CLI


def load_targets(from_dir: Path | None = None, run_ids: list[str] | None = None) -> list[Target]:
    """`full` runs with status done from `<from_dir>/runs.jsonl` (or any runs.jsonl below it), plus run ids."""
    targets: list[Target] = []
    seen: set[str] = set()
    if from_dir is not None:
        top = from_dir / "runs.jsonl"
        files = [top] if top.exists() else sorted(from_dir.rglob("runs.jsonl"))
        if not files:
            raise FileNotFoundError(f"no runs.jsonl under {from_dir}")
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = RunRecord.model_validate_json(line)
                except ValueError:
                    continue
                if r.system == "full" and r.status == "done" and r.run_id and r.run_id not in seen:
                    seen.add(r.run_id)
                    targets.append(Target(r.run_id, r.question_id, r))
    for rid in run_ids or []:
        if rid not in seen:
            seen.add(rid)
            targets.append(Target(rid, f"run-{rid[:8]}"))
    return targets


def parse_mutations(s: str) -> list[str]:
    ms = [m.strip() for m in s.split(",") if m.strip()]
    bad = [m for m in ms if m not in MUTATIONS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown mutation(s) {bad}; choose from {list(MUTATIONS)}")
    return ms


def estimate_usd(n_runs: int, mutations: list[str], baseline: str, run_usd: float) -> float:
    per_run = sum(EST_SHARE[m] for m in mutations) + (1.0 if baseline == "fresh_run" else 0.0)
    return n_runs * per_run * run_usd


def make_cost_fn(offline: bool = False) -> tuple[CostFn, dict]:
    """Cost function that loads OpenRouter prices for each new model it sees; also returns the price dict."""
    from lgeval import costs

    prices: dict = {}

    def fn(usage: dict[str, TokenUsage]) -> float:
        missing = [m for m in usage if m not in prices]
        if missing:
            prices.update(costs.load_pricing(missing, offline=offline))
        return costs.cost_usd(usage, prices)

    return fn, prices


def register(subparsers: Any) -> None:
    p = subparsers.add_parser("recompute", help="Claim C: incremental recompute vs starting over",
                              description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--from", dest="from_dir", type=Path, help="quality results dir (reads runs.jsonl)")
    p.add_argument("--run-ids", default="", help="comma-separated LiveGraph run ids")
    p.add_argument("--mutations", type=parse_mutations, default=parse_mutations(DEFAULT_MUTATIONS),
                   help=f"comma list of {','.join(MUTATIONS)} (default {DEFAULT_MUTATIONS})")
    p.add_argument("--baseline", choices=["original_run", "fresh_run"], default="original_run")
    p.add_argument("--limit", type=int, default=None, help="at most N runs")
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for a run to settle")
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=None, help="results root (default eval-results/)")
    p.add_argument("--tag", default=None, help="run tag; pass an existing one to resume")
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--spawn", action="store_true", help="start an API server instead of using --api-url")
    p.set_defaults(func=cmd_recompute)


def cmd_recompute(args: argparse.Namespace) -> int:
    return asyncio.run(run_cli(args))


async def run_cli(args: argparse.Namespace) -> int:
    from lgeval import costs
    from lgeval.store import DEFAULT_ROOT, ResultStore

    run_ids = [x.strip() for x in (args.run_ids or "").split(",") if x.strip()]
    targets = load_targets(args.from_dir, run_ids)[: args.limit]
    if not targets:
        log("no runs to benchmark (need --from with full/done RunRecords, or --run-ids)")
        return 2
    est = estimate_usd(len(targets), args.mutations, args.baseline, costs.est_run_usd())
    log(f"plan: {len(targets)} runs x {len(args.mutations)} mutations ({','.join(args.mutations)}), "
        f"baseline={args.baseline}; estimated ≤ ${est:.2f}")
    if args.dry_run:
        return 0
    store = ResultStore(args.out or DEFAULT_ROOT, "recompute", run_tag=args.tag)
    done = store.done_keys("recompute", lambda d: f"{d['run_id']}:{d['mutation']}")
    cost_fn, prices = make_cost_fn()
    budget = costs.Budget(args.max_usd)
    manifest = Manifest(experiment="recompute", args={k: (str(v) if isinstance(v, Path) else v)
                                                      for k, v in vars(args).items() if k != "func"})
    store.write_manifest(manifest)

    async def go(base_url: str) -> None:
        async with LiveGraphClient(base_url) as client:
            if not await client.health():
                raise SystemExit(f"API not reachable at {base_url} (start it or pass --spawn)")
            for t in targets:
                if budget.exhausted():
                    log(f"budget exhausted ({budget!r})")
                    break
                todo = [m for m in args.mutations if f"{t.run_id}:{m}" not in done]
                if not todo:
                    continue
                try:
                    await bench_run(client, t, todo, baseline=args.baseline, cost_fn=cost_fn, budget=budget,
                                    sink=lambda r: store.append("recompute", r), timeout=args.timeout,
                                    poll=args.poll)
                except (httpx.HTTPError, TimeoutError, ValueError) as e:
                    log(f"{t.run_id}: failed: {e}")

    if args.spawn:
        from lgeval.server import ApiServer

        async with ApiServer(log_path=store.dir / "api.log") as srv:
            await go(srv.base_url)
    else:
        await go(args.api_url)
    store.write_manifest(manifest.model_copy(update={"pricing": costs.pricing_manifest(prices)}))
    log(f"done: {budget!r}; records in {store.path('recompute')}")
    return 0


__all__ = [
    "MUTATIONS", "Delta", "Plan", "Target", "account", "bench_run", "drop_claim_edit", "load_targets",
    "make_plan", "pick_distrust_doc", "pick_research_node", "register", "reset", "summarize",
]
