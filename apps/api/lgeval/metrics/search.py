"""Search behaviour of a LiveGraph run, from its saved snapshot (no LLM calls, no cost).

Per run, over its active research nodes' current versions (the version's ``trace`` and ``retrievals``):

| metric | meaning |
|---|---|
| ``search_tool_calls`` | mean top-level tool calls per research node |
| ``search_budget_exhausted`` | share of research nodes where a search tool hit its budget |
| ``search_rule_stop`` | share stopped by the research loop's stop rule (``reflect``) |
| ``search_rounds`` | mean ``reflect`` rounds per research node |
| ``search_challenge`` | share of research nodes that ran at least one ``kind=challenge`` web search |
| ``search_domains_cited`` | mean distinct sites among a research node's cited chunks |
| ``search_cited_share`` | cited chunks / retrieved chunks, over the run's research nodes |
| ``search_follow_ups`` | follow-up research nodes in the run |
| ``search_follow_up_depth`` | deepest follow-up chain (0 = none) |

Older snapshots (before the research loop) score 0 rounds, 0 rule stops and 0 challenge searches, which is
what they did. Imported runs and runs without a snapshot get no search scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from lgeval.schema import RunRecord, ScoreRecord

SEARCH_TOOLS = ("web_search", "search_documents", "read_source", "fetch_page")
METRICS = (
    "search_tool_calls", "search_budget_exhausted", "search_rule_stop", "search_rounds", "search_challenge",
    "search_domains_cited", "search_cited_share", "search_follow_ups", "search_follow_up_depth",
)


def _domain(url: str | None) -> str:
    host = (urlparse(url or "").hostname or "").lower()
    return host.removeprefix("www.")


def load_snapshot(run: RunRecord, results_dir: Path) -> dict[str, Any] | None:
    if not run.snapshot_path:
        return None
    path = Path(run.snapshot_path)
    path = path if path.is_absolute() else results_dir / path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _nodes(snap: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = snap.get("nodes") or []
    return list(nodes.values()) if isinstance(nodes, dict) else list(nodes)


def research_versions(snap: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(node, current version) of every active research node that has a version. A snapshot's ``versions`` maps
    node id -> that node's current version (``GET /runs/{id}`` shape); a list of versions is matched by id."""
    versions = snap.get("versions") or {}
    if isinstance(versions, dict):
        by_node = versions
        by_version = {str(v.get("id")): v for v in versions.values() if isinstance(v, dict)}
    else:
        by_node, by_version = {}, {str(v.get("id")): v for v in versions}
    out = []
    for n in _nodes(snap):
        if n.get("type") != "research" or n.get("removed"):
            continue
        v = by_node.get(str(n.get("id"))) or by_version.get(str(n.get("current_version_id")))
        if v is not None:
            out.append((n, v))
    return out


def node_stats(version: dict[str, Any]) -> dict[str, Any]:
    trace = [s for s in version.get("trace") or [] if not s.get("parent_call_id")]
    search_steps = [s for s in trace if s.get("tool") in SEARCH_TOOLS]
    stats = (version.get("output") or {}).get("search_stats") or {}
    retrievals = version.get("retrievals") or []
    cited = [r for r in retrievals if r.get("cited")]
    challenge = sum(
        1 for s in trace if s.get("tool") == "web_search" and "kind=challenge" in str(s.get("args_summary", ""))
    ) or int((stats.get("queries") or {}).get("challenge", 0))
    return {
        "tool_calls": len(trace),
        "exhausted": any(s.get("summary") == "budget exhausted" for s in search_steps),
        "rule_stop": stats.get("stopped_by") == "rule",
        "rounds": sum(1 for s in trace if s.get("tool") == "reflect"),
        "challenge": challenge > 0,
        "domains_cited": len({d for d in (_domain(r.get("source_url")) for r in cited) if d}),
        "retrieved": len(retrievals),
        "cited": len(cited),
    }


def run_metrics(snap: dict[str, Any]) -> dict[str, tuple[float, int]] | None:
    """{metric: (value, n)} for one run; None when it has no research versions."""
    pairs = research_versions(snap)
    if not pairs:
        return None
    rows = [node_stats(v) for _, v in pairs]
    n = len(rows)

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in rows) / n

    depths = [
        int(((node.get("config") or {}).get("params") or {}).get("depth") or 1)
        for node, _ in pairs if ((node.get("config") or {}).get("params") or {}).get("follow_up")
    ]
    retrieved = sum(r["retrieved"] for r in rows)
    out = {
        "search_tool_calls": (mean("tool_calls"), n),
        "search_budget_exhausted": (mean("exhausted"), n),
        "search_rule_stop": (mean("rule_stop"), n),
        "search_rounds": (mean("rounds"), n),
        "search_challenge": (mean("challenge"), n),
        "search_domains_cited": (mean("domains_cited"), n),
        "search_follow_ups": (float(len(depths)), n),
        "search_follow_up_depth": (float(max(depths, default=0)), n),
    }
    if retrieved:
        out["search_cited_share"] = (sum(r["cited"] for r in rows) / retrieved, retrieved)
    return out


def score_search(run: RunRecord, results_dir: Path) -> list[ScoreRecord]:
    snap = load_snapshot(run, results_dir)
    metrics = run_metrics(snap) if snap else None
    if not metrics:
        return []
    return [
        ScoreRecord(question_id=run.question_id, system=run.system, repeat=run.repeat, metric=m, value=v, n=n)
        for m, (v, n) in metrics.items()
    ]
