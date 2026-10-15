"""Agent activity traces: fold ``tool.call`` / ``tool.result`` / ``agent.message`` payloads into TraceSteps.

One fold is shared by the running agent (its ``NodeDeps.trace`` becomes ``node_versions.trace``) and by the
snapshot (``RunSnapshot.activity`` is folded from persisted events), so both always agree.
"""

from __future__ import annotations

import json
from typing import Any

TRUNC = 300
TOOL_EVENTS = ("tool.call", "tool.result", "agent.message")


def truncate(s: str, n: int = TRUNC) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def summarize_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    if len(args) == 1:
        v = next(iter(args.values()))
        return truncate(v if isinstance(v, str) else json.dumps(v, default=str))
    return truncate(", ".join(f"{k}={v if isinstance(v, str) else json.dumps(v, default=str)}" for k, v in args.items()))


class TraceBuilder:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self._by_call: dict[str, dict[str, Any]] = {}

    def apply(self, type: str, payload: dict[str, Any], ts: str = "") -> None:
        if type == "tool.call":
            self.call(payload, ts)
        elif type == "tool.result":
            self.result(payload)
        elif type == "agent.message":
            self.message(payload)

    def call(self, p: dict[str, Any], ts: str) -> None:
        step = {
            "call_id": p["call_id"], "tool": p["tool"], "agent": p.get("agent"),
            "parent_call_id": p.get("parent_call_id"), "args_summary": p.get("args_summary", ""),
            "ok": None, "summary": "", "chunk_ids": [], "node_ids": [], "ms": 0, "ts": ts, "exchange": [],
            "ref": None, "reads": [],
        }
        self.steps.append(step)
        self._by_call[p["call_id"]] = step

    def result(self, p: dict[str, Any]) -> None:
        step = self._by_call.get(p["call_id"])
        if step is None:
            return
        step["ok"] = bool(p.get("ok"))
        step["summary"] = p.get("summary", "")
        step["ms"] = int(p.get("ms") or 0)
        step["chunk_ids"] = [str(c["chunk_id"]) for c in p.get("chunks") or []]
        step["node_ids"] = [str(n) for n in p.get("node_ids") or []]
        step["ref"] = p.get("ref")  # context store ref of the result (recall(ref)); None when not stored
        step["reads"] = [{"node_key": str(r["node_key"])} for r in p.get("reads") or []]

    def message(self, p: dict[str, Any]) -> None:
        step = self._by_call.get(p["call_id"])
        if step is not None:
            step["exchange"].append(
                {"from_agent": p["from_agent"], "to_agent": p["to_agent"], "kind": p["kind"], "text": p["text"]}
            )

    def dump(self) -> list[dict[str, Any]]:
        return [dict(s, exchange=list(s["exchange"]), reads=list(s["reads"])) for s in self.steps]
