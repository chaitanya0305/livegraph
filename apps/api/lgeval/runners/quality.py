"""Claim A run matrix: questions x configs x repeats -> `RunRecord`s in `runs.jsonl`.

One API server per config (env overrides from `configs.py`), started one config at a time; within a config
the runs go through `LiveGraphClient` with bounded concurrency. Every run's tokens are summed over ALL node
versions (recomputes and agent-requested reruns included). A `human_input` gate is answered automatically
with a neutral answer. Failures and timeouts are recorded and never stop the matrix; `--resume <tag>` skips
(question, config, repeat) triples already in runs.jsonl; `--max-usd` stops starting new runs once the
measured spend reaches it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lgeval.client import LiveGraphClient, brief_of, extract_record_parts, is_quiet, usage_of
from lgeval.costs import (
    CONFIG_COST_FACTOR,
    Budget,
    Pricing,
    cost_usd,
    est_run_usd,
    pricing_manifest,
    pricing_table,
    resolve_model,
)
from lgeval.datasets import load_questions
from lgeval.runners import configs as cfgs
from lgeval.schema import Manifest, Question, RunRecord, TokenUsage, now
from lgeval.server import ApiServer
from lgeval.store import DEFAULT_ROOT, ResultStore

log = logging.getLogger("lgeval.quality")

EXPERIMENT = "quality"
RUNS = "runs"
NEUTRAL_ANSWER = "Proceed with your best judgment."
MAX_AUTO_ANSWERS = 5  # human_input answers per run before giving up on it

ServerFactory = Callable[[str, dict[str, str]], AbstractAsyncContextManager[Any]]  # (config, env) -> has .base_url
ClientFactory = Callable[[str], LiveGraphClient]


def run_key(r: dict | RunRecord) -> str:
    d = r.model_dump() if isinstance(r, RunRecord) else r
    return f"{d['question_id']}|{d['system']}|{d.get('repeat', 0)}"


@dataclass(frozen=True)
class Job:
    question: Question
    system: str
    repeat: int

    @property
    def key(self) -> str:
        return f"{self.question.id}|{self.system}|{self.repeat}"


def plan_jobs(questions: list[Question], systems: list[str], repeats: int, done: set[str] | None = None) -> dict[str, list[Job]]:
    """Config -> jobs still to run (repeat-major, so a budget stop leaves whole repeats rather than holes)."""
    done = done or set()
    return {
        s: [j for r in range(repeats) for q in questions if (j := Job(q, s, r)).key not in done]
        for s in systems
    }


def estimate_usd(jobs: dict[str, list[Job]], pricing: dict[str, Pricing]) -> dict[str, float]:
    per_run = est_run_usd(pricing)
    return {s: len(js) * per_run * CONFIG_COST_FACTOR.get(s, 1.0) for s, js in jobs.items()}


# ---------------------------------------------------------------- one run


async def _settle(client: LiveGraphClient, run_id: str, *, deadline: float, poll: float) -> tuple[dict, int]:
    """Poll until the run is quiet (as `LiveGraphClient.wait_settled`), answering human_input gates on the way;
    -> (snapshot, number of answers given). Own loop so waiting gates get answered instead of ending the wait."""
    answered, prev_seq = 0, None
    while True:
        snap = await client.snapshot(run_id)
        seq = int(snap.get("last_seq", 0))
        waiting = [n for n in snap["nodes"] if n["status"] == "waiting_input" and not n.get("removed")]
        if snap["run"]["status"] == "waiting_input" and waiting:
            if answered >= MAX_AUTO_ANSWERS:
                return snap, answered
            for n in waiting:
                await client.answer(n["id"], NEUTRAL_ANSWER)
                answered += 1
            prev_seq = None
        elif is_quiet(snap) and seq == prev_seq:
            return snap, answered
        else:
            prev_seq = seq
        if time.monotonic() >= deadline:
            raise TimeoutError(f"run {run_id} not settled after the timeout (status={snap['run']['status']})")
        await asyncio.sleep(poll)


async def _first_error(client: LiveGraphClient, run_id: str) -> str | None:
    with contextlib.suppress(Exception):
        for e in await client.events(run_id):
            if e["type"] == "error":
                return str((e.get("payload") or {}).get("message", ""))[:500]
    return None


async def run_one(
    client: LiveGraphClient, job: Job, *, store: ResultStore, pricing: dict[str, Pricing], timeout: float,
    poll: float = 2.0,
) -> tuple[RunRecord, int]:
    """Run one question on the server behind `client`; -> (record, human_input answers given)."""
    started, t0 = now(), time.monotonic()
    run_id: str | None = None
    snap: dict | None = None
    status, error, answered = "failed", None, 0
    try:
        run_id = str((await client.create_run(job.question.question))["run"]["id"])
        snap, answered = await _settle(client, run_id, deadline=t0 + timeout, poll=poll)
        status = snap["run"]["status"]
        if status == "waiting_input":
            status, error = "failed", f"still waiting for input after {answered} automatic answers"
        elif status != "done":
            error = await _first_error(client, run_id)
    except TimeoutError as e:
        status, error, snap = "timeout", str(e), None
        if run_id:
            with contextlib.suppress(Exception):
                await client.cancel(run_id)
    except Exception as e:  # noqa: BLE001 -- never let one run take the matrix down
        status, error = "failed", f"{type(e).__name__}: {e}"
    rec = RunRecord(
        question_id=job.question.id, dataset=job.question.dataset, system=job.system, repeat=job.repeat,
        run_id=run_id, status=status, error=error, started_at=started, wall_s=round(time.monotonic() - t0, 3),
    )
    if run_id is None:
        return rec, answered
    try:  # a failed or timed-out run still spent tokens: record whatever exists
        snap = snap or await client.snapshot(run_id)
        versions = await client.all_versions(snap)
        rec.usage = usage_of(versions)
        rec.cost_usd = cost_usd(rec.usage, pricing)
        rec.n_nodes = sum(1 for n in snap.get("nodes", []) if not n.get("removed"))
        rec.brief = brief_of(snap)
        rec.claims, rec.citations, rec.evidence = extract_record_parts(snap, versions)
        path = store.save_snapshot(run_id, {**snap, "node_versions": versions})
        rec.snapshot_path = str(path.relative_to(store.dir))
    except Exception as e:  # noqa: BLE001 -- recorded on the run
        rec.error = "; ".join(x for x in (rec.error, f"collect: {type(e).__name__}: {e}") if x)
    return rec, answered


# ---------------------------------------------------------------- the matrix


@dataclass
class MatrixResult:
    records: list[RunRecord]
    skipped: int  # jobs not started because the budget ran out (or their server failed to start)
    auto_answers: int


def _default_server(store: ResultStore) -> ServerFactory:
    def make(name: str, env: dict[str, str]) -> ApiServer:
        return ApiServer(env, log_path=store.dir / "logs" / f"api-{name}.log")

    return make


async def run_matrix(
    jobs: dict[str, list[Job]],
    configs: dict[str, dict[str, str]],
    *,
    store: ResultStore,
    budget: Budget,
    pricing: dict[str, Pricing],
    concurrency: int = 2,
    timeout: float = 900,
    poll: float = 2.0,
    server_factory: ServerFactory | None = None,
    client_factory: ClientFactory = LiveGraphClient,
) -> MatrixResult:
    server_factory = server_factory or _default_server(store)
    res = MatrixResult([], 0, 0)
    for name, todo in jobs.items():
        if not todo:
            continue
        if budget.exhausted():
            res.skipped += len(todo)
            continue
        log.info("config %s: %d runs (concurrency %d)", name, len(todo), concurrency)
        try:
            async with server_factory(name, configs[name]) as srv, client_factory(srv.base_url) as client:
                sem = asyncio.Semaphore(max(1, concurrency))

                async def one(job: Job, client: LiveGraphClient = client, sem: asyncio.Semaphore = sem) -> None:
                    async with sem:
                        if budget.exhausted():
                            res.skipped += 1
                            return
                        rec, answered = await run_one(client, job, store=store, pricing=pricing, timeout=timeout, poll=poll)
                    budget.add(rec.cost_usd or 0.0)
                    res.auto_answers += answered
                    res.records.append(rec)
                    store.append(RUNS, rec)
                    log.info(
                        "%s %s r%d: %s in %.0fs, $%.4f (%r)", job.system, job.question.id, job.repeat, rec.status,
                        rec.wall_s or 0, rec.cost_usd or 0, budget,
                    )

                await asyncio.gather(*(one(j) for j in todo))
        except Exception as e:  # noqa: BLE001 -- a server that will not start: leave its jobs for --resume
            log.error("config %s aborted: %s: %s", name, type(e).__name__, e)
            res.skipped += len(todo) - sum(1 for r in res.records if r.system == name)
    return res


# ---------------------------------------------------------------- CLI


def _models(configs: dict[str, dict[str, str]]) -> dict[str, str]:
    try:
        from livegraph.config import settings  # model names only (the API server reads the same .env)
    except Exception:  # noqa: BLE001  # pragma: no cover
        return {}
    env = next(iter(configs.values()), {})
    return {"smart": env.get("MODEL_SMART", settings.model_smart), "fast": env.get("MODEL_FAST", settings.model_fast)}


def args_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}


def used_pricing(usages: list[dict[str, TokenUsage]], pricing: dict[str, Pricing]) -> dict[str, dict[str, float]]:
    seen = {m for u in usages for m in u}
    return pricing_manifest({m: pricing[k] for m in sorted(seen) if (k := resolve_model(m, pricing))})


def print_plan(jobs: dict[str, list[Job]], est: dict[str, float], *, what: str) -> None:
    print(f"{what}: {sum(len(j) for j in jobs.values())} runs")
    for s, js in jobs.items():
        print(f"  {s:<17} {len(js):>4} runs  est ${est.get(s, 0.0):.2f}")
    print(f"  {'total':<17} {'':>4}       est ${sum(est.values()):.2f}")


async def cmd_quality(args: argparse.Namespace) -> int:
    try:
        configs = cfgs.resolve(args.configs)
    except ValueError as e:
        print(e)
        return 2
    questions = load_questions(args.dataset, limit=args.limit, seed=args.seed)
    store: ResultStore | None = None
    done: set[str] = set()
    if args.resume:
        if not (args.out / EXPERIMENT / args.resume).is_dir():
            print(f"nothing to resume at {args.out / EXPERIMENT / args.resume}")
            return 2
        store = ResultStore(args.out, EXPERIMENT, run_tag=args.resume)
        done = store.done_keys(RUNS, run_key)
    jobs = plan_jobs(questions, list(configs), args.repeats, done)
    pricing = pricing_table(offline=args.offline)
    print_plan(jobs, estimate_usd(jobs, pricing), what=f"quality on {args.dataset} ({len(questions)} questions x "
               f"{args.repeats} repeats; {len(done)} already done)")
    if args.dry_run:
        return 0
    store = store or ResultStore(args.out, EXPERIMENT)
    manifest = Manifest(
        experiment=EXPERIMENT, args=args_dict(args), configs=configs, models=_models(configs),
        pricing=pricing_manifest({m: p for m in ("openai/gpt-4.1", "deepseek/deepseek-v4.1-flash") if (p := pricing.get(m))}),
    )
    store.write_manifest(manifest)
    budget = Budget(args.max_usd)
    res = await run_matrix(
        jobs, configs, store=store, budget=budget, pricing=pricing, concurrency=args.concurrency,
        timeout=args.timeout, poll=args.poll,
    )
    manifest.args["auto_answers"] = res.auto_answers
    manifest.pricing = {**manifest.pricing, **used_pricing([r.usage for r in res.records], pricing)}
    store.write_manifest(manifest)
    by_status: dict[str, int] = {}
    for r in res.records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print(f"{len(res.records)} runs {by_status}, {res.skipped} not started, spent ${budget.spent:.2f} -> {store.dir}")
    return 0


def add_common(p: argparse.ArgumentParser, *, concurrency: int = 2) -> None:
    p.add_argument("--dataset", default="custom", help="custom | frames | path/to/questions.jsonl")
    p.add_argument("--limit", type=int, default=None, help="sample this many questions (seeded)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-usd", type=float, default=None, help="stop starting runs once measured spend reaches this")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the estimated cost, then exit")
    p.add_argument("--timeout", type=float, default=900, help="seconds per run before it is cancelled")
    p.add_argument("--concurrency", type=int, default=concurrency, help="runs in flight at once")
    p.add_argument("--out", type=Path, default=DEFAULT_ROOT, help=f"results root (default {DEFAULT_ROOT})")
    p.add_argument("--resume", metavar="TAG", default=None, help="continue <out>/<experiment>/<TAG>, skipping done runs")
    p.add_argument("--offline", action="store_true", help="fallback price table (no pricing fetch)")


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "quality", help="Claim A: run questions x configs x repeats on LiveGraph -> runs.jsonl",
        description="Configs:\n" + cfgs.describe(), formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--configs", default="full", help=f"comma list of {', '.join(cfgs.CONFIGS)}")
    add_common(p)
    p.add_argument("--poll", type=float, default=2.0, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_quality)
