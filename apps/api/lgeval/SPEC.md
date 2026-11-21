# lgeval — Evaluation Harness Contract

The single source of truth for the evaluation harness. Every module is built against it.
`lgeval` is a sibling package of `livegraph` in `apps/api/` and runs in the same uv environment.

## What we want to show (claims)
- **A. Quality**: briefs are more factual and better cited than ablations and external baselines.
- **B. Calibration**: the judge's per-claim confidence predicts whether a claim is true.
- **C. Incremental recompute**: after a change (distrust a source, steer, edit, identical rerun), LiveGraph
  updates the brief at a fraction of the cost and time of starting over, with no quality loss.

## Hard rules
- `lgeval` talks to LiveGraph **only over the REST/WS API** (see `docs/CONTRACT.md`). It never imports
  `livegraph.engine`/`livegraph.db` at runtime. (Exception: ablation knobs added to `livegraph` itself,
  which must be env-driven, default-off, and covered by the existing `tests/` suite.)
- Tests: `cd apps/api && uv run pytest lgeval/tests -q` must pass **with no network, no API key and no
  Postgres**. Use `httpx.MockTransport` for the API and fake LLM callables for scorers.
- Spending guard: every command that calls a real LLM or starts real runs accepts `--max-usd` and
  `--dry-run`. `--dry-run` prints the plan (runs × configs × repeats) and the estimated cost, then exits.
  A run stops starting new work once the measured spend (from `costs.py`) reaches `--max-usd`.
- No new heavy dependencies. stdlib + httpx + pydantic (+ pydantic-ai, already present) only. Charts are
  written as standalone SVG strings (no matplotlib).
- Do not `git commit`. Do not edit files owned by another module (table below) — if you need something
  from another module, code against the interface below; if it is missing, note it in your final report.
- Match the style of `livegraph/`: `from __future__ import annotations`, type hints, short module docstring,
  sparse comments.

## Layout and ownership
| Path | Owner | Purpose |
|---|---|---|
| `SPEC.md`, `schema.py`, `__init__.py` | lead | this contract; result record models (Pydantic) |
| `client.py` | foundation | async REST/WS client for the LiveGraph API |
| `server.py` | foundation | start/stop an API subprocess with env overrides on a free port |
| `costs.py` | foundation | OpenRouter pricing (fetch + cache + fallback table), token → USD, budget guard |
| `store.py` | foundation | results directory layout, JSONL read/write, manifest |
| `datasets/` | foundation | question sets: loader, `custom.jsonl` template, FRAMES loader |
| `__main__.py`, `cli.py` | foundation | `python -m lgeval <command>`; other modules register subcommands via `register(subparsers)` |
| `runners/quality.py` | runners | Claim A run matrix: questions × configs × repeats → `RunRecord`s |
| `runners/configs.py` | runners | named system configs (full, no_debate, no_critic, single_shot, ...) as env overrides |
| `runners/baseline.py` | runners | single-agent baseline (same models + same search) and import of external briefs |
| `livegraph/` ablation knobs | runners | e.g. `DISABLED_NODE_TYPES` (env, default empty) — minimal, tested |
| `runners/recompute.py` | recompute | Claim C benchmark → `RecomputeRecord`s |
| `metrics/` | metrics | factual precision, citation precision/recall, answer accuracy, pairwise preference, calibration |
| `report.py` | report | results → markdown tables + SVG charts (`docs/results/`) |
| `docs/EVAL.md`, `docs/paper/`, README section | report | methodology protocol, paper skeleton, results section |

## Interfaces (code against these)
```python
# client.py
class LiveGraphClient:
    def __init__(self, base_url: str = "http://localhost:8000", *, transport: httpx.AsyncBaseTransport | None = None,
                 timeout: float = 30.0): ...
    async def __aenter__(self) -> LiveGraphClient: ...; async def __aexit__(self, *exc) -> None: ...
    async def health(self) -> bool
    async def create_run(self, goal: str, document_ids: list[str] | None = None) -> dict   # RunSnapshot
    async def snapshot(self, run_id: str) -> dict                                          # RunSnapshot
    async def events(self, run_id: str, since: int = 0) -> list[dict]
    async def versions(self, node_id: str) -> list[dict]                                  # NodeVersion[] with retrievals
    async def impact(self, node_id: str, action: str) -> dict
    async def steer(self, node_id: str, text: str) -> None
    async def edit(self, node_id: str, output_text: str) -> dict
    async def rerun(self, node_id: str) -> None
    async def restore(self, node_id: str, version_id: str) -> None
    async def veto(self, node_id: str) -> None
    async def patch_node(self, node_id: str, **fields) -> dict
    async def set_trust(self, document_id: str, trust: str) -> dict
    async def upload(self, filename: str, data: bytes, run_id: str | None = None) -> dict
    async def answer(self, node_id: str, answer: str) -> None
    async def cancel(self, run_id: str) -> None
    async def wait_settled(self, run_id: str, *, timeout: float = 900, poll: float = 2.0,
                           since_seq: int = 0) -> dict
        # returns the RunSnapshot once run.status in {done, failed, cancelled, waiting_input} AND no node is
        # running/pending/stale AND last_seq has stopped moving for one poll. Raises TimeoutError otherwise.

def brief_of(snapshot: dict) -> str            # editor node's current output_text ("" if none)
def usage_of(versions_by_node: dict[str, list[dict]], *, after_seq_versions: set[str] | None = None) -> dict[str, TokenUsage]
    # sums tokens_in/out per model over the given versions (ALL versions, not only current ones)

# server.py
class ApiServer:   # async context manager; launches `uvicorn livegraph.main:app` from apps/api on a free port
    def __init__(self, env: dict[str, str] | None = None, *, port: int | None = None, log_path: Path | None = None): ...
    base_url: str
    # Runs share the dev Postgres (DATABASE_URL from apps/api/.env) — runs are isolated by run_id.

# costs.py
class Pricing: in_per_m: float; out_per_m: float
def load_pricing(models: list[str], *, cache: Path | None = None, offline: bool = False) -> dict[str, Pricing]
    # OpenRouter GET /api/v1/models, cached to cache file; offline/failed -> FALLBACK_PRICING
def cost_usd(usage: dict[str, TokenUsage], pricing: dict[str, Pricing]) -> float
class Budget:
    def __init__(self, max_usd: float | None): ...
    def add(self, usd: float) -> None; spent: float; def exhausted(self) -> bool
# Measured averages per LiveGraph run (8 real runs, Oct 2026): deepseek-v4.1-flash 72.3K in / 21.4K out,
# gpt-4.1 27.7K in / 2.7K out. The study runs both roles on deepseek-v4.1-flash (runners.configs.STUDY_MODELS)
# -> ~$0.06/run. Use these in --dry-run estimates (EST_RUN_USAGE).

# store.py
class ResultStore:
    def __init__(self, root: Path, experiment: str, *, run_tag: str | None = None): ...   # root/experiment/run_tag/
    dir: Path
    def write_manifest(self, manifest: Manifest) -> None
    def append(self, name: str, record: BaseModel) -> None          # name.jsonl
    def read(self, name: str, model: type[T]) -> list[T]
    def save_snapshot(self, run_id: str, snapshot: dict) -> Path    # snapshots/<run_id>.json
    def load_snapshot(self, run_id: str) -> dict
    def done_keys(self, name: str, key: Callable[[dict], str]) -> set[str]   # for resume: skip finished work
DEFAULT_ROOT = <repo>/eval-results   (gitignored)

# datasets
def load_questions(name: str, *, limit: int | None = None, seed: int = 0) -> list[Question]
    # "custom" -> datasets/custom.jsonl ; "frames" -> google/frames-benchmark (HF datasets-server HTTP API, cached)

# metrics (each scorer takes an `llm` callable so tests can fake it)
LLM = Callable[[str, str], Awaitable[str]]      # (system, user) -> text ; real impl via OpenRouter
def make_llm(model: str, *, budget: Budget | None = None) -> LLM
```

## Result records (`schema.py`, already written — import, do not redefine)
`Question`, `TokenUsage`, `Manifest`, `RunRecord`, `RecomputeRecord`, `ScoreRecord`, `PairwiseRecord`,
`ClaimLabel`. JSONL file names: `runs.jsonl`, `recompute.jsonl`, `scores.jsonl`, `pairwise.jsonl`,
`claim_labels.jsonl`.

## Brief → citations
The editor cites claims inline as `(C1)`; each claim's `support_chunk_ids` live in the `claims` node's
current version output (`output.claims[*].support_chunk_ids`); chunk text is in that node's / research
nodes' version `retrievals`. So a brief sentence's evidence = union of the chunks of the claims it cites.
External baselines cite URLs instead; metrics must handle both (`RunRecord.citations`).

## CLI (final shape)
```
python -m lgeval plan       --suite pilot|full            # dry-run estimate of the whole study
python -m lgeval quality    --dataset custom --configs full,no_debate --repeats 3 --limit 10 --max-usd 20
python -m lgeval baseline   --dataset custom --limit 10 --max-usd 5
python -m lgeval import     --system chatgpt_dr --dir briefs/           # external briefs as RunRecords
python -m lgeval recompute  --from <results dir> --mutations distrust,steer,edit,identical --max-usd 10
python -m lgeval score      --results <dir> --metrics factual,citation,accuracy,calibration --judge-model ...
python -m lgeval pairwise   --results <dir> --a full --b no_debate --judge-model ...
python -m lgeval report     --results <dir>... --out docs/results
```
