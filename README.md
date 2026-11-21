# LiveGraph

A live, multi-viewer workflow canvas for multi-agent research. You type a goal, and a planner agent
builds the workflow graph on screen. Every step streams its output as it runs. You can steer, edit,
rerun, restore or veto any step, and distrust or pin a source; only the steps affected by the change
recompute. A timeline scrubber replays the whole run. The first workflow is **Living Brief**:
research → claims → prosecutor vs defender → judge (per-claim confidence) → editor.

Spec: [`docs/CONTRACT.md`](docs/CONTRACT.md)

## Run it
```bash
cp apps/api/.env.example apps/api/.env      # add OPENROUTER_API_KEY
cd apps/web && npm install && cd ../..
make dev                                    # postgres :5433, api :8000, web :3000
```
To demo the UI without a backend or API key: `cd apps/web && npm run dev:mock`.

## Run it in Docker
```bash
cp apps/api/.env.example apps/api/.env      # add OPENROUTER_API_KEY (and LOGFIRE_TOKEN if wanted)
make up                                     # builds livegraph-api + livegraph-web, starts db/api/web
make logs                                   # follow api + web logs
make down
```
Then open http://localhost:3000. Keys are passed into the api container from `apps/api/.env` at runtime
and are never copied into an image. `NEXT_PUBLIC_API_URL` is baked into the web image at build time
(default `http://localhost:8000`); rebuild with a different `args` value in `docker-compose.yml` to host elsewhere.

## Test
```bash
make test     # backend pytest (offline, fake models) and web typecheck
cd apps/web && npm test
```

## Evaluation
`lgeval` (in `apps/api/lgeval`) is the evaluation harness. It drives LiveGraph over its REST API and measures:
- **Quality**: factual precision (FActScore/SAFE-style atomic facts), citation precision/recall (ALCE-style),
  answer accuracy on FRAMES, and position-swapped pairwise preference, against ablations (`no_debate`,
  `no_critic`, `single_shot`, `no_context_tools`) and a single-agent baseline with the same models and search.
- **Calibration**: whether the judge's per-claim confidence predicts truth (Brier, ECE, AUROC).
- **Incremental recompute**: cost and time of an update (distrust a source, steer, edit, identical rerun) as a
  share of starting over, and how many nodes were reused.

```bash
make eval-test                                        # offline harness tests (= cd apps/api && uv run pytest lgeval/tests -q)
cd apps/api
uv run python -m lgeval plan --suite pilot            # dry-run: runs and estimated cost (pilot ≈ $10–20)
uv run python -m lgeval quality --dataset custom --configs full,no_debate --repeats 3 --limit 10 --max-usd 20
uv run python -m lgeval recompute --from ../../eval-results/quality/<tag> --mutations distrust,steer,edit,identical --max-usd 10
uv run python -m lgeval score --results ../../eval-results --metrics factual,citation,accuracy,calibration --judge-model <id>
uv run python -m lgeval report --results ../../eval-results --out ../../docs/results
```
Protocol (hypotheses, metrics, judge-bias controls, budget, user study): [`docs/EVAL.md`](docs/EVAL.md).
Results: [`docs/results/results.md`](docs/results/results.md) (numbers in `docs/results/summary.json`).

| System | Factual precision | Citation precision | Citation recall | Cost / run | Update cost (distrust) |
|---|---|---|---|---|---|
| `full` | results pending | results pending | results pending | ≈ $0.011 (est., DeepSeek Flash via baidu/fp8) | results pending |
| `single_agent` | results pending | results pending | results pending | results pending | — |

## Stack
- **api**: FastAPI, Pydantic AI, OpenRouter (`openai/gpt-4.1`, `deepseek/deepseek-v4.1-flash`), SQLAlchemy async, Postgres with pgvector
- **web**: Next.js, TypeScript, React Flow, zustand, Tailwind
