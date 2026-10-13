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

## Stack
- **api**: FastAPI, Pydantic AI, OpenRouter (`openai/gpt-4.1`, `deepseek/deepseek-v4.1-flash`), SQLAlchemy async, Postgres with pgvector
- **web**: Next.js, TypeScript, React Flow, zustand, Tailwind
