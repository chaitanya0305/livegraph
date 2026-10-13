# LiveGraph — Build Contract

This is the single source of truth that the backend, RAG, and frontend are built against.
Change it only deliberately, and update every side when you do.

## Product in one paragraph
A user types a research goal (and optionally uploads documents). A **Planner** agent builds a
workflow graph *live* on a canvas — nodes appear one by one and start running as soon as their
inputs are ready. Every node streams its output token-by-token. The user can **steer** any node
(type an instruction), **edit** a node's output, **veto** a node, **rerun** it, **restore** an older
version, **distrust/pin** a source document, and answer **human-input** gates. Every node output is
stored as an immutable **version**; editing anything marks downstream nodes **stale** and they
**recompute live** (only nodes whose input hash changed). A **timeline scrubber** replays the run
from the persisted event log. The first built-in workflow is **Living Brief**: research → claims →
prosecutor vs defender debate → judge sets per-claim confidence → editor writes the brief.
Anyone with the run URL sees the same run live (multi-viewer via WebSocket).

## Stack
- `apps/api`: Python 3.12, uv, FastAPI, **Pydantic AI** (agents), SQLAlchemy 2 async + asyncpg,
  Postgres 16 + **pgvector**. LLMs via **OpenRouter** (OpenAI-compatible).
- `apps/web`: Next.js (App Router) + TypeScript + Tailwind + `@xyflow/react` + zustand.
- `docker-compose.yml`: `pgvector/pgvector:pg16` on host port **5433** (5432 may be taken by a local Postgres).
- NO LangChain / LangGraph.

## Config (env, `apps/api/.env`)
```
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_SMART=openai/gpt-4.1                 # planner, judge, editor
MODEL_FAST=deepseek/deepseek-v4.1-flash    # research, claims, prosecutor, defender, critic
EMBED_MODEL=openai/text-embedding-3-small  # via OpenRouter /embeddings
EMBED_DIM=1536
WEB_SEARCH=on                              # on|off — research nodes use OpenRouter web plugin
DATABASE_URL=postgresql+asyncpg://livegraph:livegraph@localhost:5433/livegraph
CORS_ORIGINS=http://localhost:3000
```
`livegraph/config.py` exposes `settings` (pydantic-settings). Tests must never need a real key:
LLM calls go through Pydantic AI `TestModel`/`FunctionModel`, and embeddings through `FakeEmbedder`.

## Python package layout (`apps/api/livegraph/`) and ownership
| Path | Owner | Purpose |
|---|---|---|
| `config.py`, `db/base.py` | scaffold (done) | settings, `Base`, engine/session factory |
| `db/models.py` | backend | runs, nodes, edges, node_versions, events, steer_messages, claims, claim_history |
| `rag/` | rag | documents, chunks, retrievals models + ingest/search/web/embeddings |
| `protocol.py` | backend | Pydantic models for every API body + event payload (mirrors §Events) |
| `events.py` | backend | `EventBus`: persist + fan-out to WebSockets |
| `engine/` | backend | scheduler, hashing, recompute, steer/cancel, human input |
| `agents/` | backend | Pydantic AI agents + prompts + `models.py` (OpenRouter model factory) |
| `api/` | backend | FastAPI routers |
| `main.py` | backend | app factory, lifespan (create extension + tables) |

### RAG module public interface (rag owns, backend calls)
```python
# livegraph/rag/__init__.py re-exports these
class RetrievedChunk(BaseModel):
    chunk_id: UUID; document_id: UUID; document_title: str; source_url: str | None
    text: str; score: float; trust: Literal["normal","distrusted","pinned"]

async def ingest_text(session, *, run_id: UUID | None, title: str, text: str,
                      source_url: str | None = None, origin: Literal["upload","web"] = "upload") -> Document
async def ingest_file(session, *, run_id: UUID | None, filename: str, data: bytes) -> Document  # .txt .md .pdf
async def search(session, *, run_id: UUID, query: str, k: int = 6) -> list[RetrievedChunk]
    # searches docs with run_id == run_id OR run_id IS NULL (global library);
    # EXCLUDES distrusted docs; pinned docs' chunks get a score boost and are always included if relevant.
async def web_research(session, *, run_id: UUID, query: str, max_results: int = 5) -> list[Document]
    # uses OpenRouter web plugin, ingests each result page/snippet as Document(origin="web"); no-op when WEB_SEARCH=off
async def record_retrieval(session, *, node_version_id: UUID, chunks: list[RetrievedChunk]) -> None
async def set_trust(session, document_id: UUID, trust: str) -> Document
def get_embedder() -> Embedder  # Embedder.embed(list[str]) -> list[list[float]]; FakeEmbedder for tests
```

## Data model (Postgres)
All ids are UUID (server-generated), all timestamps `timestamptz default now()`.

- **runs**: id, goal, workflow (`living_brief`), status `planning|running|paused|waiting_input|done|failed|cancelled`, created_at, updated_at
- **nodes**: id, run_id, key (short slug, unique per run), type, title, status, config jsonb (`{instructions, question?, params}`), position jsonb `{x,y}` (nullable — frontend auto-layouts), created_by `planner|critic|user`, current_version_id (nullable), removed bool, created_at
- **edges**: id, run_id, source_node_id, target_node_id (unique pair)
- **node_versions**: id, node_id, version int (1..n per node), input_hash, output jsonb (typed per node type), output_text (markdown rendering), reason `initial|steer|edit|recompute|retry|restore`, steer_text nullable, model, tokens_in, tokens_out, created_at
- **events**: run_id, seq (bigint, monotonically increasing per run, PK with run_id), type, node_id nullable, payload jsonb, ts
- **steer_messages**: id, node_id, text, created_at, consumed_at nullable
- **claims**: id, run_id, key (`C1`..), text, confidence float 0..1 (nullable until judged), status `active|struck`, source_node_id, updated_at
- **claim_history**: id, claim_id, confidence, node_version_id, rationale, ts
- (rag) **documents**: id, run_id nullable, title, source_url, origin `upload|web`, trust `normal|distrusted|pinned`, created_at
- (rag) **chunks**: id, document_id, idx, text, embedding vector(EMBED_DIM) (HNSW cosine index)
- (rag) **retrievals**: id, node_version_id, chunk_id, score, rank

## Node types (Living Brief)
| type | model | inputs | output (Pydantic) |
|---|---|---|---|
| `plan` | SMART | goal | `{subquestions: [str], rationale}` — and it *creates the graph* via tools |
| `research` | FAST | config.question | `{findings: [{point, chunk_ids}], gaps: [str]}` — does `web_research` + `search`, records retrievals |
| `claims` | FAST | all research | `{claims: [{key, text, support_chunk_ids}]}` — upserts `claims` rows |
| `prosecutor` | FAST | claims + evidence | `{attacks: [{claim_key, argument, chunk_ids}]}` |
| `defender` | FAST | claims + evidence | `{defenses: [{claim_key, argument, chunk_ids}]}` |
| `judge` | SMART | claims + attacks + defenses | `{verdicts: [{claim_key, confidence, rationale, strike: bool}]}` — writes claim confidence + history |
| `human_input` | — | prompt in config | `{answer: str}` — waits for user |
| `editor` | SMART | claims (with confidence) + verdicts | `{markdown}` — final brief; struck claims excluded, low-confidence hedged |
| `critic` | FAST | a research node's output | `{verdict: ok|thin, follow_up_question?}` — if thin, adds a new `research` node + edge into `claims` (graph grows mid-run) |

Planner tools (each call persists + emits an event immediately, so the graph builds on screen):
`add_node(key, type, title, question?, instructions?)`, `add_edge(source_key, target_key)`, `finish_plan()`.
Default shape: `plan → research×(3–5) → claims → {prosecutor, defender} → judge → editor`, with
`critic` attached after each research node.

## Engine semantics
- **Ready rule**: a node runs when all non-removed parents have status `done`. Independent nodes run concurrently (bounded semaphore, default 4).
- **input_hash** = sha256 of canonical JSON of: node type, config, parents' `current_version_id`s (sorted), unconsumed+consumed steer texts that apply, and for research nodes the set of (document_id, trust) the retrieval touched. If a version with the same input_hash already exists for this node → **reuse** it (set current_version_id, emit `node.version` with `reused: true`), with no LLM call.
- **Edit**: user edits a node's output → new version (reason `edit`), then all descendants → `stale`, then the scheduler recomputes them in topological order.
- **Steer**: a running node is cancelled and restarted with the steer text appended; a done node gets a new run (reason `steer`) and its descendants go stale.
- **Rerun**: forces a new version even if the hash matches (reason `retry`).
- **Restore**: sets current_version_id to the chosen old version (reason `restore` on an event, no new version row), then descendants go stale.
- **Veto/remove**: node.removed = true, status `skipped`; descendants that lose all inputs → `skipped`; the others go stale.
- **Distrust/pin document**: research nodes whose latest retrieval touched that doc → stale, followed by recompute.
- **Human input**: node status `waiting_input`, run status `waiting_input`; resumes on `POST /nodes/{id}/input`.
- **Pause**: no new nodes start; running ones finish. **Cancel**: cancel all tasks.
- Every state change goes through `EventBus.emit(...)`, which writes the `events` row **then** fans it out.

## REST API (prefix `/api`)
| Method | Path | Body → Response |
|---|---|---|
| POST | `/runs` | `{goal, workflow?="living_brief", document_ids?: []}` → `RunSnapshot` (starts planner in background) |
| GET | `/runs` | → `[RunSummary]` |
| GET | `/runs/{id}` | → `RunSnapshot` |
| GET | `/runs/{id}/events?since=0` | → `[Event]` (persisted only) |
| POST | `/runs/{id}/pause` `/resume` `/cancel` | → `RunSummary` |
| GET | `/nodes/{id}/versions` | → `[NodeVersion]` (with retrievals) |
| POST | `/nodes/{id}/steer` | `{text}` → 202 |
| POST | `/nodes/{id}/edit` | `{output_text}` → `NodeVersion` |
| POST | `/nodes/{id}/rerun` | → 202 |
| POST | `/nodes/{id}/restore` | `{version_id}` → 202 |
| POST | `/nodes/{id}/input` | `{answer}` → 202 |
| DELETE | `/nodes/{id}` | → 202 (veto) |
| POST | `/documents` | multipart `file` (+ optional `run_id`) → `Document` |
| GET | `/documents?run_id=` | → `[Document]` (global + run-scoped) |
| PATCH | `/documents/{id}` | `{trust}` → `Document` |
| GET | `/health` | → `{ok: true}` |

`RunSnapshot = {run, nodes: [Node], edges: [Edge], versions: {node_id: NodeVersion /*current*/}, claims: [Claim], documents: [Document], last_seq}`
`Node = {id, run_id, key, type, title, status, config, position, created_by, current_version_id, removed}`
`NodeVersion = {id, node_id, version, input_hash, output, output_text, reason, steer_text, model, tokens_in, tokens_out, created_at, retrievals?: [RetrievedChunk & {rank}]}`
`Claim = {id, key, text, confidence, status, history: [{confidence, rationale, ts}]}`
JSON is snake_case. Datetimes are ISO strings.

## WebSocket
`GET /api/runs/{id}/ws?since=<seq>`: the server first sends every persisted event with seq > since, then streams live events.
Envelope: `{seq: number|null, run_id, type, node_id: string|null, payload, ts}`. `seq: null` means ephemeral (not persisted, so not replayable).

### Event types
| type | persisted | payload |
|---|---|---|
| `run.status` | yes | `{status}` |
| `node.added` | yes | `{node: Node}` |
| `edge.added` | yes | `{edge: Edge}` |
| `node.status` | yes | `{status, reason?}` — statuses: `pending|running|waiting_input|done|failed|stale|skipped` |
| `node.delta` | **no** | `{text}` — streamed token chunk(s), batched ≈50ms; frontend appends to a live buffer |
| `node.version` | yes | `{version: NodeVersion, reused: bool}` (sets current) |
| `node.restored` | yes | `{version_id}` |
| `node.removed` | yes | `{}` |
| `retrieval.done` | yes | `{node_version_id, chunks: [RetrievedChunk & {rank}]}` |
| `claim.upserted` | yes | `{claim: Claim}` |
| `claim.confidence` | yes | `{claim_id, key, confidence, previous, rationale}` |
| `steer.queued` / `steer.consumed` | yes | `{steer_id, text}` |
| `input.requested` | yes | `{prompt}` |
| `document.updated` | yes | `{document: Document}` |
| `error` | yes | `{message, node_id?}` |

**Timeline/rewind (frontend)**: the state at seq N = fold(events with seq ≤ N) starting from an empty run. Persisted events alone must therefore be enough to rebuild nodes, edges, statuses, current versions, and claims. The scrubber is view-only; acting on an old version goes through `/restore`.

## Definition of done (MVP)
1. `make dev`: docker postgres, then api on :8000, then web on :3000.
2. Typing a goal on `/` → `/runs/{id}` shows nodes appearing and running live, with streaming text.
3. Steer, edit, rerun, restore, veto, and distrust-document all work and visibly recompute downstream.
4. Claims panel: confidence bars animate as the judge runs, and struck claims are crossed out.
5. Scrubber replays the run, and side-by-side diff of any two versions of a node works.
6. A second browser tab on the same URL sees everything live.
7. `uv run pytest` passes with no network and no API key.
