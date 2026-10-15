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
WEB_SEARCH=on                              # on|off — web_search tool on/off
WEB_PROVIDER=auto                          # exa|openrouter|auto — auto = Exa /search when EXA_API_KEY is set, else OpenRouter web plugin
EXA_API_KEY=                               # Exa API key (x-api-key header)
DATABASE_URL=postgresql+asyncpg://livegraph:livegraph@localhost:5433/livegraph
CORS_ORIGINS=http://localhost:3000
# multi-agent nodes
OUTPUT_MODE_AGENTS=prompted                # prompted|tool — how tool-using fast-tier agents return output
AGENT_TOOLS=research,prosecutor,defender,judge,critic,plan   # kill switch: other types run single-shot
RESEARCH_SEED_K=4                          # chunks retrieved up front as a researcher's starting evidence
SEARCH_K_MAX=8                             # cap on search_documents k
WEB_SEARCH_MAX_PER_NODE=2                  # researcher web_search soft budget
FETCH_PAGE=false                           # enable fetch_page (research, planner)
FETCH_MAX_BYTES=400000
FETCH_TIMEOUT=10
JUDGE_MAX_REBUTTALS=2                      # ask_rebuttal calls per judge run
HISTORY_BUDGET_TOKENS=6000                 # history compaction threshold (chars/4 estimate)
HISTORY_KEEP_RECENT=2                      # last N model requests always stay verbatim
CONTEXT_TOOLS=true                         # cross-agent read tools (list_agent_runs, read_agent_output, ...)
CONTEXT_TOOLS_SINGLE_SHOT=false            # also give them to claims/editor (they then run as tool agents)
MAX_UPSTREAM_REQUESTS=3                    # request_upstream_work: agent requests per run (one per requester+target)
NODE_TIMEOUT_SCALE=1.0                     # multiplies every per-node agent timeout
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```
`livegraph/config.py` exposes `settings` (pydantic-settings). Tests must never need a real key:
LLM calls go through Pydantic AI `TestModel`/`FunctionModel`, and embeddings through `FakeEmbedder`.

## Python package layout (`apps/api/livegraph/`) and ownership
| Path | Owner | Purpose |
|---|---|---|
| `config.py`, `db/base.py` | scaffold (done) | settings, `Base`, engine/session factory |
| `db/models.py` | backend | runs, nodes, edges, node_versions, events, steer_messages, claims, claim_history, context_items |
| `db/migrate.py`, `../migrations/` | backend | Alembic: `migrate()` at startup (advisory lock, stamp pre-Alembic DB at `0001`, `upgrade head`); one revision per schema change |
| `rag/` | rag | documents, chunks, retrievals models + ingest/search/web/embeddings |
| `protocol.py` | backend | Pydantic models for every API body + event payload (mirrors §Events) |
| `events.py` | backend | `EventBus`: persist + fan-out to WebSockets |
| `engine/` | backend | scheduler, hashing, recompute, steer/cancel, human input |
| `agents/` | backend | Pydantic AI agents + prompts + `models.py` (OpenRouter model factory); `runner.py` (`run_agent`: `agent.iter()` + streaming), `deps.py` (`NodeDeps`), `guard.py` (`run_tool`: budgets, supersede guard, context store write, tool events, untrusted wrapping), `context.py` (context store), `history.py` (history compaction capability), `validators.py` (output validators), `tools/` (rag, graph, debate, context tools) |
| `trace.py` | backend | folds tool events into `TraceStep`s (shared by agents and the snapshot) |
| `api/` | backend | FastAPI routers |
| `main.py` | backend | app factory, lifespan (Alembic migrations, crash recovery) |

### RAG module public interface (rag owns, backend calls)
```python
# livegraph/rag/__init__.py re-exports these
class RetrievedChunk(BaseModel):
    chunk_id: UUID; document_id: UUID; document_title: str; source_url: str | None
    text: str; score: float; trust: Literal["normal","distrusted","pinned"]

async def ingest_text(session, *, run_id: UUID | None, title: str, text: str,
                      source_url: str | None = None, origin: Literal["upload","web"] = "upload") -> Document
async def ingest_file(session, *, run_id: UUID | None, filename: str, data: bytes) -> Document  # .txt .md .pdf
async def search(session, *, run_id: UUID, query: str, k: int = 6,
                 document_ids: list[UUID] | None = None) -> list[RetrievedChunk]  # document_ids: restrict to these
    # searches docs with run_id == run_id OR run_id IS NULL (global library);
    # EXCLUDES distrusted docs; pinned docs' chunks get a score boost and are always included if relevant.
async def web_research(session, *, run_id: UUID, query: str, max_results: int = 5) -> list[Document]
    # uses Exa /search (highlights) or the OpenRouter web plugin per WEB_PROVIDER, ingests each result page/snippet as Document(origin="web"); no-op when WEB_SEARCH=off
async def record_retrieval(session, *, node_version_id: UUID, chunks: list[RetrievedChunk],
                           cited: set[str] | None = None, via: dict[str, str] | None = None) -> None
async def set_trust(session, document_id: UUID, trust: str) -> Document
async def fetch_citations(query, max_results=5) -> list[dict]       # web_research's network half (no DB)
async def ingest_citations(session, *, run_id, citations) -> list[Document]  # its DB half (caller serializes)
async def get_chunk_window(session, *, run_id, chunk_id, before=1, after=1) -> list[RetrievedChunk]
async def resolve_chunk_id(session, *, run_id, ref: str) -> UUID | None   # full UUID or 8+ char prefix
async def corpus_fingerprint(session, run_id) -> list[tuple[str, str]]     # see input_hash
def get_embedder() -> Embedder  # Embedder.embed(list[str]) -> list[list[float]]; FakeEmbedder for tests
```

## Data model (Postgres)
All ids are UUID (server-generated), all timestamps `timestamptz default now()`.

- **runs**: id, goal, workflow (`living_brief`), status `planning|running|paused|waiting_input|done|failed|cancelled`, created_at, updated_at
- **nodes**: id, run_id, key (short slug, unique per run), type, title, status, config jsonb (`{instructions, question?, params}`), position jsonb `{x,y}` (nullable — frontend auto-layouts), created_by `planner|critic|user`, current_version_id (nullable), removed bool, created_at
- **edges**: id, run_id, source_node_id, target_node_id (unique pair)
- **node_versions**: id, node_id, version int (1..n per node), input_hash, output jsonb (typed per node type), output_text (markdown rendering), reason `initial|steer|edit|recompute|retry|restore`, steer_text nullable, model, tokens_in, tokens_out, trace jsonb not null default `[]` (`TraceStep`s of the attempt that produced it), output_hash varchar(64) nullable (content hash, see early cutoff; NULL for versions older than `0003`), created_at
- **events**: run_id, seq (bigint, monotonically increasing per run, PK with run_id), type, node_id nullable, payload jsonb, ts
- **steer_messages**: id, node_id, text, source varchar(32) not null default `'user'` (`user` or `agent:<node_key>` for a `request_upstream_work`), created_at, consumed_at nullable
- **claims**: id, run_id, key (`C1`..), text, confidence float 0..1 (nullable until judged), status `active|struck`, source_node_id, updated_at
- **claim_history**: id, claim_id, confidence, node_version_id, rationale, ts
- (rag) **documents**: id, run_id nullable, title, source_url, origin `upload|web`, trust `normal|distrusted|pinned`, created_at; partial unique index `(run_id, source_url) WHERE source_url IS NOT NULL`
- (rag) **chunks**: id, document_id, idx, text, embedding vector(EMBED_DIM) (HNSW cosine index)
- (rag) **retrievals**: id, node_version_id, chunk_id, score, rank, cited bool (the output cites it; cited rows rank first), via `seed|search|read|web` (nullable)
- (rag) **web_cache**: id, run_id, query_norm (unique per run), doc_ids jsonb, created_at — web results per normalized query
- **context_items**: PK (run_id, ref `ctx_` + 8 hex), node_id, attempt_id, node_version_id (nullable: backfilled when the attempt's version persists; NULL forever for superseded attempts), agent, parent_ref (the `ask_rebuttal` call of a nested result), kind `tool_result|rebuttal`, tool, args jsonb, ok, text (full result the model saw), summary (deterministic stub line incl. full chunk ids), chunk_ids jsonb, tokens_est, created_at; index (run_id, node_version_id), GIN `to_tsvector('english', text)`

Schema changes are Alembic revisions (`apps/api/migrations/versions`, `make db-rev m=...`; `make db-check`
fails when one is missing). `0001_baseline` is the pre-Alembic schema (existing databases are stamped at it
on startup), `0002_context_items` the context store, `0003_traversal` `node_versions.output_hash` +
`steer_messages.source`. The API runs `upgrade head` at startup.

## Node types (Living Brief)
| type | model | inputs | output (Pydantic) |
|---|---|---|---|
| `plan` | SMART | goal | `{subquestions: [str], rationale}` — and it *creates the graph* via tools |
| `research` | FAST | config.question | `{findings: [{point, chunk_ids}], gaps: [str]}` — agent: starts from `RESEARCH_SEED_K` seeded chunks, gathers more with tools; every chunk it saw is recorded as retrievals |
| `claims` | FAST | all research | `{claims: [{key, text, support_chunk_ids}]}` — upserts `claims` rows |
| `prosecutor` | FAST | claims + evidence | `{attacks: [{claim_key, argument, chunk_ids}]}` |
| `defender` | FAST | claims + evidence | `{defenses: [{claim_key, argument, chunk_ids}]}` |
| `judge` | SMART | claims + attacks + defenses | `{verdicts: [{claim_key, confidence, rationale, strike: bool}], rebuttals: [{claim_key, side, question, answer, chunk_ids}]}` — writes claim confidence + history; rebuttals rendered after the verdicts |
| `human_input` | — | prompt in config | `{answer: str}` — waits for user |
| `editor` | SMART | claims (with confidence) + verdicts | `{markdown}` — final brief; struck claims excluded, low-confidence hedged |
| `critic` | FAST | a research node's output | `{verdict: ok|thin, follow_up_question?, reason}` — if thin it calls `request_follow_up_research`, which adds a `research` node (`{key}_fu`) + edge critic→it; the edge into `claims` is added once the critic's version persists (graph grows mid-run) |

Planner tools (each call persists + emits an event immediately, so the graph builds on screen):
`add_node(key, type, title, question?, instructions?)`, `add_edge(source_key, target_key)`, `finish_plan()`
(always on), plus `list_documents()`, `search_documents(query, k)` and (`FETCH_PAGE=true`) `fetch_page(url)`.

### Agents and tools
Six tool-using agents (Planner, Researcher, Critic, Prosecutor, Defender, Judge) and two validated single
shots (Claims, Editor). The engine stays the (non-LLM) orchestrator. Soft budgets: past its cap a tool
answers `BUDGET EXHAUSTED for X ...; finalize now` instead of failing; `request_limit` is a hard Pydantic AI
`UsageLimits` cap; the timeout is wall-clock (`× NODE_TIMEOUT_SCALE`).

| Agent | Tools | Soft budgets / request_limit / timeout |
|---|---|---|
| Planner | add_node, add_edge, finish_plan, list_documents, search_documents, (fetch_page) | list 1, search 2, fetch 1 / 20 / 90s |
| Researcher | web_search, search_documents, read_source, (fetch_page) | web `WEB_SEARCH_MAX_PER_NODE`, search 5, read 6, fetch 2 / 12 / 120s |
| Critic | request_follow_up_research | 1 / 4 / 45s |
| Prosecutor, Defender | search_documents, web_search, read_source, request_upstream_work | web 1, search 4, read 3 / 8 / 90s |
| Judge | read_source, ask_rebuttal, request_upstream_work | read 6, rebuttal `JUDGE_MAX_REBUTTALS` / 10 + 4×rebuttals / 150s |
| Claims, Editor | — (output validators only; context tools with `CONTEXT_TOOLS_SINGLE_SHOT=true`) | — / 3 / 60s, 90s |

Researcher, Critic, Prosecutor, Defender and Judge also get the **context tools** (not the planner, which
runs first): `list_agent_runs()` (budget 2), `read_agent_output(node_key, offset=0)` (4000-char pages; 4),
`read_agent_trace(node_key)` (3), `recall(ref, offset=0)` (6), `search_context(query, k=5)` (Postgres FTS; 3).
Scope rule: an agent reads only its own attempt and the current versions of its transitive active ancestors
(anything else: `not readable: <key> is not upstream of you`), so everything it reads is covered by its input
hash. Results are `<source …>` data. `ask_rebuttal` debaters get recall 2 / search 1.

- `search_documents(query, k≤SEARCH_K_MAX)`, `read_source(chunk_id)` (chunk + neighbours; full UUID or 8-char prefix; run scope + distrust respected), `web_search(query)` (Exa /search or OpenRouter web plugin → web documents → best chunks of those docs), `fetch_page(url)` (public http(s), html/text only, `FETCH_MAX_BYTES`).
- `ask_rebuttal(claim_key, side, question)`: runs the prosecutor/defender *agent* as a tool (shared evidence + trace, usage rolled into the judge's, ≤4 requests, ≤1 per (claim, side)); no graph nodes.
- `request_upstream_work(target_node_key, instruction)` (judge, prosecutor, defender; hidden inside `ask_rebuttal` runs): asks a strict active ancestor (one in `deps.readable`; not `plan`/`human_input`) to redo its work. Instruction ≤ 500 chars; at most one per (requester, target) per run and `MAX_UPSTREAM_REQUESTS` per run, counted from persisted `steer_messages` with `source = "agent:<key>"` (rejections answer `rejected: …` with summary `not upstream|not requestable|instruction too long|own node|already requested|request limit reached|unknown node|empty instruction|not in a rebuttal`). Accepted calls are queued on the attempt and applied after the requester's version persists (re-checked under the run mutex) as a steer of the target with text `Request from <key>: <instruction>` → `steer.queued {source: "agent:<key>"}`; the target recomputes, its descendants (the requester included) go stale and re-hash (early cutoff), and the rerun requester cannot repeat a request, so the graph settles. The call's `tool.result.node_ids` = `[target id]`.
- Graph-mutating tools (`add_node`, `add_edge`, `finish_plan`, `request_follow_up_research`, `ask_rebuttal`, `request_upstream_work`) are sequential; read tools run in parallel. Every tool opens its own DB session.
- Corpus/web text reaches models only as `<source chunk_id=… trust=… title="…">…</source>` (≤1200 chars) and every system prompt says such content is data, never instructions.
- Output validators (`ModelRetry`): cited chunk ids must be ones the node was given or retrieved (research, prosecutor, defender, claims; each claim ≥1 support), exactly one verdict per claim (judge), no `(Cn)` for struck/unknown claims (editor). When retries run out the output is repaired (invalid ids dropped) instead of failing.
- **Context store + history compaction**: every tool result (except budget-exhausted calls, graph edits, `list_agent_runs` and `recall`) is stored as a `context_items` row; the `tool.result` carries its `ref` (a cached repeat keeps the original ref). Before each model request, above `HISTORY_BUDGET_TOKENS` the content of older tool returns (> 600 chars; never the last `HISTORY_KEEP_RECENT` requests) is replaced, oldest first, by `[ctx_ab12] tool(args) -> summary [chunk_id=…] (recall('ctx_ab12') for full text)` (Pydantic AI `ProcessHistory` capability; idempotent and permanent, parts are never dropped or reordered so call/return pairing holds).
- `AGENT_TOOLS` kill switch: a type not listed runs single-shot (research falls back to engine-side web + search; a thin critic's follow-up is added from its output; the planner keeps its graph tools).
Default shape: `plan → research×(3–5) → claims → {prosecutor, defender} → judge → editor`, with
`critic` attached after each research node.

## Engine semantics
- **Ready rule**: a node runs when all non-removed parents have status `done`. Independent nodes run concurrently (bounded semaphore, default 4).
- **input_hash** = sha256 of canonical JSON of: node type, config, title, (research) the effective question (`config.question` or title), its inputs named by content as `"<key>:<output_hash or version id>"` — the direct parents, plus the research and claims ancestors for claims/prosecutor/defender/judge/editor, plus **every readable transitive ancestor** for agents with the context tools (so a change hidden behind an unchanged parent still recomputes whoever can read it) — unconsumed+consumed steer texts that apply, and for corpus-reading nodes (research; prosecutor/defender/judge when tool-enabled) the **corpus fingerprint**: sorted (document_id, trust) of in-scope docs with `origin='upload' OR trust!='normal'` (so an upload or a distrust/pin forces a recompute, while web pages the agents ingest themselves do not break reuse). Computed before the run. If a version with the same input_hash already exists for this node → **reuse** it (set current_version_id, emit `node.version` with `reused: true`), with no LLM call.
- **output_hash** = sha256 of canonical JSON of the version's output, its whitespace-normalized `output_text` and the sorted `(chunk_id, cited)` retrievals (edits/answers: no retrievals). **Early cutoff**: when a node's current version changes to one with the same `output_hash`, nothing downstream is staled; descendants that were already stale (steer/edit/restore/PATCH stale them up front) re-hash, find their current version again and go back to `done` with `reason: "upstream unchanged"` (a `node.version` with `reused: true` is still emitted). An identical rerun therefore stops at the rerun node.
- **Impact preview**: `GET /nodes/{id}/impact?action=steer|rerun|config|edit|restore|veto` → `{action, node_ids, skipped_ids, upper_bound: true}`, read-only (no events). `node_ids` in topological order: steer/rerun/config = the node + its active descendants; edit/restore = descendants only; veto = descendants that keep an active input (`skipped_ids` = the ones left without inputs). An upper bound: early cutoff can stop sooner.
- **PATCH node**: `PATCH /nodes/{id}` `{title?, question? (research only, else 422), instructions?}` (`""` clears question/instructions; nothing given → 422) → `Node`. Merges into config/title, emits `node.updated {node}`, cancels the node if running, stales it and its descendants, and lets its next run search the web live once (fresh sources for a new question). No force: patching a field back re-hashes to the old input hash and reuses the old versions (no LLM, no network). A patch that changes nothing emits nothing.
- **Agent attempt**: every transition to `running` gets a fresh `attempt_id` (on `node.status`). Tool events carry it and are emitted only while that attempt still owns the node; a superseded attempt (steer/rerun/edit/restore/veto/cancel, even mid-tool) never emits `tool.result` or writes anything. The final successful attempt's steps are stored as `node_versions.trace`.
- **Retrievals**: for every tool-using node, everything the agent saw (seed, search, read, web) is recorded on the new version (cited chunks first, `cited`/`via` set) and announced with `retrieval.done`. Downstream nodes get their research ancestors' evidence, cited chunks first.
- **Web determinism**: `web_search` reads `web_cache` (per run, normalized query); only a first run, a steer or a Rerun may hit the network, so recomputes are reproducible and free. Web ingest is serialized per run (+ the unique index), so parallel agents never duplicate a page.
- **Edit**: user edits a node's output → new version (reason `edit`), then all descendants → `stale` (unless the edited content equals the previous version's: early cutoff), then the scheduler recomputes them in topological order.
- **Steer**: a running node is cancelled and restarted with the steer text appended; a done node gets a new run (reason `steer`) and its descendants go stale.
- **Rerun**: forces a new version even if the hash matches (reason `retry`).
- **Restore**: sets current_version_id to the chosen old version (reason `restore` on an event, no new version row), then descendants go stale.
- **Veto/remove**: node.removed = true, status `skipped`; descendants that lose all inputs → `skipped`; the others go stale.
- **Distrust/pin document**: research/prosecutor/defender/judge nodes whose current version's retrievals touched that doc → stale, followed by recompute (their hash includes the corpus fingerprint, so they do not reuse).
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
| GET | `/nodes/{id}/impact?action=steer\|rerun\|config\|edit\|restore\|veto` | → `{action, node_ids: [id] /*topological*/, skipped_ids: [id], upper_bound: true}` (read-only) |
| PATCH | `/nodes/{id}` | `{title?, question?, instructions?}` → `Node` (emits `node.updated`; node + descendants recompute) |
| POST | `/nodes/{id}/steer` | `{text}` → 202 |
| POST | `/nodes/{id}/edit` | `{output_text}` → `NodeVersion` |
| POST | `/nodes/{id}/rerun` | → 202 |
| POST | `/nodes/{id}/restore` | `{version_id}` → 202 |
| POST | `/nodes/{id}/input` | `{answer}` → 202 |
| DELETE | `/nodes/{id}` | → 202 (veto) |
| POST | `/documents` | multipart `file` (+ optional `run_id`) → `Document` |
| GET | `/documents?run_id=` | → `[Document]` (global + run-scoped) |
| PATCH | `/documents/{id}` | `{trust}` → `Document` |
| GET | `/runs/{id}/context/{ref}` | → `ContextItem` (a stored tool result in full; inspector use) |
| GET | `/health` | → `{ok: true}` |

`RunSnapshot = {run, nodes: [Node], edges: [Edge], versions: {node_id: NodeVersion /*current*/}, claims: [Claim], documents: [Document], last_seq, activity: {node_id: [TraceStep]}, attempts: {node_id: attempt_id}}`
— `activity`/`attempts` cover nodes currently `running`: the steps of their current attempt, folded from persisted tool events with seq ≤ last_seq.
`Node = {id, run_id, key, type, title, status, config, position, created_by, current_version_id, removed}`
`NodeVersion = {id, node_id, version, input_hash, output_hash /*null for pre-0003 versions*/, output, output_text, reason, steer_text, model, tokens_in, tokens_out, created_at, retrievals?: [RetrievedChunk & {rank, cited, via}], trace: [TraceStep]}`
`TraceStep = {call_id, tool, agent, parent_call_id, args_summary, ok /*null only while in flight (activity)*/, summary, chunk_ids: [str], node_ids: [str], ms, ts, exchange: [{from_agent, to_agent, kind: question|answer, text}], ref: str|null /*context store ref*/, reads: [{node_key}] /*upstream agents a context tool read*/}`
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
| `node.status` | yes | `{status, reason?, attempt_id?}` — statuses: `pending|running|waiting_input|done|failed|stale|skipped`; `attempt_id` is new on every `running` |
| `node.delta` | **no** | `{text, reset?}` — streamed chunk(s) of the node's *final output* rendering, batched ≈50ms; the frontend appends to a live buffer, or replaces it with `text` when `reset: true` (streamed output superseded: the run continued with tools or a validator asked for a retry). Tool text never goes into deltas |
| `node.version` | yes | `{version: NodeVersion, reused: bool}` (sets current) |
| `node.restored` | yes | `{version_id}` |
| `node.removed` | yes | `{}` |
| `node.updated` | yes | `{node: Node}` — title/config after `PATCH /nodes/{id}` (status changes follow as `node.status`) |
| `retrieval.done` | yes | `{node_version_id, chunks: [RetrievedChunk & {rank}]}` |
| `claim.upserted` | yes | `{claim: Claim}` |
| `claim.confidence` | yes | `{claim_id, key, confidence, previous, rationale}` |
| `steer.queued` / `steer.consumed` | yes | `{steer_id, text, source}` — `source` = `user` or `agent:<node_key>` (a downstream agent's `request_upstream_work`; text `Request from <key>: …`) |
| `input.requested` | yes | `{prompt}` |
| `document.updated` | yes | `{document: Document}` |
| `tool.call` | yes | `{attempt_id, call_id, tool, args, args_summary, agent, parent_call_id, step, budget}` — `agent` = node type of the calling agent (e.g. `prosecutor` inside the judge's `ask_rebuttal`), `parent_call_id` = that ask_rebuttal call; `step` = 1-based count of this tool's calls by that agent in the attempt, `budget` = its soft cap (null = none) |
| `tool.result` | yes | `{attempt_id, call_id, ok, summary, error, ms, chunks: [{chunk_id, document_id, title, score, trust, snippet}], node_ids: [str], ref, reads: [{node_key}]}` — `ref` = context store ref of the stored result (null when not stored) |
| `agent.message` | yes | `{attempt_id, call_id, from_agent, to_agent, kind: question|answer, text, chunk_ids}` — the judge↔debater exchange of an `ask_rebuttal` call |
| `error` | yes | `{message, node_id?}` |

**Timeline/rewind (frontend)**: the state at seq N = fold(events with seq ≤ N) starting from an empty run. Persisted events alone must therefore be enough to rebuild nodes, edges, statuses, current versions, claims, and each running node's tool activity (tool events whose `attempt_id` is not the node's latest one are ignored; args/summaries are truncated to ~300 chars). The scrubber is view-only; acting on an old version goes through `/restore`.

## Definition of done (MVP)
1. `make dev`: docker postgres, then api on :8000, then web on :3000.
2. Typing a goal on `/` → `/runs/{id}` shows nodes appearing and running live, with streaming text.
3. Steer, edit, rerun, restore, veto, and distrust-document all work and visibly recompute downstream.
4. Claims panel: confidence bars animate as the judge runs, and struck claims are crossed out.
5. Scrubber replays the run, and side-by-side diff of any two versions of a node works.
6. A second browser tab on the same URL sees everything live.
7. `uv run pytest` passes with no network and no API key.
8. Running agent nodes show their tool steps live (Activity tab), and a version's trace replays with it.
   `apps/api/scripts/smoke_agents.py` is an opt-in live check of one research agent against OpenRouter.
