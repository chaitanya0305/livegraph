from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_smart: str = "openai/gpt-4.1"
    model_fast: str = "deepseek/deepseek-v4.1-flash"
    # OpenRouter provider slugs tried first for each tier (falls back to others); "" = OpenRouter default routing
    provider_smart: str = ""
    provider_fast: str = "baidu/fp8"
    # Hidden reasoning on the fast tier (DeepSeek reasons by default and it ~4x the tokens/latency)
    reasoning_fast: bool = False
    # Hidden reasoning on the smart tier; None = model default (DeepSeek reasons by default, which made the
    # planner take minutes). Leave unset for non-reasoning models such as gpt-4.1.
    reasoning_smart: bool | None = None
    # Fast-tier node types that reason anyway (comma-separated), e.g. the debate agents
    reasoning_nodes: str = "prosecutor,defender"
    # How the fast tier returns structured output: "prompted" (JSON in the text stream) or "tool" (tool-call
    # arguments). DeepSeek on OpenRouter generates tool-call arguments ~3x slower than plain text.
    output_mode_fast: str = "prompted"
    embed_model: str = "openai/text-embedding-3-small"
    embed_dim: int = 1536
    web_search: bool = True
    # Web search backend: "exa" (Exa /search API), "openrouter" (OpenRouter web plugin) or "auto" (exa when
    # EXA_API_KEY is set, else openrouter)
    web_provider: str = "auto"
    exa_api_key: str = ""
    exa_base_url: str = "https://api.exa.ai"
    exa_timeout: float = 20.0
    database_url: str = "postgresql+asyncpg://livegraph:livegraph@localhost:5433/livegraph"
    cors_origins: str = "http://localhost:3000"
    max_concurrency: int = 4
    # Multi-agent nodes. How tool-using fast-tier agents return structured output: "prompted" (JSON text,
    # safe inside tool loops with end_strategy="graceful") or "tool" (a `submit` output tool).
    output_mode_agents: str = "prompted"
    # Node types that run as tool-using agents (comma-separated). Others fall back to single-shot calls.
    agent_tools: str = "research,prosecutor,defender,judge,critic,plan"
    # Ablation knob (evaluation): node types never added to a graph (comma-separated, from DISABLABLE_TYPES);
    # the default shape is rewired around them (e.g. claims -> editor without the debate). Empty = off.
    disabled_node_types: str = ""
    research_seed_k: int = 4  # chunks the engine retrieves up front as a research agent's starting evidence
    search_k_max: int = 8  # cap on `k` for the search_documents tool
    # soft cap on web_search calls per research node (RESEARCH_LOOP=false; the loop uses research_web_max)
    web_search_max_per_node: int = 2
    # Web pages: ingest each result's page text (Exa `contents.text`, up to web_page_chars) instead of only its
    # highlight snippets, so agents search and read whole pages. false = highlights only (pre-loop behaviour).
    web_page_text: bool = True
    web_page_chars: int = 12_000
    fetch_page: bool = False  # enable the fetch_page tool (research, planner)
    fetch_max_bytes: int = 400_000
    fetch_timeout: float = 10.0
    judge_max_rebuttals: int = 2  # ask_rebuttal calls per judge run
    # Context store + history compaction: once an agent's message history is above this estimate (chars/4),
    # older long tool results are replaced by one-line stubs the agent can recall(ref); the last
    # history_keep_recent model requests always stay verbatim.
    history_budget_tokens: int = 6000
    history_keep_recent: int = 2
    # cross-agent read tools (list_agent_runs, read_agent_output, read_agent_trace, recall, search_context)
    context_tools: bool = True
    # also give them to the single-shot claims/editor nodes (they then run as tool-using agents); off
    context_tools_single_shot: bool = False
    # request_upstream_work (judge, prosecutor, defender): at most this many agent requests per run, one per
    # (requester, target) pair; each becomes a steer of the target ("Request from <key>: ...")
    max_upstream_requests: int = 3
    node_timeout_scale: float = 1.0  # multiplies every per-node agent timeout
    db_pool_size: int = 10
    db_max_overflow: int = 20
    # Logfire: write token sends traces; read token is only for querying traces (never used by the app)
    logfire_token: str = ""
    logfire_read_token: str = ""
    logfire_environment: str = "dev"


def tools_enabled(node_type: str) -> bool:
    """AGENT_TOOLS kill switch: True when ``node_type`` runs as a tool-using agent."""
    return node_type in {t.strip() for t in settings.agent_tools.split(",") if t.strip()}


DISABLABLE_TYPES = ("critic", "prosecutor", "defender", "judge", "human_input")


def disabled_types() -> set[str]:
    """DISABLED_NODE_TYPES ablation knob: node types the planner/engine never add (unknown names ignored)."""
    return {t.strip() for t in settings.disabled_node_types.split(",") if t.strip()} & set(DISABLABLE_TYPES)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
