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
    # Fast-tier node types that reason anyway (comma-separated), e.g. the debate agents
    reasoning_nodes: str = "prosecutor,defender"
    # How the fast tier returns structured output: "prompted" (JSON in the text stream) or "tool" (tool-call
    # arguments). DeepSeek on OpenRouter generates tool-call arguments ~3x slower than plain text.
    output_mode_fast: str = "prompted"
    embed_model: str = "openai/text-embedding-3-small"
    embed_dim: int = 1536
    web_search: bool = True
    database_url: str = "postgresql+asyncpg://livegraph:livegraph@localhost:5433/livegraph"
    cors_origins: str = "http://localhost:3000"
    max_concurrency: int = 4
    # Logfire: write token sends traces; read token is only for querying traces (never used by the app)
    logfire_token: str = ""
    logfire_read_token: str = ""
    logfire_environment: str = "dev"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
