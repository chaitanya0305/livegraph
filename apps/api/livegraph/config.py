from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_smart: str = "openai/gpt-4.1"
    model_fast: str = "deepseek/deepseek-v4.1-flash"
    embed_model: str = "openai/text-embedding-3-small"
    embed_dim: int = 1536
    web_search: bool = True
    database_url: str = "postgresql+asyncpg://livegraph:livegraph@localhost:5433/livegraph"
    cors_origins: str = "http://localhost:3000"
    max_concurrency: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
