"""Offline test setup: a dedicated `livegraph_test` database, fake LLMs, fake embedder, no web search."""

from __future__ import annotations

import asyncio
import os

TEST_DB = "livegraph_test"
_ADMIN_DSN = "postgresql://livegraph:livegraph@localhost:5433/livegraph"
os.environ["DATABASE_URL"] = f"postgresql+asyncpg://livegraph:livegraph@localhost:5433/{TEST_DB}"
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["WEB_SEARCH"] = "off"
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
os.environ["LOGFIRE_TOKEN"] = ""


async def _ensure_db() -> None:
    import asyncpg

    conn = await asyncpg.connect(_ADMIN_DSN)
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB):
            await conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    finally:
        await conn.close()


asyncio.run(_ensure_db())

import httpx  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from sqlalchemy import text  # noqa: E402

import livegraph.db.models  # noqa: E402,F401
from livegraph import rag  # noqa: E402
from livegraph.config import settings  # noqa: E402
from livegraph.db import base as db_base  # noqa: E402
from livegraph.engine import engine  # noqa: E402
from livegraph.main import create_app  # noqa: E402
from tests import fakes  # noqa: E402

assert settings.database_url.endswith(f"/{TEST_DB}"), settings.database_url


@pytest_asyncio.fixture(scope="session")
async def db():
    settings.web_search = False
    async with db_base.engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(db_base.Base.metadata.drop_all)
        await conn.run_sync(db_base.Base.metadata.create_all)
    yield db_base.engine
    await engine.shutdown()
    await db_base.engine.dispose()


@pytest_asyncio.fixture
async def app(db):
    rag.set_embedder(rag.FakeEmbedder())
    fakes.reset()
    fakes.install()
    application = create_app(use_lifespan=False)
    yield application
    await engine.shutdown()
    rag.set_embedder(None)


@pytest_asyncio.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def fake():
    return fakes.FAKE
