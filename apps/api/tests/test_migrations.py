"""Alembic: head matches the models, downgrade/upgrade round-trips, a pre-Alembic database is stamped.

Uses its own throwaway database (``livegraph_migr``), never the dev or test database.
"""

from __future__ import annotations

import asyncpg
import pytest_asyncio
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect, pool, text
from sqlalchemy.ext.asyncio import create_async_engine

from livegraph.db.base import Base
from livegraph.db.migrate import BASELINE, alembic_config, migrate
from tests.conftest import _ADMIN_DSN

MIGR_DB = "livegraph_migr"


async def _recreate() -> None:
    conn = await asyncpg.connect(_ADMIN_DSN)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{MIGR_DB}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{MIGR_DB}"')
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def migr_engine():
    await _recreate()
    eng = create_async_engine(
        f"postgresql+asyncpg://livegraph:livegraph@localhost:5433/{MIGR_DB}", poolclass=pool.NullPool
    )
    yield eng
    await eng.dispose()


def _diff(conn) -> list:
    return compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)


async def _version(eng) -> str | None:
    async with eng.connect() as conn:
        return await conn.scalar(text("SELECT version_num FROM alembic_version"))


async def test_upgrade_head_matches_models_and_round_trips(migr_engine):
    await migrate(migr_engine)
    async with migr_engine.connect() as conn:
        assert await conn.run_sync(_diff) == []
    assert await _version(migr_engine) == "0003"

    async with migr_engine.begin() as conn:
        await conn.run_sync(lambda c: command.downgrade(alembic_config(c), "base"))
    async with migr_engine.connect() as conn:
        tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
    assert tables <= {"alembic_version"}

    await migrate(migr_engine)
    async with migr_engine.connect() as conn:
        assert await conn.run_sync(_diff) == []
    await migrate(migr_engine)  # idempotent at head
    assert await _version(migr_engine) == "0003"


async def test_pre_alembic_database_is_stamped_then_upgraded(migr_engine):
    # a database as the old create_all + ALTER startup left it: the baseline schema, no alembic_version
    await migrate(migr_engine, BASELINE)
    async with migr_engine.begin() as conn:
        await conn.execute(text("INSERT INTO runs (goal, workflow, status) VALUES ('kept', 'living_brief', 'done')"))
        await conn.execute(text("DROP TABLE alembic_version"))

    await migrate(migr_engine)
    assert await _version(migr_engine) == "0003"
    async with migr_engine.connect() as conn:
        assert await conn.scalar(text("SELECT goal FROM runs")) == "kept"
        tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
        assert "context_items" in tables
        assert await conn.run_sync(_diff) == []
