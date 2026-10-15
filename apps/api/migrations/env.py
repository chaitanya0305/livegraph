"""Alembic environment (async). Target metadata = every LiveGraph table (core + RAG).

The URL is ``settings.database_url``. When a caller passes an open connection in
``config.attributes["connection"]`` (``livegraph.db.migrate`` at startup, the migrations test), migrations
run on it inside the caller's transaction; otherwise env.py opens its own async engine (CLI use).
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

import livegraph.db.models  # noqa: F401  (registers core tables)
import livegraph.rag.models  # noqa: F401  (registers documents, chunks, retrievals, web_cache)
from livegraph.config import settings
from livegraph.db.base import Base

config = context.config
target_metadata = Base.metadata

if config.config_file_name and "connection" not in config.attributes:
    fileConfig(config.config_file_name, disable_existing_loggers=False)  # CLI only: keep the app's logging


def _configure(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)


def run_migrations_offline() -> None:
    context.configure(url=settings.database_url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(settings.database_url, poolclass=pool.NullPool)
    async with engine.connect() as conn:
        await conn.run_sync(_run_sync)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
elif (conn := config.attributes.get("connection")) is not None:
    _run_sync(conn)
else:
    asyncio.run(run_migrations_online())
