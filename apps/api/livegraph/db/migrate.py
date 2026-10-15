"""Schema migrations at startup: Alembic ``upgrade head`` under a Postgres advisory lock.

A database created before Alembic (it has ``runs`` but no ``alembic_version``) is first stamped at the
baseline revision ``0001``, which matches that schema exactly, so no DDL runs against existing data. Every
schema change after that is its own revision in ``migrations/versions`` (``make db-rev m=...``).
Everything runs in one transaction on one connection (Postgres DDL is transactional), and the transaction
scoped advisory lock keeps concurrent API processes from migrating at the same time.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from livegraph.db import base as db_base

API_DIR = Path(__file__).resolve().parents[2]  # apps/api (alembic.ini + migrations/)
BASELINE = "0001"
LOCK_KEY = 0x4C47_4D49  # "LGMI": pg_advisory_xact_lock key for migrations


def alembic_config(connection: Connection | None = None) -> Config:
    cfg = Config(str(API_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(API_DIR / "migrations"))
    if connection is not None:
        cfg.attributes["connection"] = connection  # env.py runs on it (inside our transaction)
    return cfg


def _upgrade(conn: Connection, revision: str = "head") -> None:
    cfg = alembic_config(conn)
    tables = set(inspect(conn).get_table_names())
    if "alembic_version" not in tables and "runs" in tables:
        command.stamp(cfg, BASELINE)  # pre-Alembic database: its schema is the baseline
    command.upgrade(cfg, revision)


async def migrate(engine: AsyncEngine | None = None, revision: str = "head") -> None:
    async with (engine or db_base.engine).begin() as conn:
        await conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": LOCK_KEY})
        await conn.run_sync(_upgrade, revision)
