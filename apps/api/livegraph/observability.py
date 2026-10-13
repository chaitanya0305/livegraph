"""Logfire tracing: FastAPI requests, Pydantic AI agent runs, SQLAlchemy queries, outgoing HTTP.

Configured only from settings (``LOGFIRE_TOKEN`` in ``.env``); with no token nothing is sent.
"""

from __future__ import annotations

from fastapi import FastAPI

from livegraph.config import settings
from livegraph.db import base as db_base

_configured = False


def setup_logfire(app: FastAPI) -> None:
    global _configured
    if not settings.logfire_token:
        return
    import logfire

    if not _configured:
        logfire.configure(
            token=settings.logfire_token,
            service_name="livegraph-api",
            environment=settings.logfire_environment,
            send_to_logfire="if-token-present",
            console=False,
        )
        logfire.instrument_pydantic_ai()
        logfire.instrument_sqlalchemy(engine=db_base.engine.sync_engine)
        logfire.instrument_httpx(capture_all=False)
        _configured = True
    logfire.instrument_fastapi(app, excluded_urls="/api/health")
