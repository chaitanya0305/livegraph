"""FastAPI app factory + lifespan (pgvector extension, create_all, crash recovery)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

import livegraph.db.models  # noqa: F401  (registers core + rag tables on Base.metadata)
import livegraph.rag.models  # noqa: F401
from livegraph.api import api_router
from livegraph.config import settings
from livegraph.db import base as db_base
from livegraph.engine import EngineError, engine

log = logging.getLogger("livegraph")


async def init_db() -> None:
    async with db_base.engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(db_base.Base.metadata.create_all)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Simple restart recovery: runs that were planning/running when the process died become `paused`.
    await engine.recover()
    yield
    await engine.shutdown()


def create_app(*, use_lifespan: bool = True) -> FastAPI:
    app = FastAPI(title="LiveGraph API", lifespan=lifespan if use_lifespan else None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(EngineError)
    async def _engine_error(_: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    app.include_router(api_router)
    return app


app = create_app()
