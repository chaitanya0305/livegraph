"""FastAPI app factory + lifespan (Alembic migrations, crash recovery)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from livegraph.api import api_router
from livegraph.config import settings
from livegraph.db.migrate import migrate
from livegraph.engine import EngineError, engine
from livegraph.observability import setup_logfire

log = logging.getLogger("livegraph")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await migrate()  # alembic upgrade head (stamps a pre-Alembic database at the baseline first)
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
    setup_logfire(app)
    return app


app = create_app()
