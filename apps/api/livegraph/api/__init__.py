from fastapi import APIRouter

from livegraph.api import documents, health, nodes, runs, ws

api_router = APIRouter(prefix="/api")
for _m in (runs, nodes, documents, ws, health):
    api_router.include_router(_m.router)

__all__ = ["api_router"]
