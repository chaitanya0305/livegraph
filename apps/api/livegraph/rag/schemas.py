"""Pydantic schemas exposed by the RAG module."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

Trust = Literal["normal", "distrusted", "pinned"]
Origin = Literal["upload", "web"]


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_title: str
    source_url: str | None = None
    text: str
    score: float
    trust: Trust


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: UUID | None = None
    title: str
    source_url: str | None = None
    origin: Origin
    trust: Trust
    created_at: datetime
    chunk_count: int = 0
