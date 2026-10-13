from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from livegraph import rag
from livegraph.db import base as db_base
from livegraph.db.models import Run
from livegraph.engine import engine
from livegraph.events import bus
from livegraph.protocol import TrustBody
from livegraph.rag import DocumentOut

router = APIRouter(tags=["documents"])


async def _emit_doc(run_ids: set[UUID], doc: DocumentOut) -> None:
    async with db_base.SessionLocal() as s:
        existing = [rid for rid in run_ids if await s.get(Run, rid) is not None]
    for rid in existing:
        await bus.emit(rid, "document.updated", None, {"document": doc.model_dump(mode="json")})


@router.post("/documents", response_model=DocumentOut)
async def upload(file: UploadFile = File(...), run_id: UUID | None = Form(default=None)) -> DocumentOut:
    data = await file.read()
    async with db_base.SessionLocal() as s:
        try:
            doc = await rag.ingest_file(s, run_id=run_id, filename=file.filename or "upload.txt", data=data)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        await s.commit()
        out = await rag.document_out(s, doc)
    if run_id is not None:
        await _emit_doc({run_id}, out)
    return out


@router.get("/documents", response_model=list[DocumentOut])
async def list_docs(run_id: UUID | None = None) -> list[DocumentOut]:
    async with db_base.SessionLocal() as s:
        return await rag.list_documents(s, run_id)


@router.patch("/documents/{document_id}", response_model=DocumentOut)
async def set_trust(document_id: UUID, body: TrustBody) -> DocumentOut:
    async with db_base.SessionLocal() as s:
        try:
            doc = await rag.set_trust(s, document_id, body.trust)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except LookupError as e:
            raise HTTPException(404, str(e)) from e
        await s.commit()
        out = await rag.document_out(s, doc)
    affected = set(await engine.document_trust_changed(document_id))
    if out.run_id is not None:
        affected.add(out.run_id)
    await _emit_doc(affected, out)
    return out
