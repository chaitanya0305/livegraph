"""Tests for livegraph.rag (offline: FakeEmbedder, mocked OpenRouter client, separate test DB)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import Column, Table, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from livegraph.config import settings
from livegraph.db.base import Base

try:  # prefer the real backend models so create_all covers node_versions properly
    import livegraph.db.models  # noqa: F401
except Exception:  # pragma: no cover - backend models not available yet
    pass
if "node_versions" not in Base.metadata.tables:
    Table("node_versions", Base.metadata, Column("id", PGUUID(as_uuid=True), primary_key=True))

from livegraph import rag  # noqa: E402
from livegraph.rag import web as rag_web  # noqa: E402
from livegraph.rag.chunking import chunk_text  # noqa: E402
from livegraph.rag.embeddings import FakeEmbedder, OpenRouterEmbedder  # noqa: E402

TEST_DB = "livegraph_test"


@pytest_asyncio.fixture(scope="session")
async def rag_engine():
    base_url = make_url(settings.database_url)
    admin = create_async_engine(base_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        exists = await conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": TEST_DB})
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{TEST_DB}"'))
    await admin.dispose()

    eng = create_async_engine(base_url.set(database=TEST_DB))
    async with eng.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(rag_engine):
    rag.set_embedder(FakeEmbedder())
    async with rag_engine.begin() as conn:
        await conn.execute(text("TRUNCATE retrievals, chunks, documents CASCADE"))
    maker = async_sessionmaker(rag_engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as s:
        yield s
    rag.set_embedder(None)
    rag.set_web_client(None)


LONG_TEXT = "\n\n".join(
    f"Paragraph {i}. " + " ".join(f"Sentence {i}.{j} talks about topic{i} and more words here." for j in range(6))
    for i in range(8)
)


# ---------------------------------------------------------------- chunking / embeddings


def test_chunker_sizes_and_overlap():
    chunks = chunk_text(LONG_TEXT)
    assert len(chunks) > 3
    assert all(len(c) <= 800 for c in chunks)
    # overlap: each chunk after the first starts with text from the end of the previous chunk
    for a, b in zip(chunks, chunks[1:]):
        assert b.split()[0] in a
    assert chunk_text("") == []
    assert chunk_text("short") == ["short"]
    # a single huge unbroken paragraph still splits
    assert all(len(c) <= 800 for c in chunk_text("word " * 1000))


async def test_fake_embedder_similarity():
    e = FakeEmbedder()
    a, b, c = await e.embed(["solar panels efficiency", "efficiency of solar panels", "medieval castle history"])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))  # noqa: E731
    assert len(a) == settings.embed_dim
    assert abs(dot(a, a) - 1.0) < 1e-9
    assert dot(a, b) > dot(a, c)
    assert (await e.embed(["solar panels efficiency"]))[0] == a


async def test_openrouter_embedder_batches():
    calls = []

    async def create(model, input):
        calls.append(len(input))
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=[float(i)] * 3) for i in reversed(range(len(input)))]
        )

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    out = await OpenRouterEmbedder(client=client, dim=3).embed([f"t{i}" for i in range(130)])
    assert calls == [64, 64, 2]
    assert len(out) == 130 and out[1] == [1.0, 1.0, 1.0]


def test_get_embedder_default_is_fake_without_key(monkeypatch):
    from livegraph.rag import embeddings

    monkeypatch.setattr(embeddings, "_default", None)
    monkeypatch.setattr(embeddings.settings, "openrouter_api_key", "")
    rag.set_embedder(None)
    assert isinstance(rag.get_embedder(), FakeEmbedder)


# ---------------------------------------------------------------- store


async def test_ingest_and_chunk(session):
    run_id = uuid.uuid4()
    doc = await rag.ingest_text(session, run_id=run_id, title="Long doc", text=LONG_TEXT)
    await session.commit()
    assert doc.origin == "upload" and doc.trust == "normal"
    docs = await rag.list_documents(session, run_id)
    assert len(docs) == 1 and docs[0].chunk_count == len(chunk_text(LONG_TEXT))
    assert (await rag.get_document(session, doc.id)).title == "Long doc"
    out = await rag.document_out(session, doc)
    assert out.chunk_count == docs[0].chunk_count and out.created_at is not None


async def test_search_ranking_and_run_scoping(session):
    run_a, run_b = uuid.uuid4(), uuid.uuid4()
    solar = await rag.ingest_text(
        session, run_id=run_a, title="Solar", text="Solar panels convert sunlight into electricity efficiently."
    )
    await rag.ingest_text(session, run_id=run_a, title="Castles", text="Medieval castles had thick stone walls.")
    glob = await rag.ingest_text(
        session, run_id=None, title="Global wind", text="Wind turbines convert wind into electricity."
    )
    other = await rag.ingest_text(
        session, run_id=run_b, title="Other run solar", text="Solar panels convert sunlight into electricity."
    )
    await session.commit()

    res = await rag.search(session, run_id=run_a, query="how do solar panels make electricity", k=6)
    doc_ids = [r.document_id for r in res]
    assert doc_ids[0] == solar.id
    assert glob.id in doc_ids  # global library included
    assert other.id not in doc_ids  # other run excluded
    assert res == sorted(res, key=lambda r: r.score, reverse=True)
    assert all(-1.0 <= r.score <= 1.0 + 1e-6 for r in res)
    assert res[0].document_title == "Solar" and res[0].trust == "normal"

    assert len(await rag.search(session, run_id=run_a, query="electricity", k=1)) == 1
    listed = {d.id for d in await rag.list_documents(session, run_a)}
    assert glob.id in listed and solar.id in listed and other.id not in listed and len(listed) == 3


async def test_distrust_excludes(session):
    run_id = uuid.uuid4()
    doc = await rag.ingest_text(session, run_id=run_id, title="Bad", text="Vaccines cause magnetism in humans.")
    await session.commit()
    assert any(r.document_id == doc.id for r in await rag.search(session, run_id=run_id, query="vaccines magnetism"))
    updated = await rag.set_trust(session, doc.id, "distrusted")
    await session.commit()
    assert updated.trust == "distrusted"
    assert await rag.search(session, run_id=run_id, query="vaccines magnetism") == []
    with pytest.raises(ValueError):
        await rag.set_trust(session, doc.id, "bogus")
    with pytest.raises(LookupError):
        await rag.set_trust(session, uuid.uuid4(), "pinned")


async def test_pin_boost_and_inclusion(session):
    run_id = uuid.uuid4()
    filler = [
        await rag.ingest_text(
            session, run_id=run_id, title=f"F{i}", text=f"Battery storage capacity grid number {i} lithium cells."
        )
        for i in range(8)
    ]
    weak = await rag.ingest_text(
        session, run_id=run_id, title="Weak", text="Battery storage costs fell while cobalt mining expanded abroad."
    )
    await session.commit()
    q = "battery storage capacity grid lithium cells"
    before = await rag.search(session, run_id=run_id, query=q, k=3)
    assert weak.id not in [r.document_id for r in before]

    await rag.set_trust(session, weak.id, "pinned")
    await session.commit()
    after = await rag.search(session, run_id=run_id, query=q, k=3)
    assert len(after) == 3
    pinned = [r for r in after if r.document_id == weak.id]
    assert pinned and pinned[0].trust == "pinned"
    # boost applied: pinned score = raw + 0.15 (raw is cosine sim, compare to unboosted search)
    await rag.set_trust(session, weak.id, "normal")
    await session.commit()
    raw = next(r for r in await rag.search(session, run_id=run_id, query=q, k=20) if r.document_id == weak.id)
    assert abs(pinned[0].score - (raw.score + 0.15)) < 1e-6
    assert filler  # silence

    # irrelevant pinned doc is NOT forced in
    junk = await rag.ingest_text(session, run_id=run_id, title="Junk", text="Zebra quokka xylophone.")
    await rag.set_trust(session, junk.id, "pinned")
    await session.commit()
    assert junk.id not in [r.document_id for r in await rag.search(session, run_id=run_id, query=q, k=3)]


async def test_record_retrieval(session):
    run_id = uuid.uuid4()
    await rag.ingest_text(session, run_id=run_id, title="A", text="Alpha beta gamma delta.")
    await rag.ingest_text(session, run_id=run_id, title="B", text="Alpha beta epsilon.")
    await session.commit()
    chunks = await rag.search(session, run_id=run_id, query="alpha beta")
    nv_id = uuid.uuid4()
    # node_versions belongs to the backend; bypass its FK so this test doesn't depend on its schema.
    await session.execute(text("SET session_replication_role = replica"))
    await rag.record_retrieval(session, node_version_id=nv_id, chunks=chunks)
    await session.commit()
    await session.execute(text("SET session_replication_role = DEFAULT"))
    got = await rag.get_retrievals(session, nv_id)
    assert [g["rank"] for g in got] == list(range(1, len(chunks) + 1))
    assert [g["chunk_id"] for g in got] == [str(c.chunk_id) for c in chunks]
    assert got[0]["score"] == pytest.approx(chunks[0].score)


def _make_pdf(text_line: str) -> bytes:
    content = f"BT /F1 12 Tf 72 720 Td ({text_line}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


async def test_ingest_file_pdf_txt_md(session):
    run_id = uuid.uuid4()
    pdf = await rag.ingest_file(
        session, run_id=run_id, filename="report.pdf", data=_make_pdf("Geothermal energy is underrated.")
    )
    md = await rag.ingest_file(session, run_id=run_id, filename="notes.md", data=b"# Notes\n\nTidal power works.")
    txt = await rag.ingest_file(session, run_id=None, filename="a.txt", data="café au lait".encode())
    await session.commit()
    assert pdf.title == "report.pdf" and pdf.origin == "upload"
    res = await rag.search(session, run_id=run_id, query="geothermal energy")
    assert res[0].document_id == pdf.id and "Geothermal" in res[0].text
    assert md.id and txt.run_id is None
    with pytest.raises(ValueError):
        await rag.ingest_file(session, run_id=run_id, filename="x.docx", data=b"zz")
    with pytest.raises(ValueError):
        await rag.ingest_file(session, run_id=run_id, filename="empty.txt", data=b"   ")


# ---------------------------------------------------------------- web research


async def test_web_research_noop_without_key(session, monkeypatch):
    monkeypatch.setattr(rag_web.settings, "openrouter_api_key", "")
    rag.set_web_client(None)
    assert await rag.web_research(session, run_id=uuid.uuid4(), query="anything") == []


async def test_web_research_noop_when_disabled(session, monkeypatch):
    monkeypatch.setattr(rag_web.settings, "web_search", False)
    rag.set_web_client(SimpleNamespace())  # would explode if used
    assert await rag.web_research(session, run_id=uuid.uuid4(), query="anything") == []


def _completion(content: str, annotations: list[dict]):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "gen-1",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek/deepseek-v4.1-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content, "annotations": annotations},
                }
            ],
        }
    )


class _FakeClient:
    def __init__(self, resp=None, exc=None):
        self.calls: list[dict] = []
        self._resp, self._exc = resp, exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        return self._resp


async def test_web_research_parses_citations(session, monkeypatch):
    monkeypatch.setattr(rag_web.settings, "web_search", True)
    body = "Perovskite cells reached 34% efficiency [nrel.gov](https://nrel.gov/pv).\n\nOther note [x](https://x.com/a)."
    s2 = body.index("Other")
    resp = _completion(
        body,
        [
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://nrel.gov/pv",
                    "title": "NREL Efficiency Chart",
                    "content": "Perovskite-silicon tandem cells hit 34 percent efficiency [...] record",
                    "start_index": 0,
                    "end_index": 40,
                },
            },
            {  # duplicate URL -> deduped
                "type": "url_citation",
                "url_citation": {"url": "https://nrel.gov/pv", "title": "dup", "start_index": 1, "end_index": 5},
            },
            {  # no content -> falls back to surrounding paragraph of the answer
                "type": "url_citation",
                "url_citation": {"url": "https://x.com/a", "title": "X post", "start_index": s2, "end_index": s2 + 5},
            },
        ],
    )
    client = _FakeClient(resp)
    rag.set_web_client(client)
    run_id = uuid.uuid4()
    docs = await rag.web_research(session, run_id=run_id, query="perovskite efficiency", max_results=4)
    await session.commit()

    call = client.calls[0]
    assert call["model"] == settings.model_fast
    assert call["extra_body"] == {"plugins": [{"id": "web", "max_results": 4}]}
    assert [d.source_url for d in docs] == ["https://nrel.gov/pv", "https://x.com/a"]
    assert all(d.origin == "web" and d.run_id == run_id for d in docs)
    assert docs[0].title == "NREL Efficiency Chart"

    res = await rag.search(session, run_id=run_id, query="perovskite tandem efficiency")
    assert res[0].document_id == docs[0].id and "34 percent" in res[0].text
    x_hit = next(r for r in await rag.search(session, run_id=run_id, query="other note", k=10) if r.document_id == docs[1].id)
    assert "Other note" in x_hit.text

    # Second call in same run: deduped by source_url, no new documents.
    again = await rag.web_research(session, run_id=run_id, query="perovskite efficiency")
    await session.commit()
    assert [d.id for d in again] == [d.id for d in docs]
    assert len(await rag.list_documents(session, run_id)) == 2


async def test_web_research_swallows_errors(session, monkeypatch):
    monkeypatch.setattr(rag_web.settings, "web_search", True)
    rag.set_web_client(_FakeClient(exc=ConnectionError("boom")))
    assert await rag.web_research(session, run_id=uuid.uuid4(), query="q") == []
