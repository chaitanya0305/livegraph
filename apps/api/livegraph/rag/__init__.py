"""LiveGraph RAG: documents, chunks, retrievals + ingest/search/web/embeddings.

Store functions only ``flush()``; the caller commits.
"""

from livegraph.rag.embeddings import Embedder, FakeEmbedder, OpenRouterEmbedder, get_embedder, set_embedder
from livegraph.rag.models import Chunk, Document, Retrieval, WebCache
from livegraph.rag.schemas import DocumentOut, RetrievedChunk
from livegraph.rag.store import (
    corpus_fingerprint,
    document_out,
    get_document,
    get_retrievals,
    get_retrievals_many,
    ingest_file,
    ingest_text,
    list_documents,
    record_retrieval,
    search,
    set_trust,
)
from livegraph.rag.sources import get_chunk_window, resolve_chunk_id
from livegraph.rag.web import (
    fetch_citations,
    ingest_citations,
    set_exa_transport,
    set_web_client,
    web_provider,
    web_research,
)

__all__ = [
    "Chunk",
    "Document",
    "DocumentOut",
    "Embedder",
    "FakeEmbedder",
    "OpenRouterEmbedder",
    "Retrieval",
    "WebCache",
    "fetch_citations",
    "get_chunk_window",
    "ingest_citations",
    "resolve_chunk_id",
    "RetrievedChunk",
    "corpus_fingerprint",
    "document_out",
    "get_document",
    "get_embedder",
    "get_retrievals",
    "get_retrievals_many",
    "ingest_file",
    "ingest_text",
    "list_documents",
    "record_retrieval",
    "search",
    "set_embedder",
    "set_trust",
    "set_exa_transport",
    "set_web_client",
    "web_provider",
    "web_research",
]
