"""LiveGraph RAG: documents, chunks, retrievals + ingest/search/web/embeddings.

Store functions only ``flush()``; the caller commits.
"""

from livegraph.rag.embeddings import Embedder, FakeEmbedder, OpenRouterEmbedder, get_embedder, set_embedder
from livegraph.rag.models import Chunk, Document, Retrieval
from livegraph.rag.schemas import DocumentOut, RetrievedChunk
from livegraph.rag.store import (
    document_out,
    get_document,
    get_retrievals,
    ingest_file,
    ingest_text,
    list_documents,
    record_retrieval,
    search,
    set_trust,
)
from livegraph.rag.web import set_web_client, web_research

__all__ = [
    "Chunk",
    "Document",
    "DocumentOut",
    "Embedder",
    "FakeEmbedder",
    "OpenRouterEmbedder",
    "Retrieval",
    "RetrievedChunk",
    "document_out",
    "get_document",
    "get_embedder",
    "get_retrievals",
    "ingest_file",
    "ingest_text",
    "list_documents",
    "record_retrieval",
    "search",
    "set_embedder",
    "set_trust",
    "set_web_client",
    "web_research",
]
