"""FastAPI dependency providers."""

from fastapi import Depends

from app.core.config import get_settings
from app.services.documents.ingestion import DocumentIngestionService
from app.services.embeddings import SentenceTransformerEmbeddingService
from app.services.vector_store import QdrantVectorStore

_embedding_service: SentenceTransformerEmbeddingService | None = None
_vector_store: QdrantVectorStore | None = None


def get_embedding_service() -> SentenceTransformerEmbeddingService:
    """Provide the shared, lazily initialized embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = SentenceTransformerEmbeddingService()
    return _embedding_service


def get_vector_store() -> QdrantVectorStore:
    """Provide the shared Qdrant vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore()
    return _vector_store


def get_document_ingestion_service(
    embedding_service: SentenceTransformerEmbeddingService = Depends(
        get_embedding_service
    ),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
) -> DocumentIngestionService:
    """Provide the document ingestion service."""
    settings = get_settings()
    return DocumentIngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        collection_name=settings.QDRANT_COLLECTION_NAME,
    )
