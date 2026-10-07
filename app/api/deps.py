"""FastAPI dependency providers."""

from fastapi import Depends

from app.core.config import get_settings
from app.models.database import AsyncSessionLocal
from app.services.booking import BookingService
from app.services.booking.tools import BOOK_INTERVIEW_TOOL
from app.services.documents.ingestion import DocumentIngestionService
from app.services.embeddings import SentenceTransformerEmbeddingService
from app.services.llm import OpenAICompatibleProvider
from app.services.memory import RedisMemoryStore
from app.services.rag import RAGService
from app.services.vector_store import QdrantVectorStore

_embedding_service: SentenceTransformerEmbeddingService | None = None
_vector_store: QdrantVectorStore | None = None
_memory_store: RedisMemoryStore | None = None
_llm_provider: OpenAICompatibleProvider | None = None
_booking_service: BookingService | None = None


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


def get_memory_store() -> RedisMemoryStore:
    """Provide the shared Redis conversation memory store."""
    global _memory_store
    if _memory_store is None:
        _memory_store = RedisMemoryStore()
    return _memory_store


def get_llm_provider() -> OpenAICompatibleProvider:
    """Provide the shared LLM provider."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = OpenAICompatibleProvider()
    return _llm_provider


def get_booking_service() -> BookingService:
    """Provide the shared interview booking service."""
    global _booking_service
    if _booking_service is None:
        _booking_service = BookingService(
            session_factory=AsyncSessionLocal
        )
    return _booking_service


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


def get_rag_service(
    embedding_service: SentenceTransformerEmbeddingService = Depends(
        get_embedding_service
    ),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    memory_store: RedisMemoryStore = Depends(get_memory_store),
    llm_provider: OpenAICompatibleProvider = Depends(get_llm_provider),
    booking_service: BookingService = Depends(get_booking_service),
) -> RAGService:
    """Provide the RAG orchestration service."""
    settings = get_settings()
    return RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        memory_store=memory_store,
        llm_provider=llm_provider,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        top_k=settings.RAG_TOP_K,
        score_threshold=settings.RAG_SCORE_THRESHOLD,
        max_context_chars=settings.RAG_MAX_CONTEXT_CHARS,
        booking_service=booking_service,
        tools=[BOOK_INTERVIEW_TOOL],
    )
