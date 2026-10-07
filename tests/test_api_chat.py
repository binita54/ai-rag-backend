"""API tests for the conversational chat endpoint."""

import math

import httpx
import pytest
from httpx import ASGITransport
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.api.deps import (
    get_booking_service,
    get_embedding_service,
    get_llm_provider,
    get_memory_store,
    get_rag_service,
    get_vector_store,
)
from app.main import app
from app.models.booking import InterviewBooking
from app.models.database import Base, get_db
from app.ports.vector_store import VectorSearchResult
from app.services.booking import (
    BOOK_INTERVIEW_TOOL,
    BookingError,
    BookingService,
)
from app.services.llm import (
    LLMProviderError,
    LLMResponse,
    ToolCall,
)
from app.services.rag import (
    RAGAnswer,
    RAGError,
    RAGGenerationError,
    RAGRetrievalError,
    RetrievedSource,
)
from app.services.vector_store import QdrantVectorStore

SAMPLE_TEXT = (
    "Photosynthesis is the process by which plants convert light "
    "energy into chemical energy. Chlorophyll absorbs sunlight and "
    "uses it to synthesize sugars."
)


class FakeEmbeddingService:
    """Deterministic fake embeddings for fast tests."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def embed_documents(
        self, texts: list[str]
    ) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [float((index + 1) % 10) for index in range(self.dimension)]
        for position, character in enumerate(text[: self.dimension]):
            vector[position] = float(ord(character) % 11)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FakeVectorStore:
    """Fake vector store returning configured results."""

    def __init__(self, results: list[VectorSearchResult]) -> None:
        self._results = results

    async def ensure_collection(
        self, collection_name: str, vector_size: int
    ) -> None:
        pass

    async def upsert(
        self, collection_name: str, points: list[dict]
    ) -> None:
        pass

    async def delete_by_document(
        self, collection_name: str, document_id: int | str
    ) -> None:
        pass

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filters: dict | None = None,
    ) -> list[VectorSearchResult]:
        return self._results


class FakeMemoryStore:
    """Fake conversation memory store."""

    def __init__(self) -> None:
        self.appended: list[tuple[str, str, str]] = []

    async def get_history(
        self, conversation_id: str, limit: int | None = None
    ) -> list[dict[str, str]]:
        return []

    async def append_message(
        self, conversation_id: str, role: str, content: str
    ) -> None:
        self.appended.append((conversation_id, role, content))

    async def clear(self, conversation_id: str) -> None:
        pass

    async def close(self) -> None:
        pass


class FakeLLMProvider:
    """Fake LLM provider returning configured content."""

    def __init__(self, content: str = "the answer") -> None:
        self._content = content

    async def chat(
        self,
        messages: list,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        return LLMResponse(content=self._content)


BOOKING_ARGUMENTS = {
    "name": "Binita Ghale",
    "email": "binita@example.com",
    "date": "2026-10-10",
    "time": "14:00",
}


class BookingLLM:
    """Fake LLM that always requests the booking tool."""

    def __init__(self, arguments: dict) -> None:
        self.arguments = arguments
        self.tools_received: list[dict] | None | list[
            list[dict]
        ] = None

    async def chat(
        self,
        messages: list,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.tools_received = tools
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    name="book_interview",
                    arguments=self.arguments,
                )
            ],
        )


class FakeRAGService:
    """Controllable fake RAG service for route tests."""

    def __init__(
        self,
        answer: str = "the answer",
        sources: list[RetrievedSource] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.answer_text = answer
        self.sources = sources or []
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def answer(
        self, conversation_id: str, message: str
    ) -> RAGAnswer:
        self.calls.append((conversation_id, message))
        if self.error is not None:
            raise self.error
        return RAGAnswer(
            conversation_id=conversation_id,
            answer=self.answer_text,
            sources=self.sources,
        )


@pytest.fixture
def rag_service() -> FakeRAGService:
    return FakeRAGService()


@pytest.fixture
async def client(rag_service: FakeRAGService):
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as http_client:
        yield http_client
    app.dependency_overrides.clear()


@pytest.fixture
def db_engine(tmp_path):
    return create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/chat-test.db"
    )


@pytest.fixture
async def db_session_factory(db_engine):
    async with db_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )


@pytest.fixture
def embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
async def qdrant_client(tmp_path):
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    yield client
    await client.close()


@pytest.fixture
def vector_store(qdrant_client) -> QdrantVectorStore:
    return QdrantVectorStore(client=qdrant_client)


async def test_valid_chat_request_returns_200(
    client, rag_service
) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={
            "conversation_id": "demo-123",
            "message": "What is RAG?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "demo-123"
    assert body["answer"] == "the answer"
    assert body["sources"] == []
    assert rag_service.calls == [("demo-123", "What is RAG?")]
async def test_missing_conversation_id_rejected(client) -> None:
    response = await client.post(
        "/api/v1/chat", json={"message": "hi"}
    )
    assert response.status_code == 422


async def test_empty_message_rejected(client) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"conversation_id": "demo-123", "message": ""},
    )
    assert response.status_code == 422


async def test_oversized_message_rejected(client) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={
            "conversation_id": "demo-123",
            "message": "x" * 4001,
        },
    )
    assert response.status_code == 422


async def test_response_contains_sources(
    client, rag_service
) -> None:
    rag_service.sources = [
        RetrievedSource(filename="guide.pdf", chunk_index=2, score=0.81)
    ]
    response = await client.post(
        "/api/v1/chat",
        json={"conversation_id": "demo-123", "message": "hi"},
    )

    assert response.status_code == 200
    assert response.json()["sources"] == [
        {"filename": "guide.pdf", "chunk_index": 2, "score": 0.81}
    ]


async def test_retrieval_error_maps_to_502(
    client, rag_service
) -> None:
    rag_service.error = RAGRetrievalError("embedding failed")
    response = await client.post(
        "/api/v1/chat",
        json={"conversation_id": "demo-123", "message": "hi"},
    )
    assert response.status_code == 502


async def test_generation_error_maps_to_502(
    client, rag_service
) -> None:
    rag_service.error = RAGGenerationError("llm failed")
    response = await client.post(
        "/api/v1/chat",
        json={"conversation_id": "demo-123", "message": "hi"},
    )
    assert response.status_code == 502


async def test_unexpected_rag_error_maps_to_500(
    client, rag_service
) -> None:
    rag_service.error = RAGError("unexpected failure")
    response = await client.post(
        "/api/v1/chat",
        json={"conversation_id": "demo-123", "message": "hi"},
    )
    assert response.status_code == 500


async def test_chat_end_to_end_with_fake_dependencies(
    tmp_path,
) -> None:
    embeddings = FakeEmbeddingService()
    vector_store = FakeVectorStore(
        results=[
            VectorSearchResult(
                id="1-0",
                score=0.9,
                payload={
                    "document_id": 1,
                    "chunk_index": 0,
                    "filename": "guide.pdf",
                    "text": "RAG answers questions from documents.",
                },
            )
        ]
    )
    memory = FakeMemoryStore()
    llm = FakeLLMProvider(content="answer from documents")

    app.dependency_overrides[get_embedding_service] = (
        lambda: embeddings
    )
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_memory_store] = lambda: memory
    app.dependency_overrides[get_llm_provider] = lambda: llm
    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": "e2e-1",
                    "message": "What is RAG?",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "e2e-1"
    assert body["answer"] == "answer from documents"
    assert body["sources"] == [
        {"filename": "guide.pdf", "chunk_index": 0, "score": 0.9}
    ]
    assert memory.appended == [
        ("e2e-1", "user", "What is RAG?"),
        ("e2e-1", "assistant", "answer from documents"),
    ]


async def test_documents_endpoint_still_passes(
    db_session_factory, embedding_service, vector_store
) -> None:
    async def override_get_db():
        async with db_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedding_service] = (
        lambda: embedding_service
    )
    app.dependency_overrides[get_vector_store] = (
        lambda: vector_store
    )
    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/documents",
                files={
                    "file": (
                        "notes.txt",
                        SAMPLE_TEXT.encode(),
                        "application/octet-stream",
                    )
                },
                data={
                    "chunk_strategy": "recursive",
                    "chunk_size": "100",
                    "overlap": "20",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["filename"] == "notes.txt"


async def test_health_endpoint_still_passes(client) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_llm_provider_error_is_wrapped_as_generation_error(
    tmp_path,
) -> None:
    class FailingLLM:
        async def chat(self, messages, tools=None):
            raise LLMProviderError("provider exploded")

    embeddings = FakeEmbeddingService()
    vector_store = FakeVectorStore(results=[])
    memory = FakeMemoryStore()

    app.dependency_overrides[get_embedding_service] = (
        lambda: embeddings
    )
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_memory_store] = lambda: memory
    app.dependency_overrides[get_llm_provider] = lambda: FailingLLM()
    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": "e2e-2",
                    "message": "What is RAG?",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert memory.appended == []


def _booking_engine(tmp_path):
    return create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/booking-api-test.db"
    )


async def _create_booking_session_factory(engine):
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )


async def test_booking_through_chat_end_to_end(tmp_path) -> None:
    engine = _booking_engine(tmp_path)
    session_factory = (
        await _create_booking_session_factory(engine)
    )
    booking_service = BookingService(session_factory=session_factory)

    embeddings = FakeEmbeddingService()
    vector_store = FakeVectorStore(results=[])
    memory = FakeMemoryStore()
    llm = BookingLLM(BOOKING_ARGUMENTS)

    app.dependency_overrides[get_embedding_service] = (
        lambda: embeddings
    )
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_memory_store] = lambda: memory
    app.dependency_overrides[get_llm_provider] = lambda: llm
    app.dependency_overrides[get_booking_service] = (
        lambda: booking_service
    )
    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": "booking-1",
                    "message": (
                        "I'd like to book an interview. My name "
                        "is Binita Ghale, email is "
                        "binita@example.com, on October 10 at "
                        "2 PM."
                    ),
                },
            )
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "booking-1"
    assert body["answer"] == (
        "Your interview has been booked for 2026-10-10 at 14:00."
    )
    assert llm.tools_received == [
        BOOK_INTERVIEW_TOOL.to_spec()
    ]
    user_message = (
        "I'd like to book an interview. My name "
        "is Binita Ghale, email is "
        "binita@example.com, on October 10 at "
        "2 PM."
    )
    assert memory.appended == [
        ("booking-1", "user", user_message),
        ("booking-1", "assistant", body["answer"]),
    ]


async def test_missing_booking_information_returns_normal_response(
    tmp_path,
) -> None:
    engine = _booking_engine(tmp_path)
    session_factory = (
        await _create_booking_session_factory(engine)
    )
    booking_service = BookingService(session_factory=session_factory)

    embeddings = FakeEmbeddingService()
    vector_store = FakeVectorStore(results=[])
    memory = FakeMemoryStore()
    llm = BookingLLM({"name": "Binita Ghale"})

    app.dependency_overrides[get_embedding_service] = (
        lambda: embeddings
    )
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_memory_store] = lambda: memory
    app.dependency_overrides[get_llm_provider] = lambda: llm
    app.dependency_overrides[get_booking_service] = (
        lambda: booking_service
    )
    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": "booking-2",
                    "message": "Book an interview for me",
                },
            )
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()

    assert response.status_code == 200
    body = response.json()
    assert "email" in body["answer"]
    assert "date" in body["answer"]
    assert "time" in body["answer"]

    async with session_factory() as session:
        rows = list(
            (
                await session.execute(
                    select(InterviewBooking)
                )
            ).scalars().all()
        )
    assert rows == []


async def test_booking_database_failure_maps_to_500(tmp_path) -> None:
    class FailingBookingService:
        async def book(self, payload):
            raise BookingError("database unavailable")

    embeddings = FakeEmbeddingService()
    vector_store = FakeVectorStore(results=[])
    memory = FakeMemoryStore()
    llm = BookingLLM(BOOKING_ARGUMENTS)

    app.dependency_overrides[get_embedding_service] = (
        lambda: embeddings
    )
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_memory_store] = lambda: memory
    app.dependency_overrides[get_llm_provider] = lambda: llm
    app.dependency_overrides[get_booking_service] = (
        lambda: FailingBookingService()
    )
    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": "booking-3",
                    "message": "Book an interview for me",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert memory.appended == []
