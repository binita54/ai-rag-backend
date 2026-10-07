"""API tests for document ingestion."""

import math

import httpx
import pymupdf
import pytest
from httpx import ASGITransport
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import (
    get_embedding_service,
    get_vector_store,
)
from app.main import app
from app.models.database import Base, get_db
from app.models.document import Document, DocumentChunk
from app.services.embeddings import EmbeddingError, SentenceTransformerEmbeddingService
from app.services.vector_store import QdrantVectorStore, VectorStoreError

COLLECTION_NAME = "documents"

SAMPLE_TEXT = (
    "Photosynthesis is the process by which plants convert light "
    "energy into chemical energy. Chlorophyll absorbs sunlight and "
    "uses it to synthesize sugars. The mitochondria is the "
    "powerhouse of the cell and produces ATP."
)


class FakeEmbeddingService:
    """Deterministic fake embeddings for fast tests."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [float((index + 1) % 10) for index in range(self.dimension)]
        for position, character in enumerate(text[: self.dimension]):
            vector[position] = float(ord(character) % 11)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FailingEmbeddingService(FakeEmbeddingService):
    """Fake embeddings that always fail."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("simulated embedding failure")


class PartialFailureVectorStore(QdrantVectorStore):
    """Real Qdrant store that fails after a successful upsert."""

    async def upsert(self, collection_name: str, points: list[dict]) -> None:
        await super().upsert(collection_name, points)
        raise VectorStoreError("simulated failure after partial upsert")


def _make_pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    return document.tobytes()


@pytest.fixture
def db_engine(tmp_path):
    return create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")


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


@pytest.fixture
async def api_client(db_session_factory, embedding_service, vector_store):
    async def override_get_db():
        async with db_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedding_service] = lambda: embedding_service
    app.dependency_overrides[get_vector_store] = lambda: vector_store

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client

    app.dependency_overrides.clear()


async def _upload(
    client: httpx.AsyncClient,
    filename: str,
    content: bytes,
    strategy: str = "recursive",
) -> httpx.Response:
    return await client.post(
        "/api/v1/documents",
        files={"file": (filename, content, "application/octet-stream")},
        data={
            "chunk_strategy": strategy,
            "chunk_size": "100",
            "overlap": "20",
        },
    )


async def _all_documents(db_session_factory) -> list[Document]:
    async with db_session_factory() as session:
        result = await session.execute(select(Document))
        return list(result.scalars().all())


async def _all_chunks(db_session_factory) -> list[DocumentChunk]:
    async with db_session_factory() as session:
        result = await session.execute(select(DocumentChunk))
        return list(result.scalars().all())


async def test_ingest_txt_recursive(
    api_client, db_session_factory, embedding_service, vector_store
) -> None:
    response = await _upload(api_client, "notes.txt", SAMPLE_TEXT.encode(), "recursive")

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["file_type"] == "txt"
    assert body["chunk_strategy"] == "recursive"
    assert body["chunk_count"] > 1

    documents = await _all_documents(db_session_factory)
    assert len(documents) == 1
    assert documents[0].chunk_count == body["chunk_count"]

    chunks = await _all_chunks(db_session_factory)
    assert len(chunks) == body["chunk_count"]
    assert all(chunk.document_id == documents[0].id for chunk in chunks)

    results = await vector_store.search(
        COLLECTION_NAME,
        await embedding_service.embed_query(chunks[0].text),
        limit=10,
    )
    assert len(results) == len(chunks)
    top = results[0]
    assert top.payload["document_id"] == documents[0].id
    assert top.payload["filename"] == "notes.txt"
    assert top.payload["file_type"] == "txt"
    assert top.payload["chunk_strategy"] == "recursive"
    assert top.payload["text"] == chunks[top.payload["chunk_index"]].text


async def test_ingest_txt_sentence(
    api_client, db_session_factory, embedding_service, vector_store
) -> None:
    response = await _upload(api_client, "notes.txt", SAMPLE_TEXT.encode(), "sentence")

    assert response.status_code == 201
    assert response.json()["chunk_strategy"] == "sentence"

    documents = await _all_documents(db_session_factory)
    assert len(documents) == 1
    assert documents[0].chunk_strategy == "sentence"

    chunks = await _all_chunks(db_session_factory)
    assert len(chunks) == documents[0].chunk_count
    assert len(await vector_store.search(
        COLLECTION_NAME,
        await embedding_service.embed_query("Photosynthesis"),
        limit=10,
    )) == len(chunks)


async def test_ingest_pdf_recursive(
    api_client, db_session_factory, embedding_service, vector_store
) -> None:
    pdf_bytes = _make_pdf("PDF content about machine learning and data science.")
    response = await _upload(api_client, "report.pdf", pdf_bytes, "recursive")

    assert response.status_code == 201
    body = response.json()
    assert body["file_type"] == "pdf"
    assert body["chunk_count"] >= 1

    documents = await _all_documents(db_session_factory)
    assert len(documents) == 1
    assert documents[0].file_type == "pdf"

    chunks = await _all_chunks(db_session_factory)
    assert len(chunks) == body["chunk_count"]
    results = await vector_store.search(
        COLLECTION_NAME,
        await embedding_service.embed_query(chunks[0].text),
        limit=10,
    )
    assert len(results) == len(chunks)
    assert all(result.payload["file_type"] == "pdf" for result in results)


async def test_unsupported_extension_rejected(api_client) -> None:
    response = await _upload(api_client, "slides.docx", b"word document bytes")
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


async def test_empty_file_rejected(api_client) -> None:
    response = await _upload(api_client, "empty.txt", b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


async def test_invalid_pdf_rejected(api_client) -> None:
    response = await _upload(api_client, "broken.pdf", b"not a real pdf")
    assert response.status_code == 400


async def test_invalid_chunk_strategy_rejected(api_client) -> None:
    response = await api_client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", SAMPLE_TEXT.encode())},
        data={"chunk_strategy": "semantic"},
    )
    assert response.status_code == 422


async def test_missing_chunk_strategy_rejected(api_client) -> None:
    response = await api_client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", SAMPLE_TEXT.encode())},
    )
    assert response.status_code == 422


async def test_extraction_failure_rolls_back(
    api_client, db_session_factory, qdrant_client
) -> None:
    response = await _upload(api_client, "broken.pdf", b"garbage bytes")
    assert response.status_code == 400

    assert await _all_documents(db_session_factory) == []
    assert await _all_chunks(db_session_factory) == []
    assert not await qdrant_client.collection_exists(COLLECTION_NAME)


async def test_embedding_failure_rolls_back(
    api_client,
    db_session_factory,
    vector_store,
    embedding_service,
    monkeypatch,
) -> None:
    failing = FailingEmbeddingService()
    app.dependency_overrides[get_embedding_service] = lambda: failing
    try:
        response = await _upload(api_client, "notes.txt", SAMPLE_TEXT.encode())
    finally:
        app.dependency_overrides[get_embedding_service] = lambda: embedding_service

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Document vector storage service unavailable"
    )
    assert "simulated embedding failure" not in response.text
    assert await _all_documents(db_session_factory) == []
    assert await _all_chunks(db_session_factory) == []
    assert await vector_store.search(
        COLLECTION_NAME, [0.1] * 8, limit=10
    ) == []


async def test_vector_store_failure_cleans_up(
    api_client,
    db_session_factory,
    qdrant_client,
    embedding_service,
) -> None:
    failing_store = PartialFailureVectorStore(client=qdrant_client)
    app.dependency_overrides[get_vector_store] = lambda: failing_store
    try:
        response = await _upload(api_client, "notes.txt", SAMPLE_TEXT.encode())
    finally:
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Document vector storage service unavailable"
    )
    assert "simulated failure after partial upsert" not in response.text

    assert await _all_documents(db_session_factory) == []
    assert await _all_chunks(db_session_factory) == []

    clean_store = QdrantVectorStore(client=qdrant_client)
    results = await clean_store.search(
        COLLECTION_NAME,
        await embedding_service.embed_query(SAMPLE_TEXT),
        limit=10,
    )
    assert results == []


async def test_oversized_file_rejected(api_client, monkeypatch) -> None:
    from app.api.v1 import documents as documents_module
    from app.core.config import Settings

    monkeypatch.setattr(
        documents_module, "get_settings", lambda: Settings(MAX_UPLOAD_SIZE=8)
    )
    response = await _upload(api_client, "big.txt", b"x" * 100)
    assert response.status_code == 413


async def test_end_to_end_real_embeddings(tmp_path, db_session_factory) -> None:
    embeddings = SentenceTransformerEmbeddingService()
    qdrant_client = AsyncQdrantClient(path=str(tmp_path / "qdrant-e2e"))
    vector_store = QdrantVectorStore(client=qdrant_client)

    async def override_get_db():
        async with db_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedding_service] = lambda: embeddings
    app.dependency_overrides[get_vector_store] = lambda: vector_store

    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await _upload(
                client, "notes.txt", SAMPLE_TEXT.encode(), "recursive"
            )
            assert response.status_code == 201
            document_id = response.json()["id"]

            results = await vector_store.search(
                COLLECTION_NAME,
                await embeddings.embed_query(
                    "How do plants convert light into energy?"
                ),
                limit=5,
            )
            assert results
            assert all(
                result.payload["document_id"] == document_id
                for result in results
            )
            assert results[0].payload["filename"] == "notes.txt"
            assert results[0].payload["file_type"] == "txt"
            assert results[0].payload["chunk_strategy"] == "recursive"
    finally:
        app.dependency_overrides.clear()
        await qdrant_client.close()
