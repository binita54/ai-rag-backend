"""Tests for the Qdrant vector store using local mode."""

import pytest
from qdrant_client import AsyncQdrantClient

from app.services.embeddings import SentenceTransformerEmbeddingService
from app.services.vector_store import VectorStoreError, chunk_point_id
from app.services.vector_store.qdrant import QdrantVectorStore

CHUNKS = [
    {
        "id": chunk_point_id(1, 0),
        "payload": {
            "document_id": 1,
            "chunk_id": chunk_point_id(1, 0),
            "chunk_index": 0,
            "filename": "biology.pdf",
            "text": (
                "Photosynthesis is the process by which plants convert "
                "light energy into chemical energy."
            ),
            "file_type": "pdf",
            "chunk_strategy": "recursive",
        },
    },
    {
        "id": chunk_point_id(1, 1),
        "payload": {
            "document_id": 1,
            "chunk_id": chunk_point_id(1, 1),
            "chunk_index": 1,
            "filename": "biology.pdf",
            "text": (
                "Chlorophyll absorbs sunlight and uses it to synthesize "
                "sugars from carbon dioxide and water."
            ),
            "file_type": "pdf",
            "chunk_strategy": "recursive",
        },
    },
    {
        "id": chunk_point_id(2, 0),
        "payload": {
            "document_id": 2,
            "chunk_id": chunk_point_id(2, 0),
            "chunk_index": 0,
            "filename": "cells.txt",
            "text": (
                "The mitochondria is the powerhouse of the cell and "
                "produces ATP through respiration."
            ),
            "file_type": "txt",
            "chunk_strategy": "sentence",
        },
    },
]


@pytest.fixture(scope="session")
def embedding_service() -> SentenceTransformerEmbeddingService:
    return SentenceTransformerEmbeddingService()


@pytest.fixture
async def vector_store(tmp_path):
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantVectorStore(client=client)
    yield store
    await client.close()


async def _seed(vector_store: QdrantVectorStore, embedding_service) -> None:
    await vector_store.ensure_collection("docs", embedding_service.dimension)
    points = []
    for chunk in CHUNKS:
        points.append(
            {
                "id": chunk["id"],
                "vector": await embedding_service.embed_query(
                    chunk["payload"]["text"]
                ),
                "payload": chunk["payload"],
            }
        )
    await vector_store.upsert("docs", points)


async def test_collection_creation(
    vector_store: QdrantVectorStore, embedding_service
) -> None:
    await vector_store.ensure_collection("docs", embedding_service.dimension)
    assert await vector_store._client.collection_exists("docs")


async def test_initialization_is_idempotent(
    vector_store: QdrantVectorStore, embedding_service
) -> None:
    await vector_store.ensure_collection("docs", embedding_service.dimension)
    await vector_store.ensure_collection("docs", embedding_service.dimension)


async def test_dimension_mismatch_rejected(
    vector_store: QdrantVectorStore, embedding_service
) -> None:
    await vector_store.ensure_collection("docs", embedding_service.dimension)
    with pytest.raises(VectorStoreError):
        await vector_store.ensure_collection("docs", 768)


async def test_upsert_and_similarity_search(
    vector_store: QdrantVectorStore, embedding_service
) -> None:
    await _seed(vector_store, embedding_service)

    query_vector = await embedding_service.embed_query(
        "How do plants convert light into energy?"
    )
    results = await vector_store.search("docs", query_vector, limit=5)

    assert len(results) == 3
    top = results[0]
    assert top.payload["document_id"] == 1
    assert top.payload["chunk_index"] == 0
    assert top.payload["filename"] == "biology.pdf"
    assert top.payload["file_type"] == "pdf"
    assert top.payload["chunk_strategy"] == "recursive"
    assert "photosynthesis" in top.payload["text"].lower()
    assert top.score > 0.5
    assert isinstance(top.id, str)


async def test_search_orders_by_relevance(
    vector_store: QdrantVectorStore, embedding_service
) -> None:
    await _seed(vector_store, embedding_service)

    query_vector = await embedding_service.embed_query(
        "How do plants convert light into energy?"
    )
    results = await vector_store.search("docs", query_vector, limit=3)

    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


async def test_search_filters_by_document_id(
    vector_store: QdrantVectorStore, embedding_service
) -> None:
    await _seed(vector_store, embedding_service)

    query_vector = await embedding_service.embed_query("energy")
    results = await vector_store.search(
        "docs", query_vector, limit=10, filters={"document_id": 2}
    )

    assert results
    assert all(result.payload["document_id"] == 2 for result in results)
    assert all(result.payload["filename"] == "cells.txt" for result in results)


async def test_delete_by_document(
    vector_store: QdrantVectorStore, embedding_service
) -> None:
    await _seed(vector_store, embedding_service)

    await vector_store.delete_by_document("docs", 1)
    results = await vector_store.search(
        "docs",
        await embedding_service.embed_query("photosynthesis light energy"),
        limit=10,
    )

    assert len(results) == 1
    assert results[0].payload["document_id"] == 2


async def test_upsert_rejects_invalid_vector_dimension(
    vector_store: QdrantVectorStore, embedding_service
) -> None:
    await vector_store.ensure_collection("docs", embedding_service.dimension)
    with pytest.raises(VectorStoreError):
        await vector_store.upsert(
            "docs",
            [{"id": "bad", "vector": [0.1, 0.2, 0.3], "payload": {}}],
        )


async def test_search_on_missing_collection_rejected(
    vector_store: QdrantVectorStore, embedding_service
) -> None:
    query_vector = await embedding_service.embed_query("anything")
    with pytest.raises(VectorStoreError):
        await vector_store.search("missing", query_vector)


async def test_empty_query_vector_rejected(
    vector_store: QdrantVectorStore,
) -> None:
    with pytest.raises(VectorStoreError):
        await vector_store.search("docs", [], limit=5)


async def test_invalid_limit_rejected(
    vector_store: QdrantVectorStore,
) -> None:
    with pytest.raises(VectorStoreError):
        await vector_store.search("docs", [0.1], limit=0)


async def test_empty_upsert_rejected(vector_store: QdrantVectorStore) -> None:
    with pytest.raises(VectorStoreError):
        await vector_store.upsert("docs", [])


async def test_point_missing_required_keys_rejected(
    vector_store: QdrantVectorStore, embedding_service
) -> None:
    await vector_store.ensure_collection("docs", embedding_service.dimension)
    with pytest.raises(VectorStoreError):
        await vector_store.upsert("docs", [{"id": "incomplete"}])


async def test_chunk_point_id_is_deterministic() -> None:
    assert chunk_point_id(1, 0) == chunk_point_id(1, 0)
    assert chunk_point_id(1, 0) != chunk_point_id(1, 1)
