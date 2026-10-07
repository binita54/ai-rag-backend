"""Tests for the sentence-transformer embedding service."""

import pytest

from app.services.embeddings import (
    EmbeddingError,
    SentenceTransformerEmbeddingService,
)


@pytest.fixture(scope="session")
def embedding_service() -> SentenceTransformerEmbeddingService:
    return SentenceTransformerEmbeddingService()


def test_model_loads(embedding_service: SentenceTransformerEmbeddingService) -> None:
    assert embedding_service.dimension > 0


def test_expected_vector_dimension(
    embedding_service: SentenceTransformerEmbeddingService,
) -> None:
    assert embedding_service.dimension == 384


async def test_single_text_embedding(
    embedding_service: SentenceTransformerEmbeddingService,
) -> None:
    vector = await embedding_service.embed_query("Hello world")
    assert len(vector) == 384
    assert all(isinstance(value, float) for value in vector)


async def test_batch_embedding(
    embedding_service: SentenceTransformerEmbeddingService,
) -> None:
    vectors = await embedding_service.embed_documents(["one", "two", "three"])
    assert len(vectors) == 3
    assert all(len(vector) == 384 for vector in vectors)


async def test_empty_batch_returns_empty_list(
    embedding_service: SentenceTransformerEmbeddingService,
) -> None:
    assert await embedding_service.embed_documents([]) == []


async def test_empty_query_rejected(
    embedding_service: SentenceTransformerEmbeddingService,
) -> None:
    with pytest.raises(EmbeddingError):
        await embedding_service.embed_query("   ")


async def test_output_is_deterministic(
    embedding_service: SentenceTransformerEmbeddingService,
) -> None:
    first = await embedding_service.embed_query("Deterministic text")
    second = await embedding_service.embed_query("Deterministic text")
    assert first == second


async def test_different_texts_produce_different_vectors(
    embedding_service: SentenceTransformerEmbeddingService,
) -> None:
    first = await embedding_service.embed_query("The cat sat on the mat")
    second = await embedding_service.embed_query("An overview of quantum physics")
    assert first != second
