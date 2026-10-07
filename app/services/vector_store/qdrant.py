"""Qdrant vector store with local and remote server modes."""

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from app.core.config import get_settings
from app.ports.vector_store import VectorSearchResult


class VectorStoreError(Exception):
    """Base exception for vector store failures."""


def chunk_point_id(document_id: int, chunk_index: int) -> str:
    """Build a deterministic Qdrant point ID for a document chunk."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}"))


class QdrantVectorStore:
    """Vector store backed by Qdrant local or remote mode."""

    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        settings = get_settings()
        if client is not None:
            self._client = client
        elif settings.QDRANT_MODE == "local":
            self._client = AsyncQdrantClient(path=settings.QDRANT_LOCAL_PATH)
        else:
            self._client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
            )

    async def close(self) -> None:
        """Release the underlying Qdrant client."""
        await self._client.close()

    async def ensure_collection(
        self, collection_name: str, vector_size: int
    ) -> None:
        """Create the collection if missing; reuse it otherwise."""
        try:
            if not await self._client.collection_exists(collection_name):
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=vector_size,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
                return

            info = await self._client.get_collection(collection_name)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                vectors = vectors.get("")
            if vectors is None or vectors.size != vector_size:
                raise VectorStoreError(
                    f"Collection '{collection_name}' already exists with an "
                    f"incompatible vector dimension"
                )
        except VectorStoreError:
            raise
        except Exception as error:
            raise VectorStoreError(
                f"Failed to initialize collection '{collection_name}': {error}"
            ) from error

    async def upsert(
        self, collection_name: str, points: list[dict[str, Any]]
    ) -> None:
        """Insert or update points with vectors and payloads."""
        if not points:
            raise VectorStoreError("Cannot upsert an empty list of points")

        for point in points:
            if "id" not in point or "vector" not in point:
                raise VectorStoreError(
                    "Each point requires 'id' and 'vector' keys"
                )
            if not point["vector"]:
                raise VectorStoreError("Point vectors must not be empty")

        try:
            structures = [
                qdrant_models.PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point.get("payload") or {},
                )
                for point in points
            ]
            await self._client.upsert(
                collection_name=collection_name, points=structures
            )
        except Exception as error:
            raise VectorStoreError(
                f"Failed to upsert points into '{collection_name}': {error}"
            ) from error

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Run a cosine similarity search with optional payload filters."""
        if not query_vector:
            raise VectorStoreError("Query vector must not be empty")
        if limit <= 0:
            raise VectorStoreError("limit must be a positive integer")

        try:
            query_filter = None
            if filters:
                conditions = [
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                    for key, value in filters.items()
                ]
                query_filter = qdrant_models.Filter(must=conditions)

            response = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
            return [
                VectorSearchResult(
                    id=point.id,
                    score=point.score,
                    payload=point.payload or {},
                )
                for point in response.points
            ]
        except Exception as error:
            raise VectorStoreError(
                f"Similarity search failed on '{collection_name}': {error}"
            ) from error

    async def delete_by_document(
        self, collection_name: str, document_id: int | str
    ) -> None:
        """Delete all vectors belonging to a document."""
        try:
            await self._client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=document_id),
                        )
                    ]
                ),
            )
        except Exception as error:
            raise VectorStoreError(
                f"Failed to delete document '{document_id}' from "
                f"'{collection_name}': {error}"
            ) from error
