"""Vector store interface."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStorePort(Protocol):
    """Contract for vector database backends (e.g. Qdrant)."""

    async def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create the collection if it does not exist."""
        ...

    async def upsert(self, collection_name: str, points: list[dict[str, Any]]) -> None:
        """Insert or update points."""
        ...

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a nearest-neighbor search."""
        ...

    async def delete_by_document(self, collection_name: str, document_id: str) -> None:
        """Remove all points belonging to a document."""
        ...
