"""Vector store interface."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class VectorSearchResult:
    """A single similarity search hit with its payload metadata."""

    id: int | str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


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
    ) -> list[VectorSearchResult]:
        """Run a nearest-neighbor search."""
        ...

    async def delete_by_document(self, collection_name: str, document_id: int | str) -> None:
        """Remove all points belonging to a document."""
        ...
