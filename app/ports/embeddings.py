"""Embedding service interface."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingPort(Protocol):
    """Contract for text embedding providers."""

    @property
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        ...
