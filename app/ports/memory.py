"""Chat memory interface."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryPort(Protocol):
    """Contract for conversation memory backends (e.g. Redis)."""

    async def get_history(
        self, conversation_id: str, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Return stored messages for a conversation."""
        ...

    async def append_message(
        self, conversation_id: str, role: str, content: str
    ) -> None:
        """Append a message to the conversation history."""
        ...

    async def clear(self, conversation_id: str) -> None:
        """Clear conversation history."""
        ...

    async def close(self) -> None:
        """Close the underlying connection."""
        ...
