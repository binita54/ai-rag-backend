"""LLM provider interface."""

from typing import Any, Protocol, runtime_checkable

from app.services.llm.base import LLMMessage, LLMResponse


@runtime_checkable
class LLMPort(Protocol):
    """Contract for LLM providers."""

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given conversation."""
        ...
