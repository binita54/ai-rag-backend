"""LLM provider for OpenAI-compatible HTTP APIs (OpenAI, Ollama /v1, Groq, and others)."""

import json
from typing import Any

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.services.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCall,
)

LOCAL_ENDPOINT_API_KEY = "local-endpoint"


class LLMProviderError(Exception):
    """Raised when the LLM provider fails or returns an invalid response."""


class OpenAICompatibleProvider(BaseLLMProvider):
    """LLM provider backed by any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.LLM_MODEL
        self._temperature = (
            settings.LLM_TEMPERATURE if temperature is None else temperature
        )
        if client is not None:
            self._client = client
        else:
            self._client = AsyncOpenAI(
                base_url=base_url or settings.LLM_BASE_URL,
                api_key=(
                    api_key or settings.LLM_API_KEY or LOCAL_ENDPOINT_API_KEY
                ),
                timeout=settings.LLM_TIMEOUT if timeout is None else timeout,
            )

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate a completion via the OpenAI-compatible chat completions API."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except Exception as error:
            raise LLMProviderError(_safe_error_message(error)) from error

        if not completion.choices:
            raise LLMProviderError("LLM response contained no choices")
        choice = completion.choices[0]
        content = choice.message.content

        tool_calls = []
        for call in choice.message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(
                ToolCall(name=call.function.name, arguments=arguments)
            )

        if not content or not content.strip():
            if not tool_calls:
                raise LLMProviderError("LLM returned an empty response")
            content = ""

        return LLMResponse(content=content, tool_calls=tool_calls)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()


def _safe_error_message(error: Exception) -> str:
    """Build a provider error message without exposing credentials."""
    status_code = getattr(error, "status_code", None)
    message = getattr(error, "message", None) or str(error)
    prefix = f"HTTP {status_code}" if status_code is not None else "Provider"
    return f"{prefix} error: {message}"
