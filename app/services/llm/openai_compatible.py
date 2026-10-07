"""LLM provider for OpenAI-compatible HTTP APIs (OpenAI, Ollama /v1, Groq, and others)."""

import json
from typing import Any

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.services.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, ToolCall


class OpenAICompatibleProvider(BaseLLMProvider):
    """LLM provider backed by any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            base_url=base_url or settings.LLM_BASE_URL,
            api_key=api_key or settings.LLM_API_KEY,
        )
        self._model = model or settings.LLM_MODEL
        self._temperature = settings.LLM_TEMPERATURE if temperature is None else temperature

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

        completion = await self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]

        tool_calls = []
        for call in choice.message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(name=call.function.name, arguments=arguments))

        return LLMResponse(content=choice.message.content or "", tool_calls=tool_calls)
