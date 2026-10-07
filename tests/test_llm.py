"""Tests for the OpenAI-compatible LLM provider."""

from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIError

from app.core.config import get_settings
from app.services.llm import (
    LLMMessage,
    LLMProviderError,
    OpenAICompatibleProvider,
)

SECRET_API_KEY = "secret-api-key-12345"


class FakeChatCompletions:
    """Fake chat completions API capturing request kwargs."""

    def __init__(
        self,
        response: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeAsyncOpenAIClient:
    """Fake AsyncOpenAI client."""

    def __init__(
        self,
        response: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=FakeChatCompletions(
                response=response, error=error
            )
        )
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _completion(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None)
            )
        ]
    )


def _tool_completion(
    content: str | None, tool_name: str, arguments_json: str
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(
                                name=tool_name,
                                arguments=arguments_json,
                            ),
                        )
                    ],
                )
            )
        ]
    )


def _provider(
    response: Any = None,
    error: Exception | None = None,
    **kwargs: Any,
) -> tuple[OpenAICompatibleProvider, FakeAsyncOpenAIClient]:
    client = FakeAsyncOpenAIClient(response=response, error=error)
    provider = OpenAICompatibleProvider(client=client, **kwargs)
    return provider, client


def _sdk_error() -> APIError:
    return APIError("simulated provider failure", request=None, body=None)


async def test_messages_converted_correctly() -> None:
    provider, client = _provider(response=_completion("answer"))
    messages = [
        LLMMessage(role="user", content="hello"),
        LLMMessage(role="assistant", content="hi there"),
        LLMMessage(role="user", content="what is RAG?"),
    ]

    await provider.chat(messages)

    assert client.chat.completions.calls[0]["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "what is RAG?"},
    ]


async def test_configured_model_used() -> None:
    provider, client = _provider(
        response=_completion("answer"), model="test-model-xyz"
    )

    await provider.chat([LLMMessage(role="user", content="hi")])

    assert client.chat.completions.calls[0]["model"] == "test-model-xyz"


async def test_default_model_from_settings() -> None:
    provider, client = _provider(response=_completion("answer"))

    await provider.chat([LLMMessage(role="user", content="hi")])

    assert (
        client.chat.completions.calls[0]["model"]
        == get_settings().LLM_MODEL
    )


async def test_configured_temperature_used() -> None:
    provider, client = _provider(
        response=_completion("answer"), temperature=0.7
    )

    await provider.chat([LLMMessage(role="user", content="hi")])

    assert client.chat.completions.calls[0]["temperature"] == 0.7


async def test_default_temperature_from_settings() -> None:
    provider, client = _provider(response=_completion("answer"))

    await provider.chat([LLMMessage(role="user", content="hi")])

    assert (
        client.chat.completions.calls[0]["temperature"]
        == get_settings().LLM_TEMPERATURE
    )


async def test_assistant_text_returned() -> None:
    provider, _ = _provider(response=_completion("the answer"))

    response = await provider.chat([LLMMessage(role="user", content="hi")])

    assert response.content == "the answer"
    assert response.tool_calls == []


async def test_empty_content_raises_typed_error() -> None:
    provider, _ = _provider(response=_completion(""))

    with pytest.raises(LLMProviderError):
        await provider.chat([LLMMessage(role="user", content="hi")])


async def test_none_content_raises_typed_error() -> None:
    provider, _ = _provider(response=_completion(None))

    with pytest.raises(LLMProviderError):
        await provider.chat([LLMMessage(role="user", content="hi")])


async def test_missing_choices_raises_typed_error() -> None:
    empty_response = SimpleNamespace(choices=[])
    provider, _ = _provider(response=empty_response)

    with pytest.raises(LLMProviderError):
        await provider.chat([LLMMessage(role="user", content="hi")])


async def test_sdk_failure_wrapped_in_provider_error() -> None:
    provider, _ = _provider(error=_sdk_error())

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.chat([LLMMessage(role="user", content="hi")])

    assert "simulated provider failure" in str(exc_info.value)


async def test_api_key_not_exposed_in_errors() -> None:
    provider, _ = _provider(error=_sdk_error(), api_key=SECRET_API_KEY)

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.chat([LLMMessage(role="user", content="hi")])

    assert SECRET_API_KEY not in str(exc_info.value)


async def test_close_closes_client() -> None:
    provider, client = _provider(response=_completion("answer"))

    await provider.close()

    assert client.closed


async def test_timeout_configurable() -> None:
    provider = OpenAICompatibleProvider(
        api_key="local-endpoint", timeout=12.0
    )
    try:
        assert provider._client.timeout == 12.0
    finally:
        await provider.close()


async def test_tools_are_sent_to_the_sdk() -> None:
    tool_spec = {
        "type": "function",
        "function": {
            "name": "book_interview",
            "description": "Book an interview",
            "parameters": {"type": "object"},
        },
    }
    provider, client = _provider(response=_completion("answer"))

    await provider.chat(
        [LLMMessage(role="user", content="book it")],
        tools=[tool_spec],
    )

    assert client.chat.completions.calls[0]["tools"] == [tool_spec]


async def test_tools_are_omitted_when_not_provided() -> None:
    provider, client = _provider(response=_completion("answer"))

    await provider.chat([LLMMessage(role="user", content="hi")])

    assert "tools" not in client.chat.completions.calls[0]


async def test_tool_call_response_is_parsed() -> None:
    provider, _ = _provider(
        response=_tool_completion(
            "", "book_interview", '{"name": "Binita"}'
        )
    )

    response = await provider.chat([LLMMessage(role="user", content="hi")])

    assert response.content == ""
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "book_interview"
    assert response.tool_calls[0].arguments == {"name": "Binita"}


async def test_empty_content_with_tool_call_is_allowed() -> None:
    provider, _ = _provider(
        response=_tool_completion(None, "book_interview", "{}")
    )

    response = await provider.chat([LLMMessage(role="user", content="hi")])

    assert response.content == ""
    assert response.tool_calls[0].name == "book_interview"


async def test_malformed_tool_arguments_are_handled_safely() -> None:
    provider, _ = _provider(
        response=_tool_completion("", "book_interview", "{not json")
    )

    response = await provider.chat([LLMMessage(role="user", content="hi")])

    assert response.tool_calls[0].arguments == {}


async def test_non_object_tool_arguments_are_handled_safely() -> None:
    provider, _ = _provider(
        response=_tool_completion("", "book_interview", '["oops"]')
    )

    response = await provider.chat([LLMMessage(role="user", content="hi")])

    assert response.tool_calls[0].arguments == {}
