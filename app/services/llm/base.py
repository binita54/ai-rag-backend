"""LLM provider abstraction shared by all backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMMessage:
    """A single conversation message."""

    role: str
    content: str


@dataclass
class ToolCall:
    """A tool or function call requested by the model."""

    name: str
    arguments: dict[str, Any]


@dataclass
class LLMTool:
    """A structured tool the model may request."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> dict[str, Any]:
        """Return the OpenAI-compatible tool specification."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class LLMResponse:
    """Provider-agnostic LLM response."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class BaseLLMProvider(ABC):
    """Interface every LLM provider must implement."""

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given conversation."""
