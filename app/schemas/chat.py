"""Pydantic schemas for conversational RAG."""

from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_LENGTH = 4000


class ChatRequest(BaseModel):
    """Incoming chat message for the RAG endpoint."""

    conversation_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("conversation_id", "message")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        """Reject whitespace-only values."""
        if not value.strip():
            raise ValueError("field must not be empty")
        return value


class SourceResponse(BaseModel):
    """A retrieved chunk cited as a source for an answer."""

    filename: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    """RAG answer with the retrieved sources."""

    conversation_id: str
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
