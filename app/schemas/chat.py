"""Pydantic schemas for conversational RAG."""

from pydantic import BaseModel, Field, field_validator

from app.schemas.booking import InterviewBookingResponse


class ChatRequest(BaseModel):
    """Incoming chat message for the RAG endpoint."""

    message: str = Field(min_length=1)
    conversation_id: str | None = None

    @field_validator("conversation_id", mode="before")
    @classmethod
    def empty_conversation_id_to_none(cls, value: object) -> object:
        """Treat an empty conversation id as a new conversation."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


class SourceResponse(BaseModel):
    """A retrieved chunk cited as a source for an answer."""

    document_id: int
    filename: str
    chunk_index: int
    text: str


class ChatResponse(BaseModel):
    """RAG answer with retrieved sources and optional booking result."""

    conversation_id: str
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    booking: InterviewBookingResponse | None = None
