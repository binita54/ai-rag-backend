"""Tests for Pydantic schemas."""

from datetime import date, time

import pytest
from pydantic import ValidationError

from app.schemas.booking import InterviewBookingCreate
from app.schemas.chat import ChatRequest
from app.schemas.document import ChunkStrategy


def test_valid_booking_schema() -> None:
    """A complete booking payload should validate."""
    booking = InterviewBookingCreate(
        name="Jane Doe",
        email="jane.doe@example.com",
        date=date(2026, 10, 10),
        time=time(14, 30),
    )
    assert booking.name == "Jane Doe"
    assert booking.date == date(2026, 10, 10)
    assert booking.time == time(14, 30)


def test_booking_accepts_string_date_and_time() -> None:
    """ISO date and time strings should be parsed."""
    booking = InterviewBookingCreate(
        name="Jane Doe",
        email="jane.doe@example.com",
        date="2026-10-10",
        time="14:30",
    )
    assert booking.date == date(2026, 10, 10)
    assert booking.time == time(14, 30)


def test_invalid_email_rejected() -> None:
    """An invalid email should raise a validation error."""
    with pytest.raises(ValidationError):
        InterviewBookingCreate(
            name="Jane Doe",
            email="not-an-email",
            date="2026-10-10",
            time="14:30",
        )


def test_missing_required_booking_fields_rejected() -> None:
    """Missing required fields should raise a validation error."""
    with pytest.raises(ValidationError):
        InterviewBookingCreate(email="jane.doe@example.com")


def test_empty_name_rejected() -> None:
    """An empty or whitespace-only name should raise a validation error."""
    with pytest.raises(ValidationError):
        InterviewBookingCreate(
            name="   ",
            email="jane.doe@example.com",
            date="2026-10-10",
            time="14:30",
        )


def test_valid_chat_request() -> None:
    """A chat message should validate."""
    request = ChatRequest(message="What is RAG?")
    assert request.message == "What is RAG?"
    assert request.conversation_id is None


def test_chat_request_empty_conversation_id_normalized() -> None:
    """An empty conversation id should be treated as a new conversation."""
    request = ChatRequest(message="Hello", conversation_id="   ")
    assert request.conversation_id is None


def test_chat_request_missing_message_rejected() -> None:
    """A missing message should raise a validation error."""
    with pytest.raises(ValidationError):
        ChatRequest()


def test_chat_request_empty_message_rejected() -> None:
    """An empty message should raise a validation error."""
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chunk_strategy_values() -> None:
    """Both chunking strategies should be selectable by value."""
    assert ChunkStrategy("recursive") is ChunkStrategy.RECURSIVE
    assert ChunkStrategy("sentence") is ChunkStrategy.SENTENCE
