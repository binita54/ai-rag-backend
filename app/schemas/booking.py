"""Pydantic schemas for interview booking."""

import re
from datetime import date, datetime, time

from pydantic import BaseModel, ConfigDict, Field, field_validator

EMAIL_PATTERN = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")


class InterviewBookingCreate(BaseModel):
    """Validated booking payload, typically extracted by the LLM tool call."""

    name: str = Field(min_length=1)
    email: str = Field(min_length=1)
    date: date
    time: time

    @field_validator("name", "email")
    @classmethod
    def strip_and_require_non_empty(cls, value: str) -> str:
        """Trim whitespace and reject empty values."""
        value = value.strip()
        if not value:
            raise ValueError("field must not be empty")
        return value

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        """Require a plausible email address format."""
        if not EMAIL_PATTERN.match(value):
            raise ValueError("invalid email address")
        return value


class InterviewBookingResponse(BaseModel):
    """Stored interview booking."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    date: date
    time: time
    created_at: datetime
