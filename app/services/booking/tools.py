"""LLM tool definitions for interview booking."""

from app.schemas.booking import InterviewBookingCreate
from app.services.llm.base import LLMTool

BOOK_INTERVIEW_TOOL_NAME = "book_interview"

BOOK_INTERVIEW_TOOL = LLMTool(
    name=BOOK_INTERVIEW_TOOL_NAME,
    description=(
        "Book an interview. Call this tool only when the user "
        "clearly wants to schedule an interview and has provided "
        "their name, email address, interview date, and interview "
        "time. If any of these details are missing or unclear, "
        "ask for them instead of inventing values."
    ),
    parameters=InterviewBookingCreate.model_json_schema(),
)
