"""SQLAlchemy models exported for table registration."""

from app.models.booking import InterviewBooking
from app.models.document import Document, DocumentChunk

__all__ = ["Document", "DocumentChunk", "InterviewBooking"]
