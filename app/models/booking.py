"""SQLAlchemy model for interview bookings."""

from datetime import date, datetime, time

from sqlalchemy import (
    Date,
    DateTime,
    Integer,
    String,
    Time,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class InterviewBooking(Base):
    """An interview booking created through the chat LLM."""

    __tablename__ = "interview_bookings"
    __table_args__ = (
        UniqueConstraint(
            "name",
            "email",
            "date",
            "time",
            name="uq_interview_bookings_name_email_date_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    time: Mapped[time] = mapped_column(Time, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
