"""Interview booking persistence service."""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.booking import InterviewBooking
from app.schemas.booking import InterviewBookingCreate


class BookingError(Exception):
    """Raised when a booking cannot be persisted."""


class BookingService:
    """Persists interview bookings in the metadata database."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def book(
        self, payload: InterviewBookingCreate
    ) -> InterviewBooking:
        """Persist a validated booking, deduplicating retries."""
        try:
            async with self._session_factory() as session:
                existing = await self._find_duplicate(session, payload)
                if existing is not None:
                    return existing

                booking = InterviewBooking(
                    name=payload.name,
                    email=payload.email,
                    date=payload.date,
                    time=payload.time,
                )
                session.add(booking)
                try:
                    await session.commit()
                except IntegrityError:
                    await session.rollback()
                    existing = await self._find_duplicate(session, payload)
                    if existing is not None:
                        return existing
                    raise
                await session.refresh(booking)
                return booking
        except Exception as error:
            raise BookingError(
                f"Failed to persist the interview booking: {error}"
            ) from error

    async def _find_duplicate(
        self,
        session: AsyncSession,
        payload: InterviewBookingCreate,
    ) -> InterviewBooking | None:
        """Return an existing booking identical to the payload."""
        result = await session.execute(
            select(InterviewBooking).where(
                InterviewBooking.name == payload.name,
                InterviewBooking.email == payload.email,
                InterviewBooking.date == payload.date,
                InterviewBooking.time == payload.time,
            )
        )
        return result.scalars().first()
