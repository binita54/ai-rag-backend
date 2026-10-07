"""Tests for the interview booking service and tool schema."""

from datetime import date, time

import pytest
from pydantic import ValidationError
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models.booking import InterviewBooking
from app.models.database import Base
from app.schemas.booking import InterviewBookingCreate
from app.services.booking import (
    BOOK_INTERVIEW_TOOL,
    BOOK_INTERVIEW_TOOL_NAME,
    BookingError,
    BookingService,
)
from app.services.llm import LLMTool


@pytest.fixture
def db_engine(tmp_path):
    return create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/booking-test.db"
    )


@pytest.fixture
async def session_factory(db_engine):
    async with db_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    await db_engine.dispose()


@pytest.fixture
def booking_service(session_factory) -> BookingService:
    return BookingService(session_factory=session_factory)


def _payload(**overrides) -> InterviewBookingCreate:
    values: dict = {
        "name": "Binita Ghale",
        "email": "binita@example.com",
        "date": date(2026, 10, 10),
        "time": time(14, 0),
    }
    values.update(overrides)
    return InterviewBookingCreate(**values)


async def test_valid_booking_is_persisted(
    booking_service: BookingService,
) -> None:
    booking = await booking_service.book(_payload())

    assert booking.id is not None
    assert booking.name == "Binita Ghale"
    assert booking.email == "binita@example.com"
    assert booking.date == date(2026, 10, 10)
    assert booking.time == time(14, 0)


async def test_booking_is_written_to_the_database(
    booking_service: BookingService, session_factory
) -> None:
    await booking_service.book(_payload())

    async with session_factory() as session:
        result = await session.execute(select(InterviewBooking))
        rows = list(result.scalars().all())

    assert len(rows) == 1
    assert rows[0].name == "Binita Ghale"
    assert rows[0].email == "binita@example.com"


async def test_duplicate_booking_returns_existing_row(
    booking_service: BookingService, session_factory
) -> None:
    first = await booking_service.book(_payload())
    second = await booking_service.book(_payload())

    assert second.id == first.id

    async with session_factory() as session:
        result = await session.execute(select(InterviewBooking))
        rows = list(result.scalars().all())

    assert len(rows) == 1


async def test_identical_name_with_different_email_is_new_booking(
    booking_service: BookingService, session_factory
) -> None:
    await booking_service.book(_payload())
    await booking_service.book(_payload(email="other@example.com"))

    async with session_factory() as session:
        result = await session.execute(select(InterviewBooking))
        rows = list(result.scalars().all())

    assert len(rows) == 2


async def test_different_time_creates_separate_booking(
    booking_service: BookingService, session_factory
) -> None:
    await booking_service.book(_payload())
    await booking_service.book(_payload(time=time(15, 30)))

    async with session_factory() as session:
        result = await session.execute(select(InterviewBooking))
        rows = list(result.scalars().all())

    assert len(rows) == 2


async def test_unique_constraint_covers_name_email_date_time(
    db_engine, session_factory
) -> None:
    async with db_engine.connect() as connection:
        def read_constraints(sync_conn) -> list[dict]:
            return inspect(sync_conn).get_unique_constraints(
                "interview_bookings"
            )

        constraints = await connection.run_sync(read_constraints)

    assert any(
        set(constraint["column_names"]) == {"name", "email", "date", "time"}
        for constraint in constraints
    )


async def test_database_rejects_duplicate_row_directly(
    session_factory,
) -> None:
    payload = _payload()

    async with session_factory() as session:
        session.add(
            InterviewBooking(
                name=payload.name,
                email=payload.email,
                date=payload.date,
                time=payload.time,
            )
        )
        await session.commit()

        session.add(
            InterviewBooking(
                name=payload.name,
                email=payload.email,
                date=payload.date,
                time=payload.time,
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()


async def test_concurrent_duplicate_resolves_to_existing_booking(
    booking_service: BookingService,
    session_factory,
    monkeypatch,
) -> None:
    first = await booking_service.book(_payload())

    original_find_duplicate = BookingService._find_duplicate
    checks = {"count": 0}

    async def miss_first_check(self, session, payload):
        checks["count"] += 1
        if checks["count"] == 1:
            return None
        return await original_find_duplicate(self, session, payload)

    monkeypatch.setattr(BookingService, "_find_duplicate", miss_first_check)

    second = await booking_service.book(_payload())

    assert second.id == first.id

    async with session_factory() as session:
        result = await session.execute(select(InterviewBooking))
        rows = list(result.scalars().all())

    assert len(rows) == 1


async def test_unresolved_integrity_error_is_surfaced_as_booking_error(
    booking_service: BookingService,
    session_factory,
    monkeypatch,
) -> None:
    payload = _payload()
    async with session_factory() as session:
        session.add(
            InterviewBooking(
                name=payload.name,
                email=payload.email,
                date=payload.date,
                time=payload.time,
            )
        )
        await session.commit()

    async def find_nothing(self, session, payload):
        return None

    monkeypatch.setattr(BookingService, "_find_duplicate", find_nothing)

    with pytest.raises(BookingError):
        await booking_service.book(payload)


async def test_database_failure_is_surfaced_as_booking_error() -> None:
    class FailingSession:
        async def __aenter__(self) -> None:
            raise SQLAlchemyError("database unavailable")

        async def __aexit__(self, *args) -> bool:
            return False

    class FailingSessionFactory:
        def __call__(self) -> FailingSession:
            return FailingSession()

    service = BookingService(
        session_factory=FailingSessionFactory()
    )

    with pytest.raises(BookingError):
        await service.book(_payload())


def test_blank_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InterviewBookingCreate(
            name="   ",
            email="binita@example.com",
            date=date(2026, 10, 10),
            time=time(14, 0),
        )


def test_invalid_email_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InterviewBookingCreate(
            name="Binita Ghale",
            email="not-an-email",
            date=date(2026, 10, 10),
            time=time(14, 0),
        )


def test_invalid_date_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InterviewBookingCreate(
            name="Binita Ghale",
            email="binita@example.com",
            date="2026-13-45",
            time=time(14, 0),
        )


def test_invalid_time_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InterviewBookingCreate(
            name="Binita Ghale",
            email="binita@example.com",
            date=date(2026, 10, 10),
            time="25:99",
        )


def test_booking_tool_schema_is_generated_from_pydantic_model() -> None:
    assert isinstance(BOOK_INTERVIEW_TOOL, LLMTool)
    assert BOOK_INTERVIEW_TOOL.name == BOOK_INTERVIEW_TOOL_NAME
    assert BOOK_INTERVIEW_TOOL.name == "book_interview"
    assert (
        BOOK_INTERVIEW_TOOL.parameters
        == InterviewBookingCreate.model_json_schema()
    )


def test_booking_tool_spec_is_openai_compatible() -> None:
    spec = BOOK_INTERVIEW_TOOL.to_spec()

    assert spec["type"] == "function"
    assert spec["function"]["name"] == "book_interview"
    assert isinstance(spec["function"]["description"], str)
    parameters = spec["function"]["parameters"]
    assert parameters["type"] == "object"
    assert set(parameters["required"]) == {
        "name",
        "email",
        "date",
        "time",
    }
    assert set(parameters["properties"]) == {
        "name",
        "email",
        "date",
        "time",
    }
