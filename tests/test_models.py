"""Tests for async database setup and SQLAlchemy models."""

from datetime import date, time

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models import Document, DocumentChunk, InterviewBooking
from app.models import database as database_module
from app.models.database import Base, get_db

EXPECTED_TABLES = {"documents", "document_chunks", "interview_bookings"}


@pytest.fixture
async def db_session(tmp_path):
    """Provide an async session against a temporary SQLite database."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
    await engine.dispose()


async def test_init_db_creates_tables(tmp_path, monkeypatch):
    """init_db should create all expected tables."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/init.db")
    monkeypatch.setattr(database_module, "engine", engine)

    await database_module.init_db()

    async with engine.connect() as connection:
        table_names = await connection.run_sync(
            lambda sync_conn: set(inspect(sync_conn).get_table_names())
        )
    assert EXPECTED_TABLES <= table_names
    await engine.dispose()


async def test_get_db_yields_session(tmp_path, monkeypatch):
    """get_db should yield an async session from the configured engine."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/dep.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(database_module, "engine", engine)
    monkeypatch.setattr(database_module, "AsyncSessionLocal", session_factory)

    async for session in get_db():
        assert isinstance(session, AsyncSession)
        assert session.is_active

    await engine.dispose()


async def test_document_and_chunk_relationship(db_session):
    """One Document should relate to many DocumentChunk records."""
    document = Document(
        filename="notes.txt",
        file_type="txt",
        chunk_strategy="recursive",
    )
    document.chunks = [
        DocumentChunk(chunk_index=0, text="first chunk"),
        DocumentChunk(chunk_index=1, text="second chunk"),
    ]
    db_session.add(document)
    await db_session.commit()
    await db_session.refresh(document)

    assert document.id is not None
    assert document.chunk_count == 0
    assert len(document.chunks) == 2
    assert document.chunks[0].chunk_index == 0
    assert document.chunks[0].document_id == document.id
    assert document.chunks[1].document_id == document.id


async def test_document_chunk_cascade_delete(db_session):
    """Deleting a Document should delete its chunks."""
    document = Document(filename="notes.txt", file_type="txt", chunk_strategy="sentence")
    document.chunks = [DocumentChunk(chunk_index=0, text="chunk")]
    db_session.add(document)
    await db_session.commit()
    document_id = document.id

    await db_session.delete(document)
    await db_session.commit()

    remaining = await db_session.get(DocumentChunk, document.chunks[0].id)
    assert remaining is None
    assert await db_session.get(Document, document_id) is None


async def test_interview_booking_creation(db_session):
    """InterviewBooking records should persist all booking fields."""
    booking = InterviewBooking(
        name="Jane Doe",
        email="jane.doe@example.com",
        date=date(2026, 10, 10),
        time=time(14, 30),
    )
    db_session.add(booking)
    await db_session.commit()
    await db_session.refresh(booking)

    assert booking.id is not None
    assert booking.name == "Jane Doe"
    assert booking.email == "jane.doe@example.com"
    assert booking.date == date(2026, 10, 10)
    assert booking.time == time(14, 30)
    assert booking.created_at is not None
