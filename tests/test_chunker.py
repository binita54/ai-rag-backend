"""Tests for chunking strategies."""

import pytest

from app.schemas.document import ChunkStrategy
from app.services.documents import chunker
from app.services.documents.chunker import (
    Chunk,
    ChunkingError,
    chunk_text,
    recursive_chunk,
    sentence_chunk,
)

WORD_TEXT = " ".join(f"word{index}" for index in range(120))
SENTENCE_TEXT = " ".join(
    f"Sentence number {index} ends here." for index in range(30)
)


def test_recursive_small_text_single_chunk() -> None:
    chunks = recursive_chunk("Small text.", chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text == "Small text."
    assert chunks[0].index == 0


def test_recursive_large_text_splits_within_size() -> None:
    chunks = recursive_chunk(WORD_TEXT, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 50 for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_recursive_overlap_carries_previous_tail() -> None:
    chunks = recursive_chunk(WORD_TEXT, chunk_size=20, overlap=8)
    for previous, current in zip(chunks, chunks[1:]):
        tail = previous.text[-8:]
        assert current.text.startswith(tail)


def test_recursive_no_empty_chunks() -> None:
    chunks = recursive_chunk(WORD_TEXT, chunk_size=15, overlap=0)
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)


def test_recursive_deterministic() -> None:
    first = recursive_chunk(WORD_TEXT, chunk_size=40, overlap=5)
    second = recursive_chunk(WORD_TEXT, chunk_size=40, overlap=5)
    assert first == second


def test_recursive_returns_chunk_dataclass() -> None:
    chunks = recursive_chunk("One two three.", chunk_size=100, overlap=10)
    assert isinstance(chunks[0], Chunk)


def test_sentence_small_text_single_chunk() -> None:
    text = "First sentence. Second sentence."
    chunks = sentence_chunk(text, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_sentence_large_text_splits_within_size() -> None:
    chunks = sentence_chunk(SENTENCE_TEXT, chunk_size=80, overlap=0)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 80 for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_sentence_chunk_preserves_sentence_boundaries() -> None:
    chunks = sentence_chunk(SENTENCE_TEXT, chunk_size=80, overlap=0)
    for chunk in chunks[:-1]:
        assert chunk.text.rstrip().endswith((".", "!", "?"))


def test_sentence_no_empty_chunks() -> None:
    chunks = sentence_chunk(SENTENCE_TEXT, chunk_size=30, overlap=0)
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)


def test_sentence_deterministic() -> None:
    first = sentence_chunk(SENTENCE_TEXT, chunk_size=60, overlap=5)
    second = sentence_chunk(SENTENCE_TEXT, chunk_size=60, overlap=5)
    assert first == second


def test_sentence_chunk_overlap_carries_tail() -> None:
    chunks = sentence_chunk(SENTENCE_TEXT, chunk_size=45, overlap=6)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.text.startswith(previous.text[-6:])


def test_strategy_selection_recursive(monkeypatch) -> None:
    calls: list[str] = []

    def fake_recursive(text: str, separators: list[str]) -> list[str]:
        calls.append("recursive")
        return ["called"]

    monkeypatch.setattr(chunker, "_recursive_pieces", fake_recursive)
    chunks = chunk_text("Any text.", ChunkStrategy.RECURSIVE, chunk_size=100, overlap=10)

    assert calls == ["recursive"]
    assert chunks[0].text == "called"


def test_strategy_selection_sentence(monkeypatch) -> None:
    calls: list[str] = []

    def fake_sentence(text: str) -> list[str]:
        calls.append("sentence")
        return ["called"]

    monkeypatch.setattr(chunker, "_sentence_pieces", fake_sentence)
    chunks = chunk_text("Any text.", ChunkStrategy.SENTENCE, chunk_size=100, overlap=10)

    assert calls == ["sentence"]
    assert chunks[0].text == "called"


def test_strategy_accepts_string_values() -> None:
    recursive_result = chunk_text(WORD_TEXT, "recursive", chunk_size=50, overlap=5)
    sentence_result = chunk_text(SENTENCE_TEXT, "sentence", chunk_size=50, overlap=5)
    assert recursive_result
    assert sentence_result


def test_empty_text_rejected() -> None:
    with pytest.raises(ChunkingError):
        chunk_text("   ", ChunkStrategy.RECURSIVE)


def test_invalid_chunk_size_rejected() -> None:
    with pytest.raises(ChunkingError):
        chunk_text("Some text.", ChunkStrategy.RECURSIVE, chunk_size=0)


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ChunkingError):
        chunk_text("Some text.", ChunkStrategy.RECURSIVE, chunk_size=10, overlap=10)


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_text("Some text.", "semantic")
