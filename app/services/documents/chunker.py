"""Text chunking strategies for retrieval."""

import re
from dataclasses import dataclass

from app.schemas.document import ChunkStrategy

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])(\s+)")


class ChunkingError(Exception):
    """Raised when chunking cannot be performed."""


@dataclass(frozen=True)
class Chunk:
    """An ordered text chunk, ready for embedding."""

    index: int
    text: str


def chunk_text(
    text: str,
    strategy: ChunkStrategy | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split text into chunks using the selected strategy."""
    if not text or not text.strip():
        raise ChunkingError("Cannot chunk empty text")

    resolved_strategy = ChunkStrategy(strategy)
    _validate_parameters(chunk_size, overlap)

    if resolved_strategy is ChunkStrategy.RECURSIVE:
        pieces = _recursive_pieces(text, RECURSIVE_SEPARATORS)
    else:
        pieces = _sentence_pieces(text)

    fitted = _fit_pieces(pieces, chunk_size)
    merged = _merge_pieces(fitted, chunk_size, overlap)
    return [
        Chunk(index=index, text=chunk_text_value)
        for index, chunk_text_value in enumerate(merged)
    ]


def recursive_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split text recursively at paragraph, line, sentence, and word boundaries."""
    return chunk_text(text, ChunkStrategy.RECURSIVE, chunk_size, overlap)


def sentence_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split text into sentences and group them into chunks."""
    return chunk_text(text, ChunkStrategy.SENTENCE, chunk_size, overlap)


def _validate_parameters(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ChunkingError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ChunkingError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ChunkingError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )


def _recursive_pieces(text: str, separators: list[str]) -> list[str]:
    """Recursively split text, keeping separators attached to their pieces."""
    if not text:
        return []

    for position, separator in enumerate(separators):
        if separator and separator in text:
            parts = text.split(separator)
            pieces: list[str] = []
            for index, part in enumerate(parts):
                piece = part + separator if index < len(parts) - 1 else part
                if piece.strip():
                    pieces.extend(
                        _recursive_pieces(piece, separators[position + 1 :])
                    )
            return pieces

    return [text]


def _sentence_pieces(text: str) -> list[str]:
    """Split text into sentences, keeping trailing whitespace."""
    parts = SENTENCE_SPLIT_PATTERN.split(text)
    pieces: list[str] = []
    for position in range(0, len(parts), 2):
        sentence = parts[position]
        trailing = parts[position + 1] if position + 1 < len(parts) else ""
        piece = sentence + trailing
        if piece.strip():
            pieces.append(piece)
    return pieces


def _fit_pieces(pieces: list[str], chunk_size: int) -> list[str]:
    """Hard-split any single piece that exceeds the chunk size."""
    fitted: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            fitted.append(piece)
        else:
            fitted.extend(
                piece[start : start + chunk_size]
                for start in range(0, len(piece), chunk_size)
            )
    return fitted


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily merge pieces into chunks of at most chunk_size characters."""
    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for piece in pieces:
        if current_parts and current_length + len(piece) > chunk_size:
            chunk_text_value = "".join(current_parts)
            chunks.append(chunk_text_value)
            carried = chunk_text_value[-overlap:] if overlap > 0 else ""
            budget = max(chunk_size - len(piece), 0)
            if budget < len(carried):
                carried = carried[len(carried) - budget :]
            current_parts = [carried] if carried else []
            current_length = len(carried)

        current_parts.append(piece)
        current_length += len(piece)

    if current_parts:
        chunks.append("".join(current_parts))

    return [chunk for chunk in chunks if chunk.strip()]
