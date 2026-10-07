"""Pydantic schemas for document ingestion."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class ChunkStrategy(str, Enum):
    """Selectable text chunking strategies."""

    RECURSIVE = "recursive"
    SENTENCE = "sentence"


class DocumentResponse(BaseModel):
    """Metadata of an ingested document."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: str
    chunk_strategy: str
    chunk_count: int
    created_at: datetime


class DocumentChunkResponse(BaseModel):
    """A stored text chunk with its source document reference."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    chunk_index: int
    text: str
    created_at: datetime
