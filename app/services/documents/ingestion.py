"""Document ingestion orchestration."""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.ports.embeddings import EmbeddingPort
from app.ports.vector_store import VectorStorePort
from app.schemas.document import ChunkStrategy
from app.services.documents.chunker import ChunkingError, chunk_text
from app.services.documents.extractor import (
    DocumentExtractionError,
    extract_text,
)
from app.services.embeddings import EmbeddingError
from app.services.vector_store import VectorStoreError, chunk_point_id


class DocumentIngestionError(Exception):
    """Raised when document ingestion fails due to invalid input."""


class DocumentProcessingError(DocumentIngestionError):
    """Raised when embedding or vector storage fails."""


class DocumentIngestionService:
    """Orchestrates extract, chunk, persist, embed, and vector upsert."""

    def __init__(
        self,
        embedding_service: EmbeddingPort,
        vector_store: VectorStorePort,
        collection_name: str,
    ) -> None:
        self._embeddings = embedding_service
        self._vector_store = vector_store
        self._collection_name = collection_name

    async def ingest(
        self,
        session: AsyncSession,
        filename: str,
        content: bytes,
        chunk_strategy: ChunkStrategy,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> Document:
        """Ingest a document and return the persisted Document record."""
        try:
            text = extract_text(filename, content)
        except DocumentExtractionError as error:
            raise DocumentIngestionError(str(error)) from error

        try:
            chunks = chunk_text(text, chunk_strategy, chunk_size, overlap)
        except ChunkingError as error:
            raise DocumentIngestionError(str(error)) from error

        document = Document(
            filename=filename,
            file_type=Path(filename).suffix.lower().lstrip("."),
            chunk_strategy=chunk_strategy.value,
            chunk_count=len(chunks),
        )
        session.add(document)

        try:
            await session.flush()
            for chunk in chunks:
                session.add(
                    DocumentChunk(
                        document_id=document.id,
                        chunk_index=chunk.index,
                        text=chunk.text,
                    )
                )
            await session.flush()
        except Exception as error:
            raise DocumentIngestionError(
                f"Failed to persist document metadata: {error}"
            ) from error

        try:
            await self._vector_store.ensure_collection(
                self._collection_name, self._embeddings.dimension
            )
            vectors = await self._embeddings.embed_documents(
                [chunk.text for chunk in chunks]
            )
            if len(vectors) != len(chunks):
                raise DocumentProcessingError(
                    "Embedding service returned an unexpected vector count"
                )

            points = [
                {
                    "id": chunk_point_id(document.id, chunk.index),
                    "vector": vector,
                    "payload": {
                        "document_id": document.id,
                        "chunk_id": chunk_point_id(document.id, chunk.index),
                        "chunk_index": chunk.index,
                        "filename": filename,
                        "text": chunk.text,
                        "file_type": document.file_type,
                        "chunk_strategy": document.chunk_strategy,
                    },
                }
                for chunk, vector in zip(chunks, vectors)
            ]
            await self._vector_store.upsert(self._collection_name, points)
        except DocumentProcessingError:
            raise
        except (EmbeddingError, VectorStoreError) as error:
            await self._cleanup_vectors(document.id)
            raise DocumentProcessingError(
                f"Failed to process document vectors: {error}"
            ) from error
        except Exception as error:
            await self._cleanup_vectors(document.id)
            raise DocumentProcessingError(
                f"Failed to process document vectors: {error}"
            ) from error

        await session.commit()
        await session.refresh(document)
        return document

    async def _cleanup_vectors(self, document_id: int) -> None:
        """Best-effort removal of vectors already stored for a failed ingestion."""
        try:
            await self._vector_store.delete_by_document(
                self._collection_name, document_id
            )
        except VectorStoreError:
            pass
