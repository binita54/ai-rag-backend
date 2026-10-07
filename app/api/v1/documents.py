"""Document ingestion API endpoints."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_document_ingestion_service
from app.core.config import get_settings
from app.models.database import get_db
from app.schemas.document import ChunkStrategy, DocumentResponse
from app.services.documents.ingestion import (
    DocumentIngestionError,
    DocumentIngestionService,
    DocumentProcessingError,
)

documents_router = APIRouter(prefix="/documents", tags=["documents"])


@documents_router.post("", status_code=201, response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: ChunkStrategy = Form(...),
    chunk_size: int = Form(1000),
    overlap: int = Form(200),
    session: AsyncSession = Depends(get_db),
    ingestion_service: DocumentIngestionService = Depends(
        get_document_ingestion_service
    ),
) -> DocumentResponse:
    """Ingest an uploaded PDF or TXT document."""
    settings = get_settings()
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413, detail="Uploaded file exceeds the size limit"
        )

    try:
        document = await ingestion_service.ingest(
            session=session,
            filename=file.filename or "unknown",
            content=content,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except DocumentProcessingError as error:
        raise HTTPException(
            status_code=502,
            detail="Document vector storage service unavailable",
        ) from error
    except DocumentIngestionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return DocumentResponse.model_validate(document)
