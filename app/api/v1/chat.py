"""Conversational RAG API endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_rag_service
from app.schemas.chat import ChatRequest, ChatResponse, SourceResponse
from app.services.rag import (
    RAGBookingError,
    RAGError,
    RAGGenerationError,
    RAGRetrievalError,
    RAGService,
)

chat_router = APIRouter(prefix="/chat", tags=["chat"])


@chat_router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """Answer a chat message with retrieved document context."""
    try:
        result = await rag_service.answer(
            request.conversation_id, request.message
        )
    except RAGRetrievalError as error:
        raise HTTPException(
            status_code=502, detail="RAG retrieval service unavailable"
        ) from error
    except RAGGenerationError as error:
        raise HTTPException(
            status_code=502, detail="LLM generation service unavailable"
        ) from error
    except RAGBookingError as error:
        raise HTTPException(
            status_code=500, detail="Failed to book the interview"
        ) from error
    except RAGError as error:
        raise HTTPException(
            status_code=500, detail="Internal RAG error"
        ) from error

    return ChatResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        sources=[
            SourceResponse(
                filename=source.filename,
                chunk_index=source.chunk_index,
                score=source.score,
            )
            for source in result.sources
        ],
    )
