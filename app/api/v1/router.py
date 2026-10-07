"""Versioned API router. Feature endpoints are attached in later implementation steps."""

from fastapi import APIRouter

documents_router = APIRouter(prefix="/documents", tags=["documents"])
chat_router = APIRouter(prefix="/chat", tags=["chat"])
bookings_router = APIRouter(prefix="/bookings", tags=["bookings"])

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(documents_router)
api_router.include_router(chat_router)
api_router.include_router(bookings_router)
