"""Versioned API router."""

from fastapi import APIRouter

from app.api.v1.chat import chat_router
from app.api.v1.documents import documents_router

bookings_router = APIRouter(prefix="/bookings", tags=["bookings"])

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(documents_router)
api_router.include_router(chat_router)
api_router.include_router(bookings_router)
