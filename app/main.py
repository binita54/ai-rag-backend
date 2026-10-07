"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.models.database import init_db


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize the database before serving requests."""
    await init_db()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.APP_NAME,
        description="Conversational RAG backend with document ingestion and interview booking.",
        version="0.1.0",
        debug=settings.APP_DEBUG,
        lifespan=lifespan,
    )

    application.include_router(api_router)

    @application.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness probe for the service."""
        return {"status": "ok"}

    return application


app = create_app()
