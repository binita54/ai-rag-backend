"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "Palm Mind AI RAG Backend"
    APP_DEBUG: bool = False
    MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024

    DATABASE_URL: str = "sqlite+aiosqlite:///./data/ai_rag.db"

    QDRANT_MODE: str = "local"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_LOCAL_PATH: str = "./data/qdrant"

    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL_SECONDS: int = 86400
    REDIS_MAX_HISTORY: int = 50

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    LLM_PROVIDER: str = "openai_compatible"
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "llama3.2"
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT: float = 60.0


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
