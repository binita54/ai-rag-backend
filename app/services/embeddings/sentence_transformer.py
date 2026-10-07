"""Embedding generation using sentence-transformers."""

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingError(Exception):
    """Base exception for embedding failures."""


class SentenceTransformerEmbeddingService:
    """Local embedding service backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._device = device or settings.EMBEDDING_DEVICE
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        """Name of the embedding model in use."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Dimension of the vectors produced by the model."""
        model = self._load_model()
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    self._model_name, device=self._device
                )
            except Exception as error:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self._model_name}': {error}"
                ) from error
        return self._model

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty query text")
        return (await self.embed_documents([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into normalized vectors."""
        if not texts:
            return []

        model = self._load_model()
        try:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise EmbeddingError(f"Embedding generation failed: {error}") from error

        return vectors.tolist()
