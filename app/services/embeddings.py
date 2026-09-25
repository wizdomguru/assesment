from collections.abc import Sequence
from typing import Any


class EmbeddingService:
    """Create normalized embeddings with the configured SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", encoder: Any = None):
        self.model_name = model_name
        self._encoder = encoder

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        vector = self.encoder.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")
        if not texts:
            return []
        vectors = self.encoder.encode(list(texts), normalize_embeddings=True)
        return vectors.tolist()