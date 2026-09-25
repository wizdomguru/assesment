from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from app.models.documents import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str, model: Any = None):
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, question: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self.model.predict([(question, item.chunk.text) for item in candidates])
        ranked = [
            replace(item, rerank_score=float(score))
            for item, score in zip(candidates, scores)
        ]
        return sorted(ranked, key=lambda item: item.rerank_score or 0, reverse=True)[:top_k]