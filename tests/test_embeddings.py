import numpy as np
import pytest

from app.services.embeddings import EmbeddingService


class FakeEncoder:
    def encode(self, values, normalize_embeddings):
        assert normalize_embeddings is True
        if isinstance(values, str):
            return np.array([3.0, 4.0])
        return np.array([[1.0, 0.0], [0.0, 1.0]])


def test_embedding_service_returns_lists_and_injects_encoder() -> None:
    service = EmbeddingService(encoder=FakeEncoder())

    assert service.embed("hello") == [3.0, 4.0]
    assert service.embed_many(["one", "two"]) == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize("value", ["", "   "])
def test_embedding_service_rejects_empty_text(value: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        EmbeddingService(encoder=FakeEncoder()).embed(value)


def test_embedding_service_returns_empty_batch_without_loading_model() -> None:
    service = EmbeddingService()

    assert service.embed_many([]) == []
    assert service._encoder is None