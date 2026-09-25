from app.models.documents import DocumentChunk, RetrievedChunk
from app.services.reranking import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.2, 0.9]


def test_cross_encoder_reranker_promotes_relevant_candidate() -> None:
    candidates = [
        RetrievedChunk(DocumentChunk("security retries", "doc", "guide.pdf", "1"), 0.51),
        RetrievedChunk(DocumentChunk("cache TTL is 60 seconds", "doc", "guide.pdf", "2"), 0.48),
    ]

    ranked = CrossEncoderReranker("fake", model=FakeCrossEncoder()).rerank(
        "What is the order-status cache TTL?", candidates, top_k=2
    )

    assert ranked[0].chunk.chunk_id == "2"
    assert ranked[0].rerank_score == 0.9