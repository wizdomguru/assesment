from app.models.documents import DocumentChunk, RetrievedChunk
from app.services.application import ApplicationServices
from app.services.retrieval import RetrievalValidator


class FakeEmbeddings:
    def embed(self, question):
        return [0.1, 0.2]


class FakeQdrant:
    def __init__(self, candidates):
        self.candidates = candidates
        self.search_options = None

    def search(self, vector, top_k, score_threshold):
        self.search_options = (top_k, score_threshold)
        return self.candidates


class FakeReranker:
    def __init__(self):
        self.top_k = None

    def rerank(self, question, candidates, top_k):
        self.top_k = top_k
        return candidates[:top_k]


class FakeCache:
    pass


def test_query_applies_retrieval_overrides_to_search_reranking_and_validation() -> None:
    candidate = RetrievedChunk(
        DocumentChunk("Use OAuth.", "doc-1", "guide.md", "doc-1-0"),
        0.8,
    )
    qdrant = FakeQdrant([candidate])
    reranker = FakeReranker()
    services = ApplicationServices(
        embeddings=FakeEmbeddings(),
        qdrant=qdrant,
        validator=RetrievalValidator(0.7),
        answers=None,
        cache=FakeCache(),
        reranker=reranker,
        retrieval_top_k=5,
    )

    result, _, _ = services.query(
        "How do I authenticate?",
        llm_enabled=False,
        top_k=3,
        retrieval_score_threshold=0.85,
    )

    assert qdrant.search_options == (6, 0.85)
    assert reranker.top_k == 3
    assert result.citations == []