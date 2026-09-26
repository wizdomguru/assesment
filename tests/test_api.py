from fastapi.testclient import TestClient

from app.main import app, get_services
from app.models.answers import AnswerCitation, AnswerResult
from app.models.documents import DocumentChunk, RetrievedChunk


class FakeServices:
    def __init__(self) -> None:
        self.ingested = []
        self.deleted = []
        self.query_calls = 0
        self.query_options = []

    def ingest(self, filename, content):
        self.ingested.append((filename, content))
        return "document-1", 2

    def query(
        self,
        question,
        llm_enabled=True,
        top_k=None,
        retrieval_score_threshold=None,
    ):
        self.query_calls += 1
        self.query_options.append((top_k, retrieval_score_threshold))
        result = AnswerResult(
            "Use OAuth.", [AnswerCitation("guide.md", "document-1-0", 2, "Auth")], True
        )
        chunks = [
            RetrievedChunk(
                DocumentChunk("Use OAuth.", "document-1", "guide.md", "document-1-0", "Auth", 2),
                0.91,
            )
        ]
        return result, self.query_calls > 1, chunks

    def delete(self, document_id):
        self.deleted.append(document_id)


def test_api_upload_query_delete_and_health() -> None:
    services = FakeServices()
    app.dependency_overrides[get_services] = lambda: services
    client = TestClient(app)
    try:
        assert client.get("/health").status_code == 200
        upload = client.post(
            "/documents", files={"file": ("guide.md", b"# Auth\nUse OAuth.", "text/markdown")}
        )
        assert upload.status_code == 201
        assert upload.json()["chunks_indexed"] == 2

        first = client.post(
            "/query",
            json={
                "question": "How?",
                "top_k": 7,
                "retrieval_score_threshold": 0.82,
            },
        )
        second = client.post("/query", json={"question": "How?"})
        assert services.query_options == [(7, 0.82), (None, None)]
        assert first.json()["grounded"] is True
        assert first.json()["cached"] is False
        assert first.json()["retrieved_sources"][0]["chunk_id"] == "document-1-0"
        assert first.json()["retrieved_sources"][0]["section"] == "Auth"
        assert second.json()["cached"] is True

        deleted = client.delete("/documents/document-1")
        assert deleted.status_code == 200
        assert services.deleted == ["document-1"]
    finally:
        app.dependency_overrides.clear()


def test_api_rejects_unsupported_upload() -> None:
    client = TestClient(app)
    app.dependency_overrides[get_services] = lambda: FakeServices()
    try:
        response = client.post("/documents", files={"file": ("notes.txt", b"text")})
        assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_api_rejects_invalid_retrieval_overrides() -> None:
    client = TestClient(app)
    app.dependency_overrides[get_services] = lambda: FakeServices()
    try:
        response = client.post(
            "/query",
            json={"question": "How?", "top_k": 0, "retrieval_score_threshold": 1.2},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()