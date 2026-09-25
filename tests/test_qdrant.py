from dataclasses import dataclass
from uuid import UUID

from app.models.documents import DocumentChunk
from app.services.qdrant import QdrantService


@dataclass
class FakePoint:
    payload: dict
    score: float


class FakeQdrantClient:
    def __init__(self) -> None:
        self.deleted = []
        self.upserted = []
        self.queried = []
        self.response_payload = {}

    def collection_exists(self, collection_name):
        return True

    def delete(self, **kwargs):
        self.deleted.append(kwargs)

    def upsert(self, **kwargs):
        self.upserted.append(kwargs)

    def query_points(self, **kwargs):
        self.queried.append(kwargs)
        return type("Response", (), {"points": [FakePoint(self.response_payload, 0.91)]})


def test_replace_document_deletes_old_points_and_stores_metadata() -> None:
    client = FakeQdrantClient()
    service = QdrantService("unused", "chunks", 2, client=client)
    chunk = DocumentChunk("OAuth details", "doc-1", "guide.md", "doc-1-0", "Auth", None)

    count = service.replace_document("doc-1", [chunk], [[0.1, 0.2]])

    assert count == 1
    assert len(client.deleted) == 1
    point = client.upserted[0]["points"][0]
    UUID(point.id)
    assert point.payload["chunk_id"] == "doc-1-0"
    assert point.payload["section"] == "Auth"


def test_search_reconstructs_chunk_and_score() -> None:
    client = FakeQdrantClient()
    service = QdrantService("unused", "chunks", 2, client=client)
    client.response_payload = {
        "text": "OAuth details",
        "document_id": "doc-1",
        "filename": "guide.md",
        "chunk_id": "doc-1-0",
        "section": "Auth",
        "page_number": 2,
    }

    results = service.search([0.1, 0.2], top_k=3, score_threshold=0.7)

    assert results[0].chunk.filename == "guide.md"
    assert results[0].chunk.page_number == 2
    assert results[0].score == 0.91
    assert client.queried[0]["score_threshold"] == 0.7