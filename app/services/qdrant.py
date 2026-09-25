from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.models.documents import DocumentChunk, RetrievedChunk


class QdrantService:
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        client: Any = None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = client or QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        self.delete_document(document_id)
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{self.collection_name}:{chunk.chunk_id}")),
                vector=list(vector),
                payload={
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "section": chunk.section,
                    "page_number": chunk.page_number,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    chunk=DocumentChunk(
                        text=payload["text"],
                        document_id=payload["document_id"],
                        filename=payload["filename"],
                        chunk_id=payload["chunk_id"],
                        section=payload.get("section"),
                        page_number=payload.get("page_number"),
                    ),
                    score=point.score,
                )
            )
        return results