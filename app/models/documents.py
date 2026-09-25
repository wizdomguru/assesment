from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSection:
    text: str
    section: str | None = None
    page_number: int | None = None


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    document_id: str
    filename: str
    chunk_id: str
    section: str | None = None
    page_number: int | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    rerank_score: float | None = None