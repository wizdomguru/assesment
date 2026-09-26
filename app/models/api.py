from pydantic import BaseModel, Field

from app.models.answers import AnswerResult
from app.models.documents import RetrievedChunk


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    llm_enabled: bool | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    retrieval_score_threshold: float | None = Field(default=None, ge=0, le=1)


class CitationResponse(BaseModel):
    filename: str
    chunk_id: str
    page_number: int | None = None
    section: str | None = None


class QueryResponse(BaseModel):
    answer: str | None = None
    grounded: bool | None
    citations: list[CitationResponse]
    cached: bool = False
    llm_enabled: bool
    retrieved_sources: list["RetrievedChunkResponse"]

    @classmethod
    def from_result(
        cls,
        result: AnswerResult,
        chunks: list[RetrievedChunk],
        cached: bool = False,
        llm_enabled: bool = True,
    ) -> "QueryResponse":
        return cls(
            answer=result.answer,
            grounded=result.grounded,
            cached=cached,
            citations=(
                [CitationResponse(**citation.__dict__) for citation in result.citations]
                if llm_enabled
                else []
            ),
            llm_enabled=llm_enabled,
            retrieved_sources=[RetrievedChunkResponse.from_chunk(chunk) for chunk in chunks],
        )


class RetrievedChunkResponse(BaseModel):
    text: str
    score: float
    rerank_score: float | None = None
    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None = None
    section: str | None = None

    @classmethod
    def from_chunk(cls, result: RetrievedChunk) -> "RetrievedChunkResponse":
        return cls(
            text=result.chunk.text,
            score=result.score,
            rerank_score=result.rerank_score,
            document_id=result.chunk.document_id,
            filename=result.chunk.filename,
            chunk_id=result.chunk.chunk_id,
            page_number=result.chunk.page_number,
            section=result.chunk.section,
        )


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool = True