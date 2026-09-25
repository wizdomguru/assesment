import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

from app.models.answers import AnswerCitation, AnswerResult
from app.services.answer_generation import AnswerGenerationService
from app.services.cache import AnswerCache
from app.services.document_processing import chunk_sections, parse_document
from app.services.embeddings import EmbeddingService
from app.services.qdrant import QdrantService
from app.services.retrieval import RetrievalValidator
from app.models.documents import RetrievedChunk
from app.services.reranking import CrossEncoderReranker


logger = logging.getLogger(__name__)


class ApplicationServices:
    def __init__(
        self,
        embeddings: EmbeddingService,
        qdrant: QdrantService,
        validator: RetrievalValidator,
        answers: AnswerGenerationService,
        cache: AnswerCache,
        reranker: CrossEncoderReranker | None = None,
        retrieval_top_k: int = 5,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.validator = validator
        self.answers = answers
        self.cache = cache
        self.reranker = reranker
        self.retrieval_top_k = retrieval_top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, filename: str, content: bytes) -> tuple[str, int]:
        document_id = hashlib.sha256(filename.lower().encode("utf-8")).hexdigest()[:16]
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            sections = parse_document(temporary_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        chunks = chunk_sections(
            sections,
            document_id=document_id,
            filename=filename,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        if not chunks:
            raise ValueError("Document contains no extractable text")
        vectors = self.embeddings.embed_many([chunk.text for chunk in chunks])
        indexed = self.qdrant.replace_document(document_id, chunks, vectors)
        self.cache.bump_collection_version()
        return document_id, indexed

    def delete(self, document_id: str) -> None:
        self.qdrant.delete_document(document_id)
        self.cache.bump_collection_version()

    def query(
        self, question: str, llm_enabled: bool = True
    ) -> tuple[AnswerResult, bool, list[RetrievedChunk]]:
        vector = self.embeddings.embed(question)
        search_top_k = self.retrieval_top_k * 2 if self.reranker else self.retrieval_top_k
        candidates = self.qdrant.search(
            vector,
            top_k=search_top_k,
            score_threshold=self.validator.score_threshold,
        )
        logger.info(
            "RAG embedding ranking question=%r candidates=%s",
            question,
            [
                {
                    "score": round(candidate.score, 4),
                    "chunk_id": candidate.chunk.chunk_id,
                    "filename": candidate.chunk.filename,
                    "page_number": candidate.chunk.page_number,
                    "section": candidate.chunk.section,
                }
                for candidate in candidates
            ],
        )
        if self.reranker:
            candidates = self.reranker.rerank(question, candidates, self.retrieval_top_k)
        logger.info(
            "RAG retrieval question=%r candidates=%s threshold=%.3f",
            question,
            [
                {
                    "score": round(candidate.score, 4),
                    "chunk_id": candidate.chunk.chunk_id,
                    "filename": candidate.chunk.filename,
                    "page_number": candidate.chunk.page_number,
                    "section": candidate.chunk.section,
                    "rerank_score": (
                        round(candidate.rerank_score, 4)
                        if candidate.rerank_score is not None
                        else None
                    ),
                    "preview": " ".join(candidate.chunk.text.split())[:200],
                }
                for candidate in candidates
            ],
            self.validator.score_threshold,
        )
        decision = self.validator.validate(candidates)
        logger.info(
            "RAG validation sufficient=%s accepted_chunks=%d reason=%s",
            decision.sufficient,
            len(decision.chunks),
            decision.reason,
        )
        if llm_enabled:
            cached_result = self.cache.get(question)
            if cached_result is not None:
                return cached_result, True, list(candidates)
        cached = False
        result = self.answers.answer(question, decision) if llm_enabled else AnswerResult(
            "",
            [
                AnswerCitation(
                    filename=chunk.chunk.filename,
                    chunk_id=chunk.chunk.chunk_id,
                    page_number=chunk.chunk.page_number,
                    section=chunk.chunk.section,
                )
                for chunk in decision.chunks
            ],
            None,
        )
        if llm_enabled:
            self.cache.set(question, result)
        return result, cached, list(candidates)


def build_application_services(settings: Any) -> ApplicationServices:
    from app.services.llm import OpenAICompatibleClient

    if settings.retrieval_score_threshold is None:
        raise RuntimeError("Retrieval score threshold is not configured")
    return ApplicationServices(
        embeddings=EmbeddingService(settings.embedding_model),
        qdrant=QdrantService(
            settings.qdrant_url,
            settings.qdrant_collection,
            settings.embedding_dimension,
        ),
        validator=RetrievalValidator(settings.retrieval_score_threshold),
        answers=AnswerGenerationService(
            OpenAICompatibleClient(
                settings.llm_base_url,
                settings.llm_api_key,
                settings.llm_model,
                settings.llm_timeout_seconds,
            )
        ),
        cache=AnswerCache(
            settings.redis_url,
            settings.cache_ttl_seconds,
            settings.qdrant_collection,
            f"top_k={settings.retrieval_top_k};threshold={settings.retrieval_score_threshold}",
            f"model={settings.llm_model}",
        ),
        retrieval_top_k=settings.retrieval_top_k,
        reranker=(
            CrossEncoderReranker(settings.reranker_model)
            if settings.reranker_enabled
            else None
        ),
    )