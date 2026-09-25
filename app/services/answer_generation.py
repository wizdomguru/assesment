from collections.abc import Sequence
from typing import Protocol

from app.models.answers import AnswerCitation, AnswerResult
from app.models.documents import RetrievedChunk
from app.services.retrieval import RetrievalDecision


class CompletionClient(Protocol):
    def complete(self, messages: Sequence[dict[str, str]]) -> str: ...


FALLBACK_ANSWER = "I don't have enough information in the indexed documentation to answer that question."


def build_grounded_messages(
    question: str, chunks: Sequence[RetrievedChunk]
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[Source {index}] {result.chunk.text}"
        for index, result in enumerate(chunks, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "Answer only from the supplied documentation. "
                "If the documentation does not contain the answer, say that you "
                "do not have enough information. Do not use outside knowledge, "
                "do not guess, and do not mention sources that are not supplied."
            ),
        },
        {
            "role": "user",
            "content": f"Documentation:\n{context}\n\nQuestion: {question}",
        },
    ]


class AnswerGenerationService:
    def __init__(self, llm: CompletionClient):
        self.llm = llm

    def answer(self, question: str, decision: RetrievalDecision) -> AnswerResult:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not decision.sufficient:
            return AnswerResult(FALLBACK_ANSWER, [], grounded=False)

        answer = self.llm.complete(build_grounded_messages(question, decision.chunks))
        citations = [
            AnswerCitation(
                filename=result.chunk.filename,
                chunk_id=result.chunk.chunk_id,
                page_number=result.chunk.page_number,
                section=result.chunk.section,
            )
            for result in decision.chunks
        ]
        return AnswerResult(answer, citations, grounded=True)