from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerCitation:
    filename: str
    chunk_id: str
    page_number: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[AnswerCitation]
    grounded: bool | None