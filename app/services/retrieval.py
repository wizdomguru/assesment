from dataclasses import dataclass
from typing import Sequence

from app.models.documents import RetrievedChunk


@dataclass(frozen=True)
class CalibrationExample:
    score: float
    relevant: bool


@dataclass(frozen=True)
class RetrievalDecision:
    sufficient: bool
    chunks: list[RetrievedChunk]
    reason: str


def calibrate_threshold(examples: Sequence[CalibrationExample]) -> float:
    """Choose a separating midpoint from labeled retrieval scores."""
    relevant = [example.score for example in examples if example.relevant]
    irrelevant = [example.score for example in examples if not example.relevant]
    if not relevant or not irrelevant:
        raise ValueError("Calibration requires relevant and irrelevant examples")

    lowest_relevant = min(relevant)
    highest_irrelevant = max(irrelevant)
    if lowest_relevant <= highest_irrelevant:
        raise ValueError("Calibration examples do not have a separable threshold")
    return (lowest_relevant + highest_irrelevant) / 2


class RetrievalValidator:
    def __init__(self, score_threshold: float, minimum_chunks: int = 1):
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between zero and one")
        if minimum_chunks <= 0:
            raise ValueError("minimum_chunks must be greater than zero")
        self.score_threshold = score_threshold
        self.minimum_chunks = minimum_chunks

    def validate(self, results: Sequence[RetrievedChunk]) -> RetrievalDecision:
        relevant = [
            result for result in results if result.score >= self.score_threshold
        ]
        if len(relevant) < self.minimum_chunks:
            return RetrievalDecision(
                sufficient=False,
                chunks=[],
                reason="Retrieved context did not meet the calibrated relevance threshold",
            )
        return RetrievalDecision(
            sufficient=True,
            chunks=relevant,
            reason="Retrieved context passed relevance validation",
        )