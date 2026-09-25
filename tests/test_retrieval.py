import pytest

from app.models.documents import DocumentChunk, RetrievedChunk
from app.services.retrieval import (
    CalibrationExample,
    RetrievalValidator,
    calibrate_threshold,
)


def retrieved(score: float) -> RetrievedChunk:
    chunk = DocumentChunk("content", "doc-1", "guide.md", "doc-1-0")
    return RetrievedChunk(chunk, score)


def test_calibrate_threshold_uses_labeled_scores() -> None:
    examples = [
        CalibrationExample(0.91, True),
        CalibrationExample(0.84, True),
        CalibrationExample(0.42, False),
        CalibrationExample(0.55, False),
    ]

    assert calibrate_threshold(examples) == pytest.approx(0.695)


def test_validator_accepts_only_scores_above_calibrated_threshold() -> None:
    decision = RetrievalValidator(0.695).validate(
        [retrieved(0.8), retrieved(0.69)]
    )

    assert decision.sufficient is True
    assert len(decision.chunks) == 1
    assert decision.chunks[0].score == 0.8


def test_validator_declines_insufficient_context() -> None:
    decision = RetrievalValidator(0.695).validate([retrieved(0.6)])

    assert decision.sufficient is False
    assert decision.chunks == []
    assert "threshold" in decision.reason


def test_calibration_rejects_overlapping_labels() -> None:
    with pytest.raises(ValueError, match="separable"):
        calibrate_threshold(
            [CalibrationExample(0.8, True), CalibrationExample(0.85, False)]
        )