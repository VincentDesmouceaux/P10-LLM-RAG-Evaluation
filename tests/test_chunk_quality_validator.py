import pytest
from pydantic import ValidationError

from utils.chunk_quality_validator import (
    ChunkQualityAssessment,
    _normalize_score,
)


def test_normalized_score_is_preserved():
    assert _normalize_score(0.95) == 0.95


def test_score_on_ten_is_normalized():
    assert _normalize_score(8) == 0.8
    assert _normalize_score(9) == 0.9
    assert _normalize_score(10) == 1.0


def test_invalid_score_is_rejected():
    with pytest.raises(ValueError):
        _normalize_score(11)


def test_chunk_quality_assessment_valid():
    result = ChunkQualityAssessment(
        is_valid=True,
        relevance_score=0.9,
        readability_score=0.8,
        reason="Chunk exploitable pour le RAG.",
    )

    assert result.is_valid is True
    assert result.relevance_score == 0.9
    assert result.readability_score == 0.8


def test_chunk_quality_assessment_rejects_out_of_range():
    with pytest.raises(ValidationError):
        ChunkQualityAssessment(
            is_valid=True,
            relevance_score=1.2,
            readability_score=0.8,
            reason="Score invalide.",
        )
