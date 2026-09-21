import pytest
from pydantic import ValidationError

from utils.schemas import (
    DocumentChunk,
    RAGAnswer,
    RAGQuery,
    RetrievalResult,
)


def test_rag_query_valid():
    query = RAGQuery(
        question="Quel joueur a marqué le plus de points ?"
    )

    assert query.question.startswith("Quel joueur")


def test_rag_query_rejects_empty_question():
    with pytest.raises(ValidationError):
        RAGQuery(question="")


def test_document_chunk_valid():
    chunk = DocumentChunk(
        text="Shai Gilgeous-Alexander a marqué 2485 points.",
        metadata={
            "source": "regular NBA.xlsx",
            "sheet": "Analyse",
        },
    )

    assert chunk.metadata["source"] == "regular NBA.xlsx"


def test_retrieval_result_valid():
    result = RetrievalResult(
        text="Contenu NBA",
        score=65.18,
        raw_score=0.6518,
        metadata={
            "source": "regular NBA.xlsx",
        },
    )

    assert result.score == 65.18


def test_retrieval_result_rejects_invalid_score():
    with pytest.raises(ValidationError):
        RetrievalResult(
            text="Contenu NBA",
            score=125.0,
            raw_score=0.5,
        )


def test_rag_answer_valid():
    answer = RAGAnswer(
        question="Quel joueur a marqué le plus de points ?",
        answer=(
            "Shai Gilgeous-Alexander a marqué "
            "2485 points."
        ),
        sources=[
            "regular NBA.xlsx",
        ],
    )

    assert len(answer.sources) == 1
