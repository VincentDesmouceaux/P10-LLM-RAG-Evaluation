from typing import Any

from pydantic import BaseModel, Field


class RAGQuery(BaseModel):
    question: str = Field(
        min_length=3,
        description="Question utilisateur envoyée au système RAG.",
    )


class DocumentChunk(BaseModel):
    text: str = Field(
        min_length=1,
        description="Contenu textuel du chunk.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées associées au document.",
    )


class RetrievalResult(BaseModel):
    text: str = Field(
        min_length=1,
        description="Texte du chunk retrouvé.",
    )
    score: float = Field(
        ge=0.0,
        le=100.0,
        description="Score de similarité exprimé en pourcentage.",
    )
    raw_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score de similarité brut utilisé par FAISS.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class RAGAnswer(BaseModel):
    question: str = Field(
        min_length=3,
    )
    answer: str = Field(
        min_length=1,
    )
    sources: list[str] = Field(
        default_factory=list,
    )
