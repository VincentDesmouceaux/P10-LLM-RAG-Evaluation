from typing import Any

from pydantic import BaseModel, Field, model_validator, field_validator


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
        min_length=20,
    )
    sources: list[str] = Field(
        default_factory=list,
    )

    @field_validator("answer")
    @classmethod
    def validate_complete_answer(
        cls,
        value: str,
    ) -> str:
        answer = value.strip()

        if answer.endswith(":"):
            raise ValueError(
                "La réponse semble incomplète : "
                "elle se termine par deux-points."
            )

        return answer


class PreparedChunk(DocumentChunk):
    id: str = Field(
        min_length=1,
        description="Identifiant unique du chunk.",
    )


class EmbeddingBatch(BaseModel):
    chunk_ids: list[str]
    vectors: list[list[float]]
    expected_dimension: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_embeddings(self):
        import math

        if len(self.chunk_ids) != len(self.vectors):
            raise ValueError(
                "Le nombre de chunk_ids doit correspondre "
                "au nombre de vecteurs."
            )

        for index, vector in enumerate(self.vectors):
            if len(vector) != self.expected_dimension:
                raise ValueError(
                    f"Embedding {index}: dimension "
                    f"{len(vector)} != "
                    f"{self.expected_dimension}"
                )

            if not all(
                math.isfinite(value)
                for value in vector
            ):
                raise ValueError(
                    f"Embedding {index}: valeur NaN ou inf."
                )

        return self
