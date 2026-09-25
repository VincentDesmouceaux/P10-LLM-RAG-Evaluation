import json

from ollama import chat
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    FunctionModel,
)


OLLAMA_MODEL = "qwen2.5:7b-instruct"


class ChunkQualityAssessment(BaseModel):
    is_valid: bool

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Score normalisé entre 0.0 et 1.0. "
            "Exemple : 0.8 et jamais 8."
        ),
    )

    readability_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Score normalisé entre 0.0 et 1.0. "
            "Exemple : 0.9 et jamais 9."
        ),
    )

    reason: str = Field(
        min_length=3,
    )


def _normalize_score(value) -> float:
    score = float(value)

    if 0.0 <= score <= 1.0:
        return score

    if 1.0 < score <= 10.0:
        return score / 10.0

    raise ValueError(
        f"Score hors plage : {score}"
    )


def validate_chunk_semantically(
    text: str,
) -> ChunkQualityAssessment:
    def ollama_model_function(
        messages,
        info: AgentInfo,
    ) -> ModelResponse:
        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es chargé d'évaluer la qualité "
                        "d'un chunk destiné à un système RAG "
                        "sur le basketball NBA.\n\n"
                        "Évalue uniquement sa qualité pour "
                        "l'indexation et la recherche.\n\n"
                        "Un bon chunk doit :\n"
                        "- contenir une information exploitable ;\n"
                        "- être suffisamment lisible ;\n"
                        "- ne pas être composé uniquement "
                        "de menus, URLs ou bruit OCR ;\n"
                        "- conserver un sens compréhensible.\n\n"
                        "IMPORTANT :\n"
                        "relevance_score et readability_score "
                        "doivent OBLIGATOIREMENT être des "
                        "nombres décimaux compris entre "
                        "0.0 et 1.0.\n\n"
                        "Exemples valides : "
                        "0.2, 0.75, 0.9, 1.0.\n"
                        "Exemples interdits : "
                        "2, 8, 9, 10.\n\n"
                        "Ne juge pas si une opinion présente "
                        "dans le texte est vraie ou fausse."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Évalue ce chunk pour son utilisation "
                        "dans le pipeline RAG :\n\n"
                        f"{text}"
                    ),
                },
            ],
            format=(
                ChunkQualityAssessment
                .model_json_schema()
            ),
            options={
                "temperature": 0,
            },
        )

        payload = json.loads(
            response.message.content
        )

        payload["relevance_score"] = (
            _normalize_score(
                payload["relevance_score"]
            )
        )

        payload["readability_score"] = (
            _normalize_score(
                payload["readability_score"]
            )
        )

        if not info.output_tools:
            raise RuntimeError(
                "Aucun output tool Pydantic AI."
            )

        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    payload,
                )
            ]
        )

    model = FunctionModel(
        ollama_model_function
    )

    agent = Agent(
        model=model,
        output_type=ChunkQualityAssessment,
        retries=2,
    )

    result = agent.run_sync(
        "Valide la qualité du chunk."
    )

    return result.output


def _sample_chunk_indices(
    total: int,
    sample_size: int,
) -> list[int]:
    """Sélectionne un échantillon déterministe réparti sur le corpus."""

    if total <= 0 or sample_size <= 0:
        return []

    count = min(total, sample_size)

    if count == 1:
        return [0]

    return [
        round(
            index * (total - 1) / (count - 1)
        )
        for index in range(count)
    ]


def audit_chunks_semantically(
    chunks: list[dict],
    sample_size: int = 5,
) -> list[dict]:
    """Audite un échantillon de chunks avec Pydantic AI."""

    results = []

    for chunk_index in _sample_chunk_indices(
        len(chunks),
        sample_size,
    ):
        chunk = chunks[chunk_index]

        assessment = validate_chunk_semantically(
            chunk["text"]
        )

        results.append(
            {
                "chunk_id": chunk["id"],
                "source": (
                    chunk.get("metadata", {})
                    .get("source", "unknown")
                ),
                "assessment": assessment.model_dump(),
            }
        )

    return results

