from typing import Literal

from langchain_ollama import ChatOllama
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIStatusException
from mistralai.models.chat_completion import ChatMessage
from pydantic import BaseModel, Field

from ragas import (
    EvaluationDataset,
    SingleTurnSample,
    evaluate,
)
from ragas.embeddings import HuggingfaceEmbeddings
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from utils.config import (
    EMBEDDING_MODEL,
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,
)
from utils.vector_store import VectorStoreManager


OLLAMA_JUDGE_MODEL = "qwen2.5:7b-instruct"


class EvaluationCase(BaseModel):
    """Cas de test validé pour l'évaluation du RAG."""

    category: Literal["simple", "complex", "noisy"]
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)


TEST_CASES = [
    EvaluationCase(
        category="simple",
        question="Quel joueur a marqué le plus de points ?",
        reference=(
            "Shai Gilgeous-Alexander est le joueur ayant marqué "
            "le plus de points, avec 2485 points."
        ),
    )
]


SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""


def retrieve_contexts(
    question: str,
    vector_store: VectorStoreManager,
) -> list[dict]:
    """Récupère les chunks FAISS associés à une question."""

    return vector_store.search(
        question,
        k=SEARCH_K,
    )


def build_context(
    search_results: list[dict],
) -> str:
    """Construit le contexte transmis au LLM métier."""

    if not search_results:
        return (
            "Aucune information pertinente trouvée dans "
            "la base de connaissances pour cette question."
        )

    return "\n\n---\n\n".join(
        (
            f"Source: "
            f"{result['metadata'].get('source', 'Inconnue')} "
            f"(Score: {result['score']:.1f}%)\n"
            f"Contenu: {result['text']}"
        )
        for result in search_results
    )


def generate_response(
    question: str,
    search_results: list[dict],
) -> str | None:
    """Génère la réponse métier avec Mistral si l'API est disponible."""

    if not MISTRAL_API_KEY:
        print()
        print("MISTRAL_API_KEY absente.")
        print("Passage en mode retrieval-only.")
        return None

    context_str = build_context(search_results)

    final_prompt = SYSTEM_PROMPT.format(
        context_str=context_str,
        question=question,
    )

    client = MistralClient(
        api_key=MISTRAL_API_KEY,
        max_retries=0,
        timeout=15,
    )

    messages = [
        ChatMessage(
            role="user",
            content=final_prompt,
        )
    ]

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
        )

        return response.choices[0].message.content

    except MistralAPIStatusException as error:
        print()
        print("Mistral indisponible.")
        print(f"Erreur API : {error}")
        print("Passage en mode retrieval-only.")

        return None


def build_dataset(
    test_case: EvaluationCase,
    search_results: list[dict],
    response: str | None = None,
) -> EvaluationDataset:
    """Construit le dataset RAGAS."""

    retrieved_contexts = [
        result["text"]
        for result in search_results
    ]

    sample_kwargs = {
        "user_input": test_case.question,
        "retrieved_contexts": retrieved_contexts,
        "reference": test_case.reference,
    }

    if response is not None:
        sample_kwargs["response"] = response

    sample = SingleTurnSample(
        **sample_kwargs,
    )

    return EvaluationDataset(
        samples=[sample],
    )


def build_judge() -> ChatOllama:
    """Initialise le juge RAGAS local."""

    return ChatOllama(
        model=OLLAMA_JUDGE_MODEL,
        temperature=0.0,
    )


def build_ragas_embeddings() -> HuggingfaceEmbeddings:
    """Initialise les embeddings locaux utilisés par RAGAS."""

    return HuggingfaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def run_retrieval_evaluation(
    dataset: EvaluationDataset,
    judge: ChatOllama,
):
    """Évalue uniquement la qualité du retrieval."""

    return evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
        ],
        llm=judge,
        raise_exceptions=True,
        show_progress=True,
    )


def run_full_evaluation(
    dataset: EvaluationDataset,
    judge: ChatOllama,
):
    """Évalue le retrieval et la réponse générée."""

    embeddings = build_ragas_embeddings()

    return evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=judge,
        embeddings=embeddings,
        raise_exceptions=True,
        show_progress=True,
    )


def main() -> None:
    """Lance l'évaluation du premier cas de test."""

    test_case = TEST_CASES[0]

    print()
    print("=" * 80)
    print("EVALUATION RAG")
    print("=" * 80)

    print(f"Catégorie : {test_case.category}")
    print(f"Question   : {test_case.question}")
    print(f"Référence  : {test_case.reference}")

    vector_store = VectorStoreManager()

    search_results = retrieve_contexts(
        test_case.question,
        vector_store,
    )

    print()
    print("=" * 80)
    print("RETRIEVAL")
    print("=" * 80)

    print(
        f"Contextes récupérés : {len(search_results)}"
    )

    for position, result in enumerate(
        search_results,
        start=1,
    ):
        source = result["metadata"].get(
            "source",
            "Inconnue",
        )

        print(
            f"{position}. {source} "
            f"| score={result['score']:.2f}%"
        )

    print()
    print("=" * 80)
    print("GENERATION")
    print("=" * 80)

    response = generate_response(
        test_case.question,
        search_results,
    )

    judge = build_judge()

    if response is None:
        print()
        print("=" * 80)
        print("MODE RETRIEVAL-ONLY")
        print("=" * 80)

        dataset = build_dataset(
            test_case=test_case,
            search_results=search_results,
        )

        result = run_retrieval_evaluation(
            dataset=dataset,
            judge=judge,
        )

    else:
        print()
        print("Réponse Mistral :")
        print(response)

        print()
        print("=" * 80)
        print("MODE FULL RAG")
        print("=" * 80)

        dataset = build_dataset(
            test_case=test_case,
            search_results=search_results,
            response=response,
        )

        result = run_full_evaluation(
            dataset=dataset,
            judge=judge,
        )

    print()
    print("=" * 80)
    print("RESULTATS RAGAS")
    print("=" * 80)

    print(result)


if __name__ == "__main__":
    main()
