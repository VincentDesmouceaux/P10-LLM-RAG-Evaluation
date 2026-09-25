from pathlib import Path

import pandas as pd

from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from sportsee.agent.hybrid_agent import HybridNBAAgent
from sportsee.rag.vector_store import VectorStoreManager


OLLAMA_MODEL = "qwen2.5:7b-instruct"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

OUTPUT_PATH = Path(
    "evaluation_results/"
    "reddit_ragas_final.csv"
)

ROBUSTNESS_OUTPUT_PATH = Path(
    "evaluation_results/"
    "reddit_robustness.csv"
)


TEST_CASES = [
    {
        "category": "simple",
        "question": (
            "Selon les discussions Reddit, "
            "quelle faiblesse de Reggie Miller "
            "est mentionnée lorsqu'il doit créer "
            "son propre tir avec le ballon, "
            "et quelle qualité liée à son jeu "
            "sans ballon est mise en avant ?"
        ),
        "reference": (
            "Des commentateurs indiquent que Reggie Miller "
            "était moins fiable lorsqu'il devait créer "
            "avec le ballon ou jouer en isolation. "
            "Ils mettent en revanche en avant son mouvement "
            "sans ballon et la gravité offensive qu'il créait."
        ),
    },
    {
        "category": "complex",
        "question": (
            "Selon les discussions Reddit, pourquoi certains "
            "utilisateurs estiment-ils que Reggie Miller "
            "n'était pas un numéro 1 offensif idéal, "
            "et dans quelles conditions pouvait-il malgré "
            "tout remplir ce rôle ?"
        ),
        "reference": (
            "Certains utilisateurs soulignent qu'il créait "
            "peu son propre tir et qu'il bénéficiait d'une "
            "équipe équilibrée. D'autres estiment qu'il "
            "pouvait rester le meilleur scoreur d'une équipe, "
            "mais qu'il avait besoin d'un partenaire capable "
            "de prendre en charge davantage de création "
            "et de scoring avec le ballon."
        ),
    },
    {
        "category": "noisy",
        "question": (
            "Oublie les stats Excel et les classements : "
            "dans les discussions Reddit, Hali est-il surtout "
            "décrit comme un joueur très vocal avec ses "
            "coéquipiers ou comme quelqu'un qui trash-talk "
            "constamment ?"
        ),
        "reference": (
            "Il est surtout décrit comme un joueur très vocal : "
            "il encourage ses coéquipiers, leur donne des "
            "informations et parle dans les huddles. "
            "Le trash-talk n'est pas présenté comme constant, "
            "mais comme particulièrement impactant lorsqu'il "
            "se produit."
        ),
    },
]


ROBUSTNESS_CASES = [
    {
        "category": "out_of_domain",
        "question": (
            "Quelle est la capitale de l'Australie ?"
        ),
        "expected_behavior": "abstention",
        "description": (
            "Question hors du domaine NBA. "
            "Le système ne doit pas fabriquer une réponse "
            "à partir du corpus documentaire."
        ),
    },
    {
        "category": "missing_context",
        "question": (
            "Selon les documents Reddit, quel était "
            "le salaire exact de Reggie Miller en 1994 ?"
        ),
        "expected_behavior": "abstention",
        "description": (
            "L'information demandée n'est pas supposée être "
            "présente dans le corpus. Le système doit signaler "
            "que le contexte disponible est insuffisant."
        ),
    },
]


ABSTENTION_MARKERS = (
    "je ne dispose pas",
    "je n'ai pas",
    "aucune information",
    "pas d'information",
    "information n'est pas",
    "information ne figure pas",
    "ne figure pas",
    "les documents ne",
    "le contexte ne",
    "contexte insuffisant",
    "ne permet pas de répondre",
    "impossible de déterminer",
    "hors du domaine",
)


def build_contexts(
    vector_store,
    question,
):
    results = vector_store.search(
        question,
        k=5,
    )

    return [
        result["text"]
        for result in results
    ]


def detect_abstention(answer):
    """Détecte un signal explicite d'abstention dans la réponse."""

    normalized_answer = str(answer).lower()

    return any(
        marker in normalized_answer
        for marker in ABSTENTION_MARKERS
    )


def run_ragas_evaluation(
    vector_store,
    agent,
):
    """Exécute le benchmark RAGAS de référence."""

    samples = []
    metadata = []

    print("=" * 100)
    print("EVALUATION RAGAS - REDDIT")
    print("=" * 100)

    for case in TEST_CASES:
        question = case["question"]

        print()
        print("-" * 100)
        print(
            f"CATEGORIE : "
            f"{case['category']}"
        )
        print("-" * 100)

        print(
            "QUESTION :",
            question,
        )

        contexts = build_contexts(
            vector_store,
            question,
        )

        result = agent.ask(
            question
        )

        print(
            "ROUTE    :",
            result["route"],
        )

        print(
            "REPONSE  :",
            result["answer"],
        )

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=result["answer"],
                retrieved_contexts=contexts,
                reference=case["reference"],
            )
        )

        metadata.append(
            {
                "category": case["category"],
                "route": result["route"],
                "question": question,
                "answer": result["answer"],
                "reference": case["reference"],
            }
        )

    dataset = EvaluationDataset(
        samples=samples
    )

    judge = LangchainLLMWrapper(
        ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.0,
        )
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    print()
    print("=" * 100)
    print("LANCEMENT RAGAS")
    print("=" * 100)

    evaluation = evaluate(
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

    scores = evaluation.to_pandas()

    rows = []

    for index, item in enumerate(
        metadata
    ):
        score = scores.iloc[index]

        rows.append(
            {
                **item,
                "context_precision": score[
                    "context_precision"
                ],
                "context_recall": score[
                    "context_recall"
                ],
                "faithfulness": score[
                    "faithfulness"
                ],
                "answer_relevancy": score[
                    "answer_relevancy"
                ],
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 100)
    print("RESULTATS PAR CATEGORIE")
    print("=" * 100)

    print(
        df[
            [
                "category",
                "route",
                "context_precision",
                "context_recall",
                "faithfulness",
                "answer_relevancy",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("MOYENNES")
    print("=" * 100)

    for metric in [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]:
        print(
            f"{metric:20s} : "
            f"{df[metric].mean():.3f}"
        )

    print()
    print(
        "Routes RAG :",
        f"{(df['route'] == 'rag').sum()}"
        f"/{len(df)}",
    )

    print(
        "CSV RAGAS :",
        OUTPUT_PATH,
    )


def run_robustness_evaluation(
    vector_store,
    agent,
):
    """
    Teste séparément les comportements hors domaine
    et en contexte documentaire insuffisant.

    Ces cas ne sont volontairement pas intégrés aux moyennes RAGAS.
    """

    rows = []

    print()
    print("=" * 100)
    print("TESTS DE ROBUSTESSE")
    print("=" * 100)

    for case in ROBUSTNESS_CASES:
        question = case["question"]

        print()
        print("-" * 100)
        print(
            f"CATEGORIE : "
            f"{case['category']}"
        )
        print("-" * 100)

        print(
            "QUESTION :",
            question,
        )

        contexts = build_contexts(
            vector_store,
            question,
        )

        result = agent.ask(
            question
        )

        answer = result["answer"]

        abstention_detected = detect_abstention(
            answer
        )

        passed = (
            case["expected_behavior"] == "abstention"
            and abstention_detected
        )

        print(
            "ROUTE    :",
            result["route"],
        )

        print(
            "REPONSE  :",
            answer,
        )

        print(
            "ATTENDU  :",
            case["expected_behavior"],
        )

        print(
            "RESULTAT  :",
            "PASS" if passed else "FAIL",
        )

        rows.append(
            {
                "category": case["category"],
                "question": question,
                "route": result["route"],
                "answer": answer,
                "expected_behavior": case[
                    "expected_behavior"
                ],
                "abstention_detected": abstention_detected,
                "passed": passed,
                "retrieved_context_count": len(contexts),
                "description": case["description"],
            }
        )

    robustness_df = pd.DataFrame(
        rows
    )

    robustness_df.to_csv(
        ROBUSTNESS_OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 100)
    print("SYNTHESE ROBUSTESSE")
    print("=" * 100)

    print(
        robustness_df[
            [
                "category",
                "route",
                "expected_behavior",
                "abstention_detected",
                "passed",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Tests réussis :",
        f"{robustness_df['passed'].sum()}"
        f"/{len(robustness_df)}",
    )

    print(
        "CSV robustesse :",
        ROBUSTNESS_OUTPUT_PATH,
    )


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = VectorStoreManager()
    agent = HybridNBAAgent()

    run_ragas_evaluation(
        vector_store,
        agent,
    )

    run_robustness_evaluation(
        vector_store,
        agent,
    )


if __name__ == "__main__":
    main()