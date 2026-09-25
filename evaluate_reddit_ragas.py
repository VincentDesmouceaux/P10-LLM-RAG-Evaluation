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


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = VectorStoreManager()
    agent = HybridNBAAgent()

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
        "CSV :",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
