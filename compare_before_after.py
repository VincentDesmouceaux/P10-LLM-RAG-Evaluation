from pathlib import Path

import pandas as pd

from hybrid_agent import HybridNBAAgent
from utils.structured_answer import generate_structured_answer
from utils.vector_store import VectorStoreManager


OUTPUT_PATH = Path(
    "evaluation_results/"
    "before_after_hybrid_comparison.csv"
)


TEST_CASES = [
    {
        "category": "simple_numeric",
        "question": (
            "Quel joueur a marqué "
            "le plus de points ?"
        ),
        "expected_terms": [
            "Shai",
            "2485",
        ],
    },
    {
        "category": "team_numeric",
        "question": (
            "Combien de joueurs jouent "
            "pour OKC ?"
        ),
        "expected_terms": [
            "18",
        ],
    },
    {
        "category": "team_numeric",
        "question": (
            "Quelle équipe totalise "
            "le plus de points ?"
        ),
        "expected_terms": [
            "Detroit",
        ],
    },
    {
        "category": "complex_numeric",
        "question": (
            "Parmi MIA, OKC, LAC, "
            "BKN et ATL, quelle équipe "
            "a marqué le plus de points, "
            "laquelle en a marqué le moins "
            "et quelle est la différence ?"
        ),
        "expected_terms": [
            "OKC",
            "9880",
            "BKN",
            "7999",
            "1881",
        ],
    },
    {
        "category": "noisy_numeric",
        "question": (
            "J'ai vu plein de commentaires "
            "Reddit contradictoires mais bref "
            "kel joueur a marker "
            "le + de points ???"
        ),
        "expected_terms": [
            "Shai",
            "2485",
        ],
    },
]


def contains_terms(
    answer: str,
    terms: list[str],
) -> bool:
    normalized = answer.lower()

    return all(
        term.lower() in normalized
        for term in terms
    )


def baseline_rag_answer(
    vector_store,
    question,
):
    results = vector_store.search(
        question,
        k=5,
    )

    context_parts = []

    for result in results:
        metadata = result.get(
            "metadata",
            {},
        )

        source = (
            metadata.get("filename")
            or metadata.get("source")
            or "source inconnue"
        )

        context_parts.append(
            (
                f"SOURCE: {source}\n"
                f"TEXTE:\n"
                f"{result['text']}"
            )
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    structured = generate_structured_answer(
        question=question,
        context=context,
    )

    return structured.answer


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = VectorStoreManager()
    hybrid_agent = HybridNBAAgent()

    rows = []

    for case in TEST_CASES:
        question = case["question"]
        expected = case["expected_terms"]

        print()
        print("=" * 100)
        print(case["category"])
        print(question)
        print("=" * 100)

        before_answer = baseline_rag_answer(
            vector_store,
            question,
        )

        after_result = hybrid_agent.ask(
            question
        )

        after_answer = (
            after_result["answer"]
        )

        before_ok = contains_terms(
            before_answer,
            expected,
        )

        after_ok = contains_terms(
            after_answer,
            expected,
        )

        print(
            "AVANT :",
            "OK" if before_ok else "ECHEC",
        )
        print(before_answer)

        print()
        print(
            "APRES :",
            "OK" if after_ok else "ECHEC",
        )
        print(
            "ROUTE :",
            after_result["route"],
        )
        print(after_answer)

        rows.append(
            {
                "category": case["category"],
                "question": question,
                "expected_terms": " | ".join(
                    expected
                ),
                "before_system": "rag_only",
                "before_correct": before_ok,
                "before_answer": before_answer,
                "after_system": "hybrid_rag_sql",
                "after_route": after_result[
                    "route"
                ],
                "after_correct": after_ok,
                "after_answer": after_answer,
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    total = len(df)

    before_score = int(
        df["before_correct"].sum()
    )

    after_score = int(
        df["after_correct"].sum()
    )

    print()
    print("=" * 100)
    print("RESULTATS AVANT / APRES")
    print("=" * 100)

    print(
        f"RAG seul        : "
        f"{before_score}/{total} "
        f"({before_score / total:.1%})"
    )

    print(
        f"RAG + SQL Tool  : "
        f"{after_score}/{total} "
        f"({after_score / total:.1%})"
    )

    print()
    print(
        "CSV :",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
