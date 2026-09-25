from pathlib import Path

import pandas as pd

from hybrid_agent import HybridNBAAgent
from utils.structured_answer import generate_structured_answer
from sportsee.rag.vector_store import VectorStoreManager


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
        "expected_route": "sql",
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
        "expected_route": "sql",
    },
    {
        "category": "team_numeric",
        "question": (
            "Quelle équipe totalise "
            "le plus de points ?"
        ),
        "expected_terms": [
            "Detroit",
            "10292",
        ],
        "expected_route": "sql",
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
        "expected_route": "sql",
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
        "expected_route": "sql",
    },
    {
        "category": "mixed_text_numeric",
        "question": (
            "Quel joueur a marqué le plus "
            "de points et que disent les "
            "discussions Reddit à son sujet ?"
        ),
        "expected_terms": [
            "Shai",
            "2485",
        ],
        "expected_route": "hybrid",
        "expected_source_fragments": [
            "SQLite NBA",
            "Reddit",
        ],
    },
]


def contains_terms(
    answer: str,
    terms: list[str],
) -> bool:
    """
    Vérifie que tous les termes attendus
    apparaissent dans la réponse.
    """
    normalized = answer.lower()

    return all(
        term.lower() in normalized
        for term in terms
    )


def contains_source_fragments(
    sources: list[str],
    expected_fragments: list[str],
) -> bool:
    """
    Vérifie que les sources contiennent
    les fragments attendus.

    Exemple :
    "Reddit" correspond à "Reddit 1.pdf".
    """
    normalized_sources = [
        str(source).lower()
        for source in sources
    ]

    return all(
        any(
            fragment.lower() in source
            for source in normalized_sources
        )
        for fragment in expected_fragments
    )


def baseline_rag_answer(
    vector_store,
    question: str,
) -> str:
    """
    Produit la réponse du système initial :
    RAG documentaire uniquement.
    """
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
            or metadata.get("file_name")
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


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = VectorStoreManager()
    hybrid_agent = HybridNBAAgent()

    rows = []

    for case in TEST_CASES:
        question = case["question"]

        expected_terms = case[
            "expected_terms"
        ]

        expected_route = case.get(
            "expected_route"
        )

        expected_source_fragments = (
            case.get(
                "expected_source_fragments",
                [],
            )
        )

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

        after_answer = after_result.get(
            "answer",
            "",
        )

        after_route = after_result.get(
            "route",
            "unknown",
        )

        after_sources = after_result.get(
            "sources",
            [],
        )

        before_terms_ok = contains_terms(
            before_answer,
            expected_terms,
        )

        after_terms_ok = contains_terms(
            after_answer,
            expected_terms,
        )

        route_ok = (
            expected_route is None
            or after_route == expected_route
        )

        sources_ok = (
            not expected_source_fragments
            or contains_source_fragments(
                after_sources,
                expected_source_fragments,
            )
        )

        before_ok = before_terms_ok

        after_ok = (
            after_terms_ok
            and route_ok
            and sources_ok
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
            after_route,
        )

        print(
            "ROUTE ATTENDUE :",
            expected_route,
        )

        print(
            "ROUTE CORRECTE :",
            route_ok,
        )

        if expected_source_fragments:
            print(
                "SOURCES CORRECTES :",
                sources_ok,
            )

        print(after_answer)

        rows.append(
            {
                "category": case["category"],
                "question": question,
                "expected_terms": " | ".join(
                    expected_terms
                ),
                "expected_route": expected_route,
                "before_system": "rag_only",
                "before_correct": before_ok,
                "before_answer": before_answer,
                "after_system": (
                    "hybrid_rag_sql"
                ),
                "after_route": after_route,
                "route_correct": route_ok,
                "sources_correct": sources_ok,
                "after_correct": after_ok,
                "after_answer": after_answer,
            }
        )

    df = pd.DataFrame(
        rows
    )

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
    print(
        "RESULTATS AVANT / APRES"
    )
    print("=" * 100)

    print(
        f"RAG seul                 : "
        f"{before_score}/{total} "
        f"({before_score / total:.1%})"
    )

    print(
        f"RAG + SQL + Hybrid       : "
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