from hybrid_agent import HybridNBAAgent


TEST_CASES = [
    {
        "category": "numeric",
        "expected_route": "sql",
        "question": (
            "Quel joueur a marqué le plus de points ?"
        ),
        "expected_terms": [
            "Shai",
            "2485",
        ],
    },
    {
        "category": "numeric",
        "expected_route": "sql",
        "question": (
            "Combien de joueurs jouent pour OKC ?"
        ),
        "expected_terms": [
            "18",
        ],
    },
    {
        "category": "numeric",
        "expected_route": "sql",
        "question": (
            "Quelle équipe totalise le plus "
            "de points ?"
        ),
        "expected_terms": [],
    },
    {
        "category": "numeric",
        "expected_route": "sql",
        "question": (
            "Quels sont les 5 joueurs ayant "
            "le meilleur pourcentage à 3 points "
            "avec au moins 100 tentatives ?"
        ),
        "expected_terms": [],
    },
    {
        "category": "text",
        "expected_route": "rag",
        "question": (
            "Que disent les discussions Reddit "
            "sur les joueurs NBA ?"
        ),
        "expected_terms": [],
    },
    {
        "category": "text",
        "expected_route": "rag",
        "question": (
            "Quelles opinions des fans apparaissent "
            "dans les PDF Reddit ?"
        ),
        "expected_terms": [],
    },
    {
        "category": "noisy_numeric",
        "expected_route": "sql",
        "question": (
            "J'ai vu plein de commentaires Reddit "
            "mais bref kel joueur a marker "
            "le + de points ???"
        ),
        "expected_terms": [
            "Shai",
            "2485",
        ],
    },
    {
        "category": "noisy_text",
        "expected_route": "rag",
        "question": (
            "Sans calculer de stats, je veux juste "
            "savoir ce que racontent les fans "
            "dans les discussions Reddit."
        ),
        "expected_terms": [],
    },
]


def contains_expected_terms(
    answer: str,
    expected_terms: list[str],
) -> bool:
    if not expected_terms:
        return True

    answer_lower = answer.lower()

    return all(
        term.lower() in answer_lower
        for term in expected_terms
    )


def main():
    agent = HybridNBAAgent()

    route_success = 0
    answer_success = 0

    print("=" * 100)
    print("EVALUATION AGENT HYBRIDE")
    print("=" * 100)

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):
        print()
        print("-" * 100)
        print(
            f"TEST {index} "
            f"[{case['category']}]"
        )
        print("-" * 100)

        print(
            "QUESTION :",
            case["question"],
        )

        result = agent.ask(
            case["question"]
        )

        route_ok = (
            result["route"]
            == case["expected_route"]
        )

        answer_ok = contains_expected_terms(
            result["answer"],
            case["expected_terms"],
        )

        if route_ok:
            route_success += 1

        if answer_ok:
            answer_success += 1

        print(
            "ROUTE ATTENDUE :",
            case["expected_route"],
        )
        print(
            "ROUTE OBTENUE  :",
            result["route"],
        )
        print(
            "ROUTAGE        :",
            "OK" if route_ok else "ECHEC",
        )

        print(
            "REPONSE        :",
            result["answer"],
        )

        print(
            "CONTENU        :",
            "OK" if answer_ok else "ECHEC",
        )

        print(
            "SOURCES        :",
            result.get(
                "sources",
                [],
            ),
        )

        if result["route"] == "sql":
            print(
                "SQL            :",
                result.get(
                    "sql",
                    "",
                ),
            )

    total = len(TEST_CASES)

    print()
    print("=" * 100)
    print("BILAN")
    print("=" * 100)

    print(
        f"Routage correct : "
        f"{route_success}/{total}"
    )

    print(
        f"Contenu valide  : "
        f"{answer_success}/{total}"
    )

    print(
        f"Taux routage    : "
        f"{route_success / total:.1%}"
    )

    if (
        route_success == total
        and answer_success == total
    ):
        print()
        print(
            "Agent hybride : OK"
        )


if __name__ == "__main__":
    main()
