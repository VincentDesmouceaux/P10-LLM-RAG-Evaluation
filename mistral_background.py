import json
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluate_ragas import (
    TEST_CASES,
    generate_response,
    retrieve_contexts,
)
from sportsee.rag.vector_store import VectorStoreManager


CHECK_INTERVAL_SECONDS = 600

CACHE_DIR = Path("evaluation_cache")
CACHE_FILE = CACHE_DIR / "mistral_response.json"


def save_response(
    question: str,
    reference: str,
    category: str,
    response: str,
) -> None:
    """Sauvegarde une réponse Mistral disponible pour évaluation ultérieure."""

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "category": category,
        "question": question,
        "reference": reference,
        "response": response,
    }

    CACHE_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Teste Mistral périodiquement et quitte dès qu'une réponse est obtenue."""

    test_case = TEST_CASES[0]

    print(
        "Background Mistral démarré.",
        flush=True,
    )
    print(
        f"Question : {test_case.question}",
        flush=True,
    )
    print(
        f"Intervalle : {CHECK_INTERVAL_SECONDS} secondes",
        flush=True,
    )

    vector_store = VectorStoreManager()

    search_results = retrieve_contexts(
        test_case.question,
        vector_store,
    )

    while True:
        now = datetime.now(
            timezone.utc
        ).isoformat()

        print(
            f"[{now}] Test Mistral...",
            flush=True,
        )

        response = generate_response(
            test_case.question,
            search_results,
        )

        if response is not None:
            save_response(
                question=test_case.question,
                reference=test_case.reference,
                category=test_case.category,
                response=response,
            )

            print(
                f"Réponse sauvegardée dans {CACHE_FILE}",
                flush=True,
            )
            print(
                "Background terminé.",
                flush=True,
            )

            break

        print(
            "Mistral indisponible. "
            f"Nouvel essai dans {CHECK_INTERVAL_SECONDS} secondes.",
            flush=True,
        )

        time.sleep(
            CHECK_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    main()
