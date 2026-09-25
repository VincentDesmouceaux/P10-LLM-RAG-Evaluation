import argparse
import logging
from typing import Optional

from sportsee.core.config import INPUT_DIR
from sportsee.rag.data_loader import (
    download_and_extract_zip,
    load_and_parse_files,
)
from sportsee.rag.vector_store import VectorStoreManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_indexing(
    input_directory: str,
    data_url: Optional[str] = None,
    semantic_audit: bool = False,
    semantic_audit_sample_size: int = 5,
) -> None:
    """
    Exécute le pipeline complet d'indexation RAG.

    Étapes :
    1. téléchargement/extraction optionnels ;
    2. chargement et parsing des sources ;
    3. chunking et validation Pydantic ;
    4. audit sémantique Pydantic AI optionnel ;
    5. génération et validation des embeddings ;
    6. construction et sauvegarde de l'index FAISS.
    """

    logging.info(
        "--- Démarrage du processus d'indexation ---"
    )

    if (
        semantic_audit
        and semantic_audit_sample_size <= 0
    ):
        logging.error(
            "La taille de l'échantillon d'audit "
            "doit être strictement supérieure à 0."
        )
        return

    # 1. Téléchargement / extraction optionnels
    if data_url:
        logging.info(
            "Téléchargement des données depuis : %s",
            data_url,
        )

        success = download_and_extract_zip(
            data_url,
            input_directory,
        )

        if not success:
            logging.error(
                "Échec du téléchargement ou "
                "de l'extraction. Arrêt."
            )
            return

    else:
        logging.info(
            "Utilisation des fichiers locaux : %s",
            input_directory,
        )

    # 2. Chargement et parsing
    logging.info(
        "Chargement et parsing des fichiers..."
    )

    documents = load_and_parse_files(
        input_directory
    )

    if not documents:
        logging.warning(
            "Aucun document n'a été chargé. "
            "Vérifiez le dossier d'entrée."
        )
        return

    logging.info(
        "%s documents chargés.",
        len(documents),
    )

    # 3. Initialisation du vector store
    logging.info(
        "Initialisation du VectorStoreManager..."
    )

    vector_store = VectorStoreManager()

    # 4. Configuration éventuelle de l'audit Pydantic AI
    if semantic_audit:
        logging.info(
            "Audit sémantique Pydantic AI activé "
            "(échantillon=%s).",
            semantic_audit_sample_size,
        )
    else:
        logging.info(
            "Audit sémantique Pydantic AI désactivé."
        )

    # 5. Construction du pipeline RAG / FAISS
    logging.info(
        "Construction de l'index FAISS..."
    )

    vector_store.build_index(
        documents,
        semantic_audit=semantic_audit,
        semantic_audit_sample_size=(
            semantic_audit_sample_size
        ),
    )

    # 6. Résultat
    if vector_store.index is None:
        logging.error(
            "L'index FAISS n'a pas pu être construit."
        )
        return

    logging.info(
        "--- Indexation terminée avec succès ---"
    )

    logging.info(
        "Documents traités : %s",
        len(documents),
    )

    logging.info(
        "Chunks indexés : %s",
        vector_store.index.ntotal,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construit l'interface CLI du pipeline d'indexation."""

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline d'indexation RAG SportSee "
            "avec Pydantic, Pydantic AI et FAISS."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=(
            "Répertoire contenant les données sources "
            f"(défaut : {INPUT_DIR})."
        ),
    )

    parser.add_argument(
        "--data-url",
        type=str,
        default=None,
        help=(
            "URL optionnelle d'une archive à télécharger "
            "et extraire avant indexation."
        ),
    )

    parser.add_argument(
        "--semantic-audit",
        action="store_true",
        help=(
            "Active l'audit sémantique des chunks "
            "avec Pydantic AI et Ollama."
        ),
    )

    parser.add_argument(
        "--semantic-audit-sample-size",
        type=int,
        default=5,
        help=(
            "Nombre de chunks audités par Pydantic AI "
            "(défaut : 5)."
        ),
    )

    return parser


def main() -> None:
    """Point d'entrée CLI."""

    parser = build_parser()
    args = parser.parse_args()

    run_indexing(
        input_directory=args.input_dir,
        data_url=args.data_url,
        semantic_audit=args.semantic_audit,
        semantic_audit_sample_size=(
            args.semantic_audit_sample_size
        ),
    )


if __name__ == "__main__":
    main()