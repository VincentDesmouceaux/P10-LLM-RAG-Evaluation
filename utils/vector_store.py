import logging
from utils.schemas import RAGQuery, RetrievalResult
import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE,
    DOCUMENT_CHUNKS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index FAISS."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, Any]] = []

        logging.info(
            "Chargement du modèle d'embedding local : %s",
            EMBEDDING_MODEL,
        )

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Charge l'index FAISS et les chunks existants."""

        if not (
            os.path.exists(FAISS_INDEX_FILE)
            and os.path.exists(DOCUMENT_CHUNKS_FILE)
        ):
            logging.warning(
                "Fichiers d'index FAISS ou de chunks non trouvés. "
                "L'index est vide."
            )
            return

        try:
            logging.info(
                "Chargement de l'index FAISS depuis %s...",
                FAISS_INDEX_FILE,
            )
            self.index = faiss.read_index(
                FAISS_INDEX_FILE
            )

            logging.info(
                "Chargement des chunks depuis %s...",
                DOCUMENT_CHUNKS_FILE,
            )
            with open(DOCUMENT_CHUNKS_FILE, "rb") as file:
                self.document_chunks = pickle.load(file)

            expected_dimension = (
                self.embedding_model
                .get_embedding_dimension()
            )

            if self.index.d != expected_dimension:
                logging.warning(
                    "Index FAISS incompatible avec le modèle "
                    "d'embedding actuel : index=%s dimensions, "
                    "modèle=%s dimensions.",
                    self.index.d,
                    expected_dimension,
                )
                logging.warning(
                    "L'index doit être reconstruit avec "
                    "'python indexer.py'."
                )

                self.index = None
                self.document_chunks = []
                return

            logging.info(
                "Index (%s vecteurs) et %s chunks chargés.",
                self.index.ntotal,
                len(self.document_chunks),
            )

        except Exception as error:
            logging.error(
                "Erreur lors du chargement de l'index/chunks : %s",
                error,
            )
            self.index = None
            self.document_chunks = []

    def _split_documents_to_chunks(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Découpe les documents en chunks avec métadonnées."""

        logging.info(
            "Découpage de %s documents en chunks "
            "(taille=%s, chevauchement=%s)...",
            len(documents),
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        all_chunks: List[Dict[str, Any]] = []

        for doc_index, doc in enumerate(documents):
            langchain_doc = Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"],
            )

            chunks = text_splitter.split_documents(
                [langchain_doc]
            )

            logging.info(
                "Document '%s' découpé en %s chunks.",
                doc["metadata"].get(
                    "filename",
                    "N/A",
                ),
                len(chunks),
            )

            for chunk_index, chunk in enumerate(chunks):
                chunk_dict = {
                    "id": f"{doc_index}_{chunk_index}",
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_id_in_doc": chunk_index,
                        "start_index": chunk.metadata.get(
                            "start_index",
                            -1,
                        ),
                    },
                }

                all_chunks.append(chunk_dict)

        logging.info(
            "Total de %s chunks créés.",
            len(all_chunks),
        )

        return all_chunks

    def _generate_embeddings(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """Génère localement les embeddings des chunks."""

        if not chunks:
            logging.warning(
                "Aucun chunk fourni pour générer "
                "les embeddings."
            )
            return None

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        logging.info(
            "Génération locale de %s embeddings "
            "(modèle : %s)...",
            len(texts),
            EMBEDDING_MODEL,
        )

        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            embeddings_array = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            logging.info(
                "Embeddings générés avec succès. Shape : %s",
                embeddings_array.shape,
            )

            return embeddings_array

        except Exception as error:
            logging.error(
                "Erreur pendant la génération locale "
                "des embeddings : %s",
                error,
            )
            return None

    def build_index(
        self,
        documents: List[Dict[str, Any]],
    ):
        """Construit un index FAISS à partir des documents."""

        if not documents:
            logging.warning(
                "Aucun document fourni pour construire "
                "l'index."
            )
            return

        # 1. Chunking
        self.document_chunks = (
            self._split_documents_to_chunks(
                documents
            )
        )

        if not self.document_chunks:
            logging.error(
                "Le découpage n'a produit aucun chunk."
            )
            return

        # 2. Embeddings locaux
        embeddings = self._generate_embeddings(
            self.document_chunks
        )

        if (
            embeddings is None
            or embeddings.shape[0]
            != len(self.document_chunks)
        ):
            logging.error(
                "Le nombre d'embeddings ne correspond "
                "pas au nombre de chunks."
            )

            self.document_chunks = []
            self.index = None
            return

        # 3. Index FAISS / similarité cosinus
        dimension = embeddings.shape[1]

        logging.info(
            "Création de l'index FAISS "
            "(dimension=%s)...",
            dimension,
        )

        # Les embeddings sont déjà normalisés par
        # SentenceTransformer.
        # IndexFlatIP équivaut alors à une similarité cosinus.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        logging.info(
            "Index FAISS créé avec %s vecteurs.",
            self.index.ntotal,
        )

        # 4. Sauvegarde
        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Sauvegarde l'index FAISS et les chunks."""

        if (
            self.index is None
            or not self.document_chunks
        ):
            logging.warning(
                "Tentative de sauvegarde d'un index "
                "ou de chunks vides."
            )
            return

        os.makedirs(
            os.path.dirname(FAISS_INDEX_FILE),
            exist_ok=True,
        )

        os.makedirs(
            os.path.dirname(DOCUMENT_CHUNKS_FILE),
            exist_ok=True,
        )

        try:
            logging.info(
                "Sauvegarde de l'index FAISS dans %s...",
                FAISS_INDEX_FILE,
            )

            faiss.write_index(
                self.index,
                FAISS_INDEX_FILE,
            )

            logging.info(
                "Sauvegarde des chunks dans %s...",
                DOCUMENT_CHUNKS_FILE,
            )

            with open(
                DOCUMENT_CHUNKS_FILE,
                "wb",
            ) as file:
                pickle.dump(
                    self.document_chunks,
                    file,
                )

            logging.info(
                "Index et chunks sauvegardés avec succès."
            )

        except Exception as error:
            logging.error(
                "Erreur lors de la sauvegarde "
                "de l'index/chunks : %s",
                error,
            )

    def search(
        self,
        query_text: str,
        k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recherche les chunks les plus pertinents.

        Args:
            query_text:
                Question utilisateur.
            k:
                Nombre maximum de chunks retournés.
            min_score:
                Similarité minimale entre 0 et 1.
        """

        validated_query = RAGQuery(
            question=query_text,
        )

        query_text = validated_query.question


        if (
            self.index is None
            or not self.document_chunks
        ):
            logging.warning(
                "Recherche impossible : "
                "index FAISS absent ou vide."
            )
            return []

        logging.info(
            "Recherche des %s chunks les plus pertinents "
            "pour : '%s'",
            k,
            query_text,
        )

        try:
            # Embedding LOCAL de la question
            query_embedding = (
                self.embedding_model.encode(
                    [query_text],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            )

            query_embedding = np.asarray(
                query_embedding,
                dtype=np.float32,
            )

            search_k = (
                k * 3
                if min_score is not None
                else k
            )

            search_k = min(
                search_k,
                self.index.ntotal,
            )

            scores, indices = self.index.search(
                query_embedding,
                search_k,
            )

            results: List[Dict[str, Any]] = []

            for position, chunk_index in enumerate(
                indices[0]
            ):
                if not (
                    0
                    <= chunk_index
                    < len(self.document_chunks)
                ):
                    continue

                raw_score = float(
                    scores[0][position]
                )

                if (
                    min_score is not None
                    and raw_score < min_score
                ):
                    continue

                chunk = self.document_chunks[
                    chunk_index
                ]

                validated_result = RetrievalResult(
                    score=raw_score * 100,
                    raw_score=raw_score,
                    text=chunk["text"],
                    metadata=chunk["metadata"],
                )

                results.append(
                    validated_result.model_dump()
                )

            results.sort(
                key=lambda item: item["raw_score"],
                reverse=True,
            )

            results = results[:k]

            logging.info(
                "%s chunks pertinents trouvés.",
                len(results),
            )

            return results

        except Exception as error:
            logging.error(
                "Erreur pendant la recherche FAISS : %s",
                error,
            )
            return []
