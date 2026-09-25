import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from sportsee.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_CHUNKS_FILE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
)
from sportsee.rag.chunk_quality_validator import (
    audit_chunks_semantically,
)
from sportsee.rag.schemas import (
    EmbeddingBatch,
    PreparedChunk,
    RAGQuery,
    RetrievalResult,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class VectorStoreManager:
    """
    Gère le cycle de vie du vector store FAISS.

    Responsabilités :
    - charger un index existant ;
    - découper les documents en chunks ;
    - valider structurellement les chunks avec Pydantic ;
    - auditer optionnellement les chunks avec Pydantic AI ;
    - générer et valider les embeddings ;
    - construire et sauvegarder l'index FAISS ;
    - effectuer les recherches sémantiques.
    """

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

            with open(
                DOCUMENT_CHUNKS_FILE,
                "rb",
            ) as file:
                self.document_chunks = pickle.load(file)

            expected_dimension = (
                self.embedding_model
                .get_sentence_embedding_dimension()
            )

            if (
                expected_dimension is None
                or self.index.d != int(expected_dimension)
            ):
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
        """
        Découpe les documents en chunks.

        Chaque chunk est validé structurellement
        avec le modèle Pydantic PreparedChunk.
        """

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
                validated_chunk = PreparedChunk(
                    id=f"{doc_index}_{chunk_index}",
                    text=chunk.page_content.strip(),
                    metadata={
                        **chunk.metadata,
                        "chunk_id_in_doc": chunk_index,
                        "start_index": chunk.metadata.get(
                            "start_index",
                            -1,
                        ),
                    },
                )

                all_chunks.append(
                    validated_chunk.model_dump()
                )

        logging.info(
            "Total de %s chunks créés et validés "
            "structurellement avec Pydantic.",
            len(all_chunks),
        )

        return all_chunks

    def _run_semantic_audit(
        self,
        chunks: List[Dict[str, Any]],
        sample_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Exécute un audit sémantique Pydantic AI.

        L'audit est volontairement effectué sur un échantillon
        afin de limiter la latence et le nombre d'appels au LLM.

        Cet audit évalue la qualité des chunks mais ne les filtre
        pas automatiquement de l'index.
        """

        if sample_size <= 0:
            raise ValueError(
                "semantic_audit_sample_size doit être "
                "strictement supérieur à 0."
            )

        effective_sample_size = min(
            sample_size,
            len(chunks),
        )

        logging.info(
            "Audit sémantique Pydantic AI activé "
            "sur %s/%s chunks.",
            effective_sample_size,
            len(chunks),
        )

        audit_results = audit_chunks_semantically(
            chunks,
            sample_size=effective_sample_size,
        )

        valid_count = 0

        for audit_result in audit_results:
            assessment = audit_result["assessment"]

            if assessment["is_valid"]:
                valid_count += 1

            logging.info(
                "Audit chunk=%s | source=%s | "
                "valid=%s | relevance=%.2f | "
                "readability=%.2f | reason=%s",
                audit_result["chunk_id"],
                audit_result["source"],
                assessment["is_valid"],
                assessment["relevance_score"],
                assessment["readability_score"],
                assessment["reason"],
            )

        logging.info(
            "Audit sémantique terminé : "
            "%s/%s chunks audités considérés valides.",
            valid_count,
            len(audit_results),
        )

        return audit_results

    def _generate_embeddings(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """
        Génère localement les embeddings.

        Le batch obtenu est validé avec Pydantic
        avant sa transmission à FAISS.
        """

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

            expected_dimension = (
                self.embedding_model
                .get_sentence_embedding_dimension()
            )

            if expected_dimension is None:
                raise ValueError(
                    "Impossible de déterminer la dimension "
                    "des embeddings du modèle."
                )

            validated_batch = EmbeddingBatch(
                chunk_ids=[
                    chunk["id"]
                    for chunk in chunks
                ],
                vectors=embeddings_array.tolist(),
                expected_dimension=int(
                    expected_dimension
                ),
            )

            embeddings_array = np.asarray(
                validated_batch.vectors,
                dtype=np.float32,
            )

            logging.info(
                "Embeddings validés avec succès. Shape : %s",
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
        semantic_audit: bool = False,
        semantic_audit_sample_size: int = 5,
    ):
        """
        Construit l'index FAISS à partir des documents.

        Pipeline :
        1. chunking ;
        2. validation structurelle Pydantic ;
        3. audit sémantique Pydantic AI optionnel ;
        4. génération et validation des embeddings ;
        5. construction de l'index FAISS ;
        6. sauvegarde.
        """

        if not documents:
            logging.warning(
                "Aucun document fourni pour construire "
                "l'index."
            )
            return

        # 1. Chunking + validation Pydantic
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

        # 2. Audit sémantique optionnel Pydantic AI
        if semantic_audit:
            self._run_semantic_audit(
                self.document_chunks,
                sample_size=semantic_audit_sample_size,
            )

        # 3. Embeddings locaux + validation Pydantic
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

        # 4. Index FAISS / similarité cosinus
        dimension = embeddings.shape[1]

        logging.info(
            "Création de l'index FAISS "
            "(dimension=%s)...",
            dimension,
        )

        # Les embeddings sont normalisés par SentenceTransformer.
        # IndexFlatIP correspond alors à une similarité cosinus.
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

        # 5. Sauvegarde
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