import numpy as np
import pytest

from sportsee.rag.vector_store import VectorStoreManager


def test_build_index_runs_semantic_audit_when_enabled(
    monkeypatch,
):
    manager = VectorStoreManager.__new__(
        VectorStoreManager
    )
    manager.index = None
    manager.document_chunks = []

    chunks = [
        {
            "id": "0_0",
            "text": "Chunk NBA de test.",
            "metadata": {
                "source": "test.pdf",
            },
        }
    ]

    embeddings = np.asarray(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    audit_calls = []
    save_calls = []

    monkeypatch.setattr(
        manager,
        "_split_documents_to_chunks",
        lambda documents: chunks,
    )

    def fake_semantic_audit(
        audited_chunks,
        sample_size,
    ):
        audit_calls.append(
            (audited_chunks, sample_size)
        )
        return []

    monkeypatch.setattr(
        manager,
        "_run_semantic_audit",
        fake_semantic_audit,
    )

    monkeypatch.setattr(
        manager,
        "_generate_embeddings",
        lambda current_chunks: embeddings,
    )

    monkeypatch.setattr(
        manager,
        "_save_index_and_chunks",
        lambda: save_calls.append(True),
    )

    manager.build_index(
        [
            {
                "page_content": "Document NBA.",
                "metadata": {
                    "source": "test.pdf",
                },
            }
        ],
        semantic_audit=True,
        semantic_audit_sample_size=3,
    )

    assert len(audit_calls) == 1
    assert audit_calls[0][0] == chunks
    assert audit_calls[0][1] == 3
    assert manager.index is not None
    assert manager.index.ntotal == 1
    assert save_calls == [True]


def test_build_index_skips_semantic_audit_by_default(
    monkeypatch,
):
    manager = VectorStoreManager.__new__(
        VectorStoreManager
    )
    manager.index = None
    manager.document_chunks = []

    chunks = [
        {
            "id": "0_0",
            "text": "Chunk NBA de test.",
            "metadata": {
                "source": "test.pdf",
            },
        }
    ]

    embeddings = np.asarray(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        manager,
        "_split_documents_to_chunks",
        lambda documents: chunks,
    )

    def forbidden_audit(*args, **kwargs):
        pytest.fail(
            "L'audit sémantique ne doit pas "
            "être exécuté par défaut."
        )

    monkeypatch.setattr(
        manager,
        "_run_semantic_audit",
        forbidden_audit,
    )

    monkeypatch.setattr(
        manager,
        "_generate_embeddings",
        lambda current_chunks: embeddings,
    )

    monkeypatch.setattr(
        manager,
        "_save_index_and_chunks",
        lambda: None,
    )

    manager.build_index(
        [
            {
                "page_content": "Document NBA.",
                "metadata": {
                    "source": "test.pdf",
                },
            }
        ]
    )

    assert manager.index is not None
    assert manager.index.ntotal == 1
