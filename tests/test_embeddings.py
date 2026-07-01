"""Tests for embedding generation and storage."""

import struct

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.embeddings import _prepare_text, _serialize_vector
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents


def test_serialize_vector():
    """Vector serialization produces correct bytes."""
    vec = [1.0, 2.0, 3.0]
    result = _serialize_vector(vec)
    assert isinstance(result, bytes)
    assert len(result) == 3 * 4  # 3 floats * 4 bytes each
    unpacked = struct.unpack("3f", result)
    assert unpacked == (1.0, 2.0, 3.0)


def test_prepare_text_with_all_fields():
    """Text preparation combines title, abstract, and text."""
    result = _prepare_text("My Title", "Abstract here", "Full text content")
    assert "My Title" in result
    assert "Abstract here" in result
    assert "Full text content" in result


def test_prepare_text_truncation():
    """Text is truncated to max_chars."""
    long_text = "x" * 100000
    result = _prepare_text(None, None, long_text, max_chars=1000)
    assert len(result) <= 1000


def test_prepare_text_empty():
    """Empty inputs produce empty string."""
    result = _prepare_text(None, None, None)
    assert result == ""


def test_embed_chunks_overwrites_existing_vectors(tmp_path, papers_dir):
    """Re-embedding must replace existing chunk_vec rows.

    Regression: vec0 virtual tables don't honor `INSERT OR REPLACE` and raise
    `UNIQUE constraint failed` on PK collision. The embed path must DELETE
    before INSERT so re-embeds succeed.

    The _stub_embedding_model session fixture in conftest.py supplies a fast
    stand-in for the real ML model, so this test exercises the DB contract
    (DELETE-before-INSERT) without paying the 11-17 s model-load cost.
    """
    from pdf_gantry.embeddings import embed_chunks

    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    paper_ids = [r["id"] for r in conn.execute("SELECT id FROM papers").fetchall()]
    assert paper_ids, "fixture should produce at least one paper"

    first = embed_chunks(conn, db_path, paper_ids=paper_ids)

    assert first.total > 0, "fixture should produce chunks"
    assert first.failed == 0, f"first embed unexpectedly failed: {first.failed}"

    second = embed_chunks(conn, db_path, paper_ids=paper_ids)
    assert second.failed == 0, (
        f"re-embed produced {second.failed} failures — INSERT OR REPLACE on vec0 "
        "doesn't work; embed_chunks must DELETE before INSERT"
    )
    assert second.succeeded == first.succeeded
    conn.close()


def test_embed_documents_overwrites_existing_vectors(tmp_path, papers_dir):
    """Re-embedding papers must replace existing paper_embeddings rows.

    Same vec0 `INSERT OR REPLACE` regression as the chunk path. Uses the
    _stub_embedding_model session fixture (conftest.py) to avoid loading the
    real ML model — tests the DELETE-before-INSERT DB contract, not vector quality.
    """
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    from pdf_gantry.embeddings import embed_documents

    first = embed_documents(conn, db_path)

    assert first.total > 0
    assert first.failed == 0, f"first embed unexpectedly failed: {first.failed}"

    # Force re-embed by clearing the flag (rows in paper_embeddings remain).
    conn.execute("UPDATE papers SET has_embeddings = 0")
    conn.commit()

    second = embed_documents(conn, db_path)
    assert second.failed == 0, (
        f"re-embed produced {second.failed} failures — INSERT OR REPLACE on vec0 "
        "doesn't work; embed_documents must DELETE before INSERT"
    )
    assert second.succeeded == first.succeeded
    conn.close()


def test_embed_documents_propagates_import_error(tmp_path, monkeypatch):
    """embed_documents surfaces ImportError when the embedding model is unavailable.

    Replaces test_embed_documents_requires_sentence_transformers, which was a
    false-green: it passed trivially whether sentence-transformers was installed
    or not (bare try/except ImportError: pass; assert total >= 0 is always true)
    while paying the full 11-second model-load cost on every run.

    This test uses function-scoped monkeypatch to temporarily shadow the
    session-level _stub_embedding_model fixture with a callable that raises
    ImportError, then asserts that embed_documents propagates the error faithfully.
    Cost: < 100 ms.
    """
    import pdf_gantry.embeddings as emb_module

    def _model_missing(_model_name):
        raise ImportError("sentence-transformers not installed.")

    monkeypatch.setattr(emb_module, "_get_embedding_model", _model_missing)

    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    conn.execute(
        """INSERT INTO papers
            (id, path, filename, file_hash, file_size, file_modified,
             page_count, has_text, has_embeddings, indexed_at, updated_at)
           VALUES (1, 'p1.pdf', 'p1.pdf', 'h1', 1, '2026-01-01', 1, 1, 0,
                   '2026-01-01', '2026-01-01')"""
    )
    conn.execute(
        "INSERT INTO paper_text (paper_id, raw_text, markdown, text_length, markdown_length)"
        " VALUES (1, 'some text', '', 9, 0)"
    )
    conn.commit()

    with pytest.raises(ImportError, match="sentence-transformers"):
        emb_module.embed_documents(conn, db_path)
    conn.close()
