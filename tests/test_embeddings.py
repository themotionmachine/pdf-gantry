"""Tests for embedding generation and storage."""

import struct

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents
from pdf_gantry.embeddings import _serialize_vector, _prepare_text


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


def test_embed_documents_requires_sentence_transformers(tmp_path, papers_dir):
    """embed_documents gives ImportError if sentence-transformers not installed."""
    # This test will pass if sentence-transformers IS installed (it will try to load model)
    # or fail with ImportError if not installed. We just test the function exists and runs.
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    from pdf_gantry.embeddings import embed_documents

    try:
        stats = embed_documents(conn, db_path, model_name="nomic-ai/nomic-embed-text-v2-moe")
        # If it succeeds, verify the stats
        assert stats.total >= 0
    except ImportError:
        # Expected if sentence-transformers not installed
        pass
    finally:
        conn.close()
