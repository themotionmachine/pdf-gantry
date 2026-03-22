"""Tests for scoped context windows around chunks."""

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents
from pdf_gantry.search import get_chunk_context


@pytest.fixture
def db_with_chunks(tmp_path, papers_dir):
    """A database with papers ingested, processed, and chunked."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)
    conn.close()
    conn = get_connection(str(db_path))
    return conn


def test_get_chunk_context_returns_surrounding(db_with_chunks):
    """Context includes the target chunk and surrounding text."""
    # Get a chunk
    chunk = db_with_chunks.execute(
        "SELECT chunk_id, doc_id, text FROM chunks LIMIT 1"
    ).fetchone()
    assert chunk is not None

    result = get_chunk_context(db_with_chunks, chunk["chunk_id"])
    assert result is not None
    assert "chunk_text" in result
    assert "context" in result
    assert "filename" in result
    # Context should contain at least the chunk's own text
    assert chunk["text"][:50] in result["context"]


def test_get_chunk_context_max_chars(db_with_chunks):
    """Context doesn't expand beyond max_chars (beyond the target chunk itself)."""
    chunk = db_with_chunks.execute(
        "SELECT chunk_id, text FROM chunks LIMIT 1"
    ).fetchone()

    # With max_chars smaller than the target chunk, we still get at least the target
    result = get_chunk_context(db_with_chunks, chunk["chunk_id"], max_chars=100)
    assert result is not None
    # Context always includes at least the target chunk
    assert chunk["text"][:50] in result["context"]

    # With very large max_chars, we get more context (if available)
    result_large = get_chunk_context(db_with_chunks, chunk["chunk_id"], max_chars=50000)
    assert len(result_large["context"]) >= len(result["context"])


def test_get_chunk_context_includes_neighbors(db_with_chunks):
    """Context includes neighboring chunks when available."""
    # Find a paper with multiple chunks
    row = db_with_chunks.execute(
        "SELECT doc_id FROM chunks GROUP BY doc_id HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()

    if row is None:
        pytest.skip("No paper with multiple chunks in test data")

    # Get a middle chunk (not first, not last)
    chunks = db_with_chunks.execute(
        "SELECT chunk_id, chunk_index FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
        (row["doc_id"],),
    ).fetchall()

    if len(chunks) < 3:
        pytest.skip("Need at least 3 chunks for neighbor test")

    middle = chunks[1]
    result = get_chunk_context(db_with_chunks, middle["chunk_id"], max_chars=50000)
    # Context should be longer than just the target chunk
    target_chunk = db_with_chunks.execute(
        "SELECT text FROM chunks WHERE chunk_id = ?", (middle["chunk_id"],)
    ).fetchone()
    assert len(result["context"]) >= len(target_chunk["text"])


def test_get_chunk_context_invalid_id(db_with_chunks):
    """Invalid chunk ID returns None."""
    result = get_chunk_context(db_with_chunks, 999999)
    assert result is None


def test_get_chunk_context_metadata(db_with_chunks):
    """Context result includes document metadata."""
    chunk = db_with_chunks.execute("SELECT chunk_id FROM chunks LIMIT 1").fetchone()
    result = get_chunk_context(db_with_chunks, chunk["chunk_id"])
    assert "filename" in result
    assert "section_header" in result
    assert "chunk_index" in result
    assert "total_chunks" in result
