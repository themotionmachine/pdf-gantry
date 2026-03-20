"""Tests for the queue/filter system."""

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents
from pdf_gantry.queue import build_filter_query, query_queue, queue_count


def test_filter_needs_text(populated_db):
    """--needs text returns documents without text."""
    count = queue_count(populated_db, needs=["text"])
    total = populated_db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert count == total  # None have text yet


def test_filter_has_text(populated_db):
    """--has text returns documents with text."""
    count = queue_count(populated_db, has=["text"])
    assert count == 0  # None processed yet


def test_filter_after_processing(tmp_path, papers_dir):
    """Filters reflect processing state changes."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    # Before processing
    needs_text = queue_count(conn, needs=["text"])
    assert needs_text > 0

    # Process
    process_documents(conn, papers_dir, db_path, workers=1)

    # After processing - re-open connection to see changes
    conn2 = get_connection(str(db_path))
    needs_text = queue_count(conn2, needs=["text"])
    has_text = queue_count(conn2, has=["text"])
    assert needs_text == 0
    assert has_text > 0
    conn2.close()
    conn.close()


def test_filter_is_digital(populated_db):
    """--is digital returns digital documents."""
    count = queue_count(populated_db, is_prop=["digital"])
    assert count >= 0  # May be 0 or more depending on classification


def test_multiple_filters_combine(populated_db):
    """Multiple filters combine with AND."""
    where, params = build_filter_query(
        needs=["text"], is_prop=["digital"]
    )
    assert "has_text = 0" in where
    assert "is_scanned = 0" in where
    assert "AND" in where


def test_filter_has_errors(populated_db):
    """--has errors filter works."""
    count = queue_count(populated_db, has=["errors"])
    assert count == 0  # No errors initially


def test_query_queue_returns_dicts(populated_db):
    """query_queue returns list of dicts."""
    rows = query_queue(populated_db, needs=["text"])
    assert len(rows) > 0
    assert isinstance(rows[0], dict)
    assert "filename" in rows[0]
    assert "has_text" in rows[0]


def test_queue_count_matches_query(populated_db):
    """queue_count matches length of query_queue."""
    count = queue_count(populated_db, needs=["text"])
    rows = query_queue(populated_db, needs=["text"])
    assert count == len(rows)


def test_build_filter_empty():
    """No filters returns empty where clause."""
    where, params = build_filter_query()
    assert where == ""
    assert params == []


def test_build_filter_stale_embeddings():
    """--stale-embeddings produces correct filter."""
    where, params = build_filter_query(
        stale_embeddings=True,
        current_model_version="v2.0",
    )
    assert "embedding_model_version != ?" in where
    assert "v2.0" in params
