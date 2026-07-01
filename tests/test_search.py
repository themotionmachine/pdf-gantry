"""Tests for full-text search."""

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents
from pdf_gantry.search import fts_search, search_count


@pytest.fixture
def searchable_db(tmp_path, papers_dir):
    """A database with papers ingested and processed (FTS populated)."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)
    # Re-open to get fresh connection
    conn.close()
    conn = get_connection(str(db_path))
    return conn


def test_fts_search_returns_results(searchable_db):
    """FTS5 search returns results for known content."""
    results = fts_search(searchable_db, "climate")
    assert len(results) >= 1
    assert results[0].filename == "test_climate.pdf"


def test_fts_search_ranking(searchable_db):
    """Results are ranked by relevance score."""
    results = fts_search(searchable_db, "climate OR neural")
    assert len(results) >= 1
    # Scores should be positive
    for r in results:
        assert r.score > 0


def test_fts_search_empty_results(searchable_db):
    """Search for nonexistent term returns empty list."""
    results = fts_search(searchable_db, "xyznonexistent")
    assert results == []


def test_fts_search_snippet(searchable_db):
    """Search results include snippets."""
    results = fts_search(searchable_db, "climate")
    assert len(results) >= 1
    # Snippet should contain something
    assert len(results[0].snippet) > 0


def test_search_count(searchable_db):
    """search_count returns total matches."""
    count = search_count(searchable_db, "climate")
    assert count >= 1


def test_search_count_zero(searchable_db):
    """search_count returns 0 for no matches."""
    count = search_count(searchable_db, "xyznonexistent")
    assert count == 0


def test_fts_search_limit(searchable_db):
    """Search respects the limit parameter."""
    results = fts_search(searchable_db, "climate OR neural OR machine", limit=1)
    assert len(results) <= 1


def test_fts_phrase_query(searchable_db):
    """Phrase queries work with FTS5."""
    results = fts_search(searchable_db, '"climate change"')
    assert len(results) >= 1


def test_fts_hyphenated_query(searchable_db):
    """Hyphenated terms don't crash FTS5."""
    # Should not raise "no such column" error
    results = fts_search(searchable_db, "cross-national")
    # May or may not find results, but should not error
    assert isinstance(results, list)


def test_sanitize_fts_query():
    """Hyphen sanitization works correctly."""
    from pdf_gantry.search import _sanitize_fts_query
    assert _sanitize_fts_query("cross-national") == "cross national"
    assert _sanitize_fts_query('"self-regulation"') == '"self regulation"'
    assert _sanitize_fts_query("climate AND cross-border") == "climate AND cross border"
    # Preserve non-hyphen content
    assert _sanitize_fts_query("climate change") == "climate change"
