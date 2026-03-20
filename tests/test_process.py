"""Tests for text extraction and processing."""

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import extract_text_pymupdf, process_documents


def test_extract_text_from_digital_pdf(sample_pdf):
    """Text extraction produces non-empty results."""
    raw_text, markdown = extract_text_pymupdf(sample_pdf)
    assert "climate change" in raw_text.lower()
    assert len(raw_text) > 0
    assert len(markdown) > 0


def test_extract_markdown_from_digital_pdf(sample_pdf):
    """Markdown extraction produces output."""
    raw_text, markdown = extract_text_pymupdf(sample_pdf)
    assert len(markdown) > 0


def test_process_sets_flags(tmp_path, papers_dir):
    """Processing sets has_text and has_markdown flags."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)

    stats = process_documents(
        conn, papers_dir, db_path,
        workers=1,
    )

    assert stats.succeeded > 0

    # Re-read from DB to check flags
    conn2 = get_connection(str(db_path))
    rows = conn2.execute(
        "SELECT has_text, has_markdown, text_method FROM papers WHERE has_text = 1"
    ).fetchall()
    assert len(rows) > 0
    for row in rows:
        assert row["has_text"] == 1
        assert row["has_markdown"] == 1
        assert row["text_method"] == "pymupdf4llm"
    conn2.close()
    conn.close()


def test_fts_populated_after_processing(tmp_path, papers_dir):
    """FTS5 index has content after processing."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    # Search FTS
    conn2 = get_connection(str(db_path))
    rows = conn2.execute(
        "SELECT * FROM papers_fts WHERE papers_fts MATCH 'climate'"
    ).fetchall()
    assert len(rows) >= 1
    conn2.close()
    conn.close()


def test_process_with_limit(tmp_path, papers_dir):
    """Processing respects the limit parameter."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)

    stats = process_documents(
        conn, papers_dir, db_path,
        workers=1, limit=1,
    )

    assert stats.total == 1
    assert stats.succeeded == 1
    conn.close()


def test_paper_text_stored(tmp_path, papers_dir):
    """Raw text and markdown are stored in paper_text table."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    conn2 = get_connection(str(db_path))
    rows = conn2.execute(
        "SELECT raw_text, markdown, text_length, markdown_length FROM paper_text"
    ).fetchall()
    assert len(rows) > 0
    for row in rows:
        assert row["raw_text"] is not None
        assert row["markdown"] is not None
        assert row["text_length"] > 0
        assert row["markdown_length"] > 0
    conn2.close()
    conn.close()
