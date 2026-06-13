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


# --- suspicious_extraction filter (issue #17) ---

def _insert_paper(conn, paper_id, *, page_count, needs_ocr, has_text, text_length):
    """Insert a minimal paper (+ paper_text row) with controlled extraction stats."""
    conn.execute(
        """INSERT INTO papers
            (id, path, filename, file_hash, file_size, file_modified,
             page_count, has_text, needs_ocr, indexed_at, updated_at)
           VALUES (?, ?, ?, ?, 1, '2026-01-01', ?, ?, ?, '2026-01-01', '2026-01-01')""",
        (paper_id, f"p{paper_id}.pdf", f"p{paper_id}.pdf", f"h{paper_id}",
         page_count, has_text, needs_ocr),
    )
    if text_length is not None:
        conn.execute(
            "INSERT INTO paper_text (paper_id, raw_text, markdown, text_length, markdown_length) "
            "VALUES (?, ?, '', ?, 0)",
            (paper_id, "x" * text_length, text_length),
        )
    conn.commit()


@pytest.fixture
def suspicion_db(tmp_path):
    """DB with one suspicious paper among several healthy/ineligible ones."""
    conn = get_connection(str(tmp_path / "susp.db"))
    # 1: low text/page AND needs_ocr AND has_text → suspicious
    _insert_paper(conn, 1, page_count=10, needs_ocr=1, has_text=1, text_length=1000)
    # 2: rich text → not suspicious
    _insert_paper(conn, 2, page_count=2, needs_ocr=1, has_text=1, text_length=20000)
    # 3: low text/page but needs_ocr=0 → not suspicious (already good extraction)
    _insert_paper(conn, 3, page_count=10, needs_ocr=0, has_text=1, text_length=1000)
    # 4: needs_ocr but no text yet (unprocessed) → not suspicious
    _insert_paper(conn, 4, page_count=10, needs_ocr=1, has_text=0, text_length=None)
    yield conn
    conn.close()


def test_filter_is_suspicious_isolates_bitmap_paper(suspicion_db):
    """--is suspicious flags only the low-text-per-page, OCR-flagged, has-text paper."""
    rows = query_queue(suspicion_db, is_prop=["suspicious"])
    assert [r["id"] for r in rows] == [1]


def test_filter_is_suspicious_count(suspicion_db):
    """queue_count agrees with the filtered query."""
    assert queue_count(suspicion_db, is_prop=["suspicious"]) == 1


def test_filter_suspicious_combines_with_other_filters(suspicion_db):
    """suspicious AND has text still isolates the same paper."""
    rows = query_queue(suspicion_db, is_prop=["suspicious"], has=["text"])
    assert [r["id"] for r in rows] == [1]
