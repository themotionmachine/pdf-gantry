"""Tests for pruning ghost entries from the database."""

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents
from pdf_gantry.prune import prune_missing


def test_prune_removes_missing_files(tmp_path, papers_dir):
    """Prune removes DB entries for files no longer on disk."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    # Delete a file from disk
    (papers_dir / "ml_nlp_paper.pdf").unlink()

    stats = prune_missing(conn, papers_dir)
    assert stats["pruned"] == 1
    assert stats["remaining"] > 0

    # Verify it's gone from DB
    row = conn.execute("SELECT COUNT(*) FROM papers WHERE filename = 'ml_nlp_paper.pdf'").fetchone()
    assert row[0] == 0
    conn.close()


def test_prune_noop_when_all_present(tmp_path, papers_dir):
    """Prune is a no-op when all files exist."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    stats = prune_missing(conn, papers_dir)
    assert stats["pruned"] == 0
    assert stats["remaining"] == 2
    conn.close()


def test_prune_dry_run(tmp_path, papers_dir):
    """Dry run reports but doesn't delete."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    (papers_dir / "ml_nlp_paper.pdf").unlink()

    stats = prune_missing(conn, papers_dir, dry_run=True)
    assert stats["pruned"] == 1

    # File should still be in DB
    row = conn.execute("SELECT COUNT(*) FROM papers WHERE filename = 'ml_nlp_paper.pdf'").fetchone()
    assert row[0] == 1
    conn.close()


def test_prune_cleans_related_tables(tmp_path, papers_dir):
    """Prune removes chunks, text, FTS, and vec entries for missing files."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    # Get the paper ID before deleting
    conn2 = get_connection(str(db_path))
    paper = conn2.execute("SELECT id FROM papers WHERE filename = 'ml_nlp_paper.pdf'").fetchone()
    paper_id = paper["id"]

    # Verify related data exists
    assert conn2.execute("SELECT COUNT(*) FROM paper_text WHERE paper_id = ?", (paper_id,)).fetchone()[0] > 0
    assert conn2.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (paper_id,)).fetchone()[0] > 0

    # Delete file and prune
    (papers_dir / "ml_nlp_paper.pdf").unlink()
    prune_missing(conn2, papers_dir)

    # All related data should be gone
    assert conn2.execute("SELECT COUNT(*) FROM paper_text WHERE paper_id = ?", (paper_id,)).fetchone()[0] == 0
    assert conn2.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (paper_id,)).fetchone()[0] == 0
    conn2.close()
    conn.close()


def test_prune_returns_filenames(tmp_path, papers_dir):
    """Prune returns the list of pruned filenames."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    (papers_dir / "ml_nlp_paper.pdf").unlink()

    stats = prune_missing(conn, papers_dir)
    assert "pruned_files" in stats
    assert "ml_nlp_paper.pdf" in stats["pruned_files"]
    conn.close()
