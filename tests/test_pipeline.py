"""Tests for the pipeline (sync) command."""

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.pipeline import run_pipeline


def test_pipeline_ingests_and_processes(tmp_path, papers_dir):
    """Pipeline ingests new files, processes them, and generates chunks."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    stats = run_pipeline(conn, papers_dir, db_path, workers=1)

    assert stats["ingested"] == 2
    assert stats["processed"] == 2
    # Chunks should be generated
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert chunk_count > 0
    conn.close()


def test_pipeline_idempotent(tmp_path, papers_dir):
    """Running pipeline twice is a no-op the second time."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    run_pipeline(conn, papers_dir, db_path, workers=1)
    conn.close()

    conn2 = get_connection(str(db_path))
    stats = run_pipeline(conn2, papers_dir, db_path, workers=1)

    assert stats["ingested"] == 0
    assert stats["processed"] == 0
    conn2.close()


def test_pipeline_limit(tmp_path, papers_dir):
    """Limit caps processing throughput (ingest scans all, process/embed capped)."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    stats = run_pipeline(conn, papers_dir, db_path, workers=1, limit=1)

    # Ingest scans everything, but only limit papers are processed
    assert stats["ingested"] == 2  # Both files discovered
    assert stats["processed"] <= 1  # Only 1 processed
    conn.close()


def test_pipeline_dry_run(tmp_path, papers_dir):
    """Dry run reports counts without modifying the database."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    stats = run_pipeline(conn, papers_dir, db_path, workers=1, dry_run=True)

    assert stats["ingested"] >= 0
    # DB should be empty
    row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
    assert row[0] == 0
    conn.close()


def test_pipeline_single_file(tmp_path, papers_dir):
    """Pipeline with filename processes only that file."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    stats = run_pipeline(
        conn, papers_dir, db_path, workers=1,
        filename="test_climate.pdf",
    )

    assert stats["processed"] == 1
    # Only the target file has text extracted
    row = conn.execute("SELECT COUNT(*) FROM papers WHERE has_text = 1").fetchone()
    assert row[0] == 1
    row = conn.execute("SELECT filename FROM papers WHERE has_text = 1").fetchone()
    assert row["filename"] == "test_climate.pdf"
    conn.close()
