"""Tests for the pipeline (sync) command."""


import shutil

from pdf_gantry.db import get_connection
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


def test_pipeline_full_sweep_skips_encrypted_paper(tmp_path, papers_dir, encrypted_pdf):
    """The default full-corpus sweep (no filename given) ingests an encrypted
    PDF but doesn't hand it to process_documents — it's flagged is_encrypted
    at ingest time and there's nothing process/extract can do with it.

    Guards the copy of the default-selection query duplicated in
    pipeline.py's full-sweep branch (separate from process.py's own copy).
    """
    shutil.copy(encrypted_pdf, papers_dir / "locked.pdf")

    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    stats = run_pipeline(conn, papers_dir, db_path, workers=1)

    # 3 files ingested (2 normal + 1 locked); only the 2 normal ones processed.
    assert stats["ingested"] == 3
    assert stats["processed"] == 2

    locked_row = conn.execute(
        "SELECT has_text, error_count FROM papers WHERE filename = 'locked.pdf'"
    ).fetchone()
    assert locked_row["has_text"] == 0
    assert locked_row["error_count"] == 0, (
        "encrypted paper should be skipped up front, not attempted and failed"
    )
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


# --- pipeline warns about suspicious extractions (issue #17) ---

def _seed_suspicious(db_path):
    conn = get_connection(str(db_path))
    conn.execute(
        """INSERT INTO papers
            (id, path, filename, file_hash, file_size, file_modified,
             page_count, has_text, needs_ocr, indexed_at, updated_at)
           VALUES (1, 'p1.pdf', 'p1.pdf', 'h1', 1, '2026-01-01', 10, 1, 1,
                   '2026-01-01', '2026-01-01')"""
    )
    conn.execute(
        "INSERT INTO paper_text (paper_id, raw_text, markdown, text_length, markdown_length) "
        "VALUES (1, ?, '', 1000, 0)",
        ("x" * 1000,),
    )
    conn.commit()
    conn.close()


def test_pipeline_json_includes_suspicious_count(tmp_path, monkeypatch):
    """pipeline --json surfaces a suspicious-extraction count from the index."""
    import json

    from click.testing import CliRunner

    from pdf_gantry.cli import cli

    empty_papers = tmp_path / "papers"
    empty_papers.mkdir()
    db_path = tmp_path / "index.db"
    get_connection(str(db_path)).close()
    _seed_suspicious(db_path)

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(empty_papers))

    result = CliRunner().invoke(cli, ["pipeline", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["suspicious"] == 1


def test_pipeline_text_warns_about_suspicious(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from pdf_gantry.cli import cli

    empty_papers = tmp_path / "papers"
    empty_papers.mkdir()
    db_path = tmp_path / "index.db"
    get_connection(str(db_path)).close()
    _seed_suspicious(db_path)

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(empty_papers))

    result = CliRunner().invoke(cli, ["pipeline"])

    assert result.exit_code == 0
    assert "suspicious" in result.output.lower()
