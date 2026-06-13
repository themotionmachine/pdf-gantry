"""Tests for status + queue surfacing of suspicious extractions (issue #17)."""

import json

from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection


def _build_db(tmp_path, monkeypatch):
    """Index DB with one suspicious paper among healthy/ineligible ones."""
    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))

    def add(paper_id, *, page_count, needs_ocr, has_text, text_length):
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
                "INSERT INTO paper_text "
                "(paper_id, raw_text, markdown, text_length, markdown_length) "
                "VALUES (?, ?, '', ?, 0)",
                (paper_id, "x" * text_length, text_length),
            )

    add(1, page_count=10, needs_ocr=1, has_text=1, text_length=1000)   # suspicious
    add(2, page_count=2, needs_ocr=1, has_text=1, text_length=20000)   # rich
    add(3, page_count=10, needs_ocr=0, has_text=1, text_length=1000)   # good extraction
    conn.commit()
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))
    return db_path


def test_status_json_reports_suspicious_count(tmp_path, monkeypatch):
    _build_db(tmp_path, monkeypatch)

    result = CliRunner().invoke(cli, ["status", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["suspicious_extraction"] == 1


def test_status_text_shows_suspicious_line(tmp_path, monkeypatch):
    _build_db(tmp_path, monkeypatch)

    result = CliRunner().invoke(cli, ["status"])

    assert result.exit_code == 0
    assert "Suspicious" in result.output


def test_queue_is_suspicious_cli(tmp_path, monkeypatch):
    """--is suspicious is an accepted choice and returns the bitmap paper."""
    _build_db(tmp_path, monkeypatch)

    result = CliRunner().invoke(cli, ["queue", "--is", "suspicious", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["count"] == 1
    assert data["documents"][0]["id"] == 1
