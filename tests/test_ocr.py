"""Tests for OCR processing with Surya."""

from unittest.mock import MagicMock, patch

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.ocr import ocr_document, process_ocr_documents


# --- Mock helpers ---

def _make_mock_predictions(num_pages):
    """Build a list of mock prediction objects mimicking Surya output."""
    predictions = []
    for _ in range(num_pages):
        line1 = MagicMock()
        line1.text = "Recognized OCR text line one."
        line2 = MagicMock()
        line2.text = "Another line of text."
        page = MagicMock()
        page.text_lines = [line1, line2]
        predictions.append(page)
    return predictions


@pytest.fixture
def mock_surya():
    """Patch Surya predictors so tests don't need the real models."""
    import pdf_gantry.ocr as ocr_mod

    mock_rec = MagicMock()
    mock_det = MagicMock()

    mock_rec.return_value = _make_mock_predictions(1)

    ocr_mod._recognition_predictor = None
    ocr_mod._detection_predictor = None
    ocr_mod._foundation_predictor = None

    with patch.object(ocr_mod, "_check_surya_available", return_value=True), \
         patch.object(ocr_mod, "_load_predictors", return_value=(mock_rec, mock_det)):
        yield {"rec": mock_rec, "det": mock_det}


def _setup_ocr_db(tmp_path, papers_dir):
    """Create a DB with papers marked as needing OCR."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    conn.execute("UPDATE papers SET needs_ocr = 1, is_scanned = 1")
    conn.commit()
    return db_path, conn


# --- Unit tests for ocr_document ---

def test_ocr_document_returns_text_and_markdown(sample_pdf, mock_surya):
    """ocr_document returns (raw_text, markdown) tuple."""
    raw_text, markdown = ocr_document(sample_pdf)
    assert len(raw_text) > 0
    assert len(markdown) > 0


def test_ocr_document_markdown_has_page_headers(sample_pdf, mock_surya):
    """Markdown output includes page headers."""
    _, markdown = ocr_document(sample_pdf)
    assert "## Page 1" in markdown


def test_ocr_document_calls_recognition_predictor(sample_pdf, mock_surya):
    """ocr_document invokes the recognition predictor with detection predictor."""
    ocr_document(sample_pdf)
    mock_surya["rec"].assert_called_once()
    call_kwargs = mock_surya["rec"].call_args
    assert call_kwargs.kwargs.get("det_predictor") is not None


def test_ocr_document_raises_when_surya_missing(sample_pdf):
    """Clear error when surya-ocr isn't installed."""
    with patch("pdf_gantry.ocr._check_surya_available", return_value=False):
        with pytest.raises(ImportError, match="[Ss]urya"):
            ocr_document(sample_pdf)


# --- Unit tests for process_ocr_documents ---

def test_process_ocr_sets_db_flags(tmp_path, papers_dir, mock_surya):
    """Processing sets has_text, has_markdown, ocr_method='surya'."""
    db_path, conn = _setup_ocr_db(tmp_path, papers_dir)

    stats = process_ocr_documents(conn, papers_dir, db_path)

    assert stats.succeeded > 0
    assert stats.failed == 0

    rows = conn.execute(
        "SELECT has_text, has_markdown, ocr_method, text_method, needs_ocr "
        "FROM papers WHERE ocr_completed_at IS NOT NULL"
    ).fetchall()
    assert len(rows) > 0
    for row in rows:
        assert row["has_text"] == 1
        assert row["has_markdown"] == 1
        assert row["ocr_method"] == "surya"
        assert row["text_method"] == "surya"
        assert row["needs_ocr"] == 0
    conn.close()


def test_process_ocr_respects_limit(tmp_path, papers_dir, mock_surya):
    """Limit restricts number of documents processed."""
    db_path, conn = _setup_ocr_db(tmp_path, papers_dir)

    stats = process_ocr_documents(conn, papers_dir, db_path, limit=1)

    assert stats.total == 1
    assert stats.succeeded == 1
    conn.close()


def test_process_ocr_dry_run_changes_nothing(tmp_path, papers_dir, mock_surya):
    """Dry run reports count but doesn't modify the database."""
    db_path, conn = _setup_ocr_db(tmp_path, papers_dir)

    stats = process_ocr_documents(conn, papers_dir, db_path, dry_run=True)

    assert stats.total > 0
    assert stats.succeeded == 0
    assert stats.failed == 0

    row = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE ocr_completed_at IS NOT NULL"
    ).fetchone()
    assert row[0] == 0
    conn.close()


def test_process_ocr_dry_run_with_limit(tmp_path, papers_dir, mock_surya):
    """Dry run with limit reports the limited count."""
    db_path, conn = _setup_ocr_db(tmp_path, papers_dir)

    stats = process_ocr_documents(conn, papers_dir, db_path, dry_run=True, limit=1)

    assert stats.total == 1
    conn.close()


def test_process_ocr_populates_fts(tmp_path, papers_dir, mock_surya):
    """OCR text is inserted into the FTS index."""
    db_path, conn = _setup_ocr_db(tmp_path, papers_dir)

    process_ocr_documents(conn, papers_dir, db_path)

    rows = conn.execute(
        "SELECT * FROM papers_fts WHERE papers_fts MATCH 'Recognized'"
    ).fetchall()
    assert len(rows) > 0
    conn.close()


# --- CLI tests ---

def _invoke_ocr(runner, tmp_path, papers_dir, extra_args, monkeypatch):
    """Helper to invoke gantry ocr with correct env config."""
    from pdf_gantry.cli import cli

    # CLI expects db at index_dir/index.db
    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    conn.execute("UPDATE papers SET needs_ocr = 1, is_scanned = 1")
    conn.commit()
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(papers_dir))

    return runner.invoke(cli, ["ocr"] + extra_args)


def test_cli_ocr_dry_run(tmp_path, papers_dir, mock_surya, monkeypatch):
    """gantry ocr --dry-run reports count without processing."""
    from click.testing import CliRunner

    result = _invoke_ocr(CliRunner(), tmp_path, papers_dir, ["--dry-run"], monkeypatch)

    assert result.exit_code == 0
    assert "Would" in result.output or "would" in result.output


def test_cli_ocr_dry_run_json(tmp_path, papers_dir, mock_surya, monkeypatch):
    """gantry ocr --dry-run --json outputs structured JSON."""
    import json
    from click.testing import CliRunner

    result = _invoke_ocr(CliRunner(), tmp_path, papers_dir,
                         ["--dry-run", "--json"], monkeypatch)

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "would_process" in data


def test_cli_ocr_with_limit(tmp_path, papers_dir, mock_surya, monkeypatch):
    """gantry ocr --limit processes only N documents."""
    import json
    from click.testing import CliRunner

    result = _invoke_ocr(CliRunner(), tmp_path, papers_dir,
                         ["--limit", "1", "--json"], monkeypatch)

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total"] == 1
    assert data["succeeded"] == 1


def test_cli_ocr_nothing_to_process(tmp_path, papers_dir, monkeypatch):
    """gantry ocr with no scanned docs exits with code 2."""
    from click.testing import CliRunner
    from pdf_gantry.cli import cli

    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(papers_dir))

    result = CliRunner().invoke(cli, ["ocr"])

    assert result.exit_code == 2
