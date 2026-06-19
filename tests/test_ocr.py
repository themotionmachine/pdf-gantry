"""Tests for OCR processing with Surya."""

import json
from unittest.mock import MagicMock, patch

import pytest

# These tests exercise the OCR path, which needs the [ocr] extra
# (surya-ocr + Pillow). Skip the whole module when those deps are absent
# instead of erroring at collection / runtime.
pytest.importorskip("PIL")
pytest.importorskip("surya")

from pdf_gantry.db import get_connection  # noqa: E402
from pdf_gantry.ingest import ingest_directory  # noqa: E402
from pdf_gantry.ocr import ocr_document, process_ocr_documents  # noqa: E402

# --- Mock helpers ---

_DEFAULT_OCR_LINES = ("Recognized OCR text line one.", "Another line of text.")


def _make_mock_predictions(num_pages, lines=_DEFAULT_OCR_LINES):
    """Build a list of mock prediction objects mimicking Surya output."""
    predictions = []
    for _ in range(num_pages):
        text_lines = []
        for text in lines:
            line = MagicMock()
            line.text = text
            text_lines.append(line)
        page = MagicMock()
        page.text_lines = text_lines
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


# --- Regression tests: OCR must re-derive chunks and not duplicate FTS (issue #16) ---

def _seed_stale_chunk(conn, paper_id):
    """Give a paper a stale chunk + chunk_vec + has_chunk_embeddings=1,
    as if it had been processed (badly) before OCR."""
    cur = conn.execute(
        """INSERT INTO chunks (doc_id, chunk_index, section_header, page_start, text, char_offset)
           VALUES (?, 0, 'STALE', 1, ?, 0)""",
        (paper_id, "STALE GARBAGE pre-ocr extraction noise"),
    )
    stale_chunk_id = cur.lastrowid
    conn.execute(
        "INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
        (stale_chunk_id, json.dumps([0.0] * 768)),
    )
    conn.execute("UPDATE papers SET has_chunk_embeddings = 1 WHERE id = ?", (paper_id,))
    conn.commit()
    return stale_chunk_id


def test_process_ocr_rederives_chunks(tmp_path, papers_dir, mock_surya):
    """After OCR, chunks reflect the OCR text — stale pre-OCR chunks are gone."""
    db_path, conn = _setup_ocr_db(tmp_path, papers_dir)
    paper_id = conn.execute("SELECT id FROM papers LIMIT 1").fetchone()["id"]
    _seed_stale_chunk(conn, paper_id)

    process_ocr_documents(conn, papers_dir, db_path, paper_ids=[paper_id])

    texts = [r["text"] for r in conn.execute(
        "SELECT text FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (paper_id,)
    ).fetchall()]
    assert texts, "OCR should have produced chunks"
    joined = "\n".join(texts)
    assert "Recognized OCR text" in joined
    assert "STALE GARBAGE" not in joined
    conn.close()


def test_process_ocr_clears_stale_chunk_vec(tmp_path, papers_dir, mock_surya):
    """OCR removes embeddings tied to the old chunks (vec0 has no CASCADE)."""
    db_path, conn = _setup_ocr_db(tmp_path, papers_dir)
    paper_id = conn.execute("SELECT id FROM papers LIMIT 1").fetchone()["id"]
    stale_chunk_id = _seed_stale_chunk(conn, paper_id)

    process_ocr_documents(conn, papers_dir, db_path, paper_ids=[paper_id])

    leftover = conn.execute(
        "SELECT COUNT(*) FROM chunk_vec WHERE chunk_id = ?", (stale_chunk_id,)
    ).fetchone()[0]
    assert leftover == 0, "stale chunk embedding should be deleted"
    conn.close()


def test_process_ocr_resets_chunk_embeddings_flag(tmp_path, papers_dir, mock_surya):
    """New chunks have no embeddings yet, so the flag must be reset to 0."""
    db_path, conn = _setup_ocr_db(tmp_path, papers_dir)
    paper_id = conn.execute("SELECT id FROM papers LIMIT 1").fetchone()["id"]
    _seed_stale_chunk(conn, paper_id)

    process_ocr_documents(conn, papers_dir, db_path, paper_ids=[paper_id])

    flag = conn.execute(
        "SELECT has_chunk_embeddings FROM papers WHERE id = ?", (paper_id,)
    ).fetchone()[0]
    assert flag == 0
    conn.close()


def test_process_ocr_fts_replaces_stale_text(tmp_path, papers_dir, mock_surya):
    """Re-OCR with new text must drop the old FTS postings (contentless delete)."""
    db_path, conn = _setup_ocr_db(tmp_path, papers_dir)
    paper_id = conn.execute("SELECT id FROM papers LIMIT 1").fetchone()["id"]

    process_ocr_documents(conn, papers_dir, db_path, paper_ids=[paper_id])

    # Second pass yields different OCR text, as if the page were re-OCR'd better.
    mock_surya["rec"].return_value = _make_mock_predictions(
        1, lines=("Totallynewword content here.",)
    )
    conn.execute("UPDATE papers SET ocr_completed_at = NULL WHERE id = ?", (paper_id,))
    conn.commit()
    process_ocr_documents(conn, papers_dir, db_path, paper_ids=[paper_id])

    def hits(term):
        return [r["rowid"] for r in conn.execute(
            "SELECT rowid FROM papers_fts WHERE papers_fts MATCH ?", (term,)
        ).fetchall()]

    assert paper_id not in hits("Recognized"), "stale OCR text still in FTS"
    assert paper_id in hits("Totallynewword"), "new OCR text missing from FTS"
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


# --- CLI tests: ocr --ids (issue #18) ---

def _setup_cli_ocr_env(tmp_path, papers_dir, monkeypatch, mark_needs_ocr=True):
    """Ingest papers, set CLI env, return (db_path, sorted paper ids)."""
    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    if mark_needs_ocr:
        conn.execute("UPDATE papers SET needs_ocr = 1, is_scanned = 1")
    conn.commit()
    ids = [r["id"] for r in conn.execute("SELECT id FROM papers ORDER BY id").fetchall()]
    conn.close()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(papers_dir))
    return db_path, ids


def test_cli_ocr_ids_targets_specific_paper(tmp_path, papers_dir, mock_surya, monkeypatch):
    """gantry ocr --ids X processes only paper X, not the whole needs_ocr set."""
    from click.testing import CliRunner

    from pdf_gantry.cli import cli

    db_path, ids = _setup_cli_ocr_env(tmp_path, papers_dir, monkeypatch)
    assert len(ids) > 1
    target = ids[0]

    result = CliRunner().invoke(cli, ["ocr", "--ids", str(target), "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total"] == 1
    assert data["succeeded"] == 1

    conn = get_connection(str(db_path))
    done = [r["id"] for r in conn.execute(
        "SELECT id FROM papers WHERE ocr_completed_at IS NOT NULL"
    ).fetchall()]
    conn.close()
    assert done == [target]


def test_cli_ocr_ids_overrides_filter(tmp_path, papers_dir, mock_surya, monkeypatch):
    """--ids re-OCRs a specific paper even when nothing is flagged needs_ocr."""
    from click.testing import CliRunner

    from pdf_gantry.cli import cli

    _db_path, ids = _setup_cli_ocr_env(tmp_path, papers_dir, monkeypatch, mark_needs_ocr=False)
    target = ids[0]

    result = CliRunner().invoke(cli, ["ocr", "--ids", str(target), "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total"] == 1
    assert data["succeeded"] == 1


def test_cli_ocr_ids_invalid(tmp_path, papers_dir, mock_surya, monkeypatch):
    """gantry ocr --ids with non-integers exits 1 with a clear error."""
    from click.testing import CliRunner

    from pdf_gantry.cli import cli

    _setup_cli_ocr_env(tmp_path, papers_dir, monkeypatch)

    result = CliRunner().invoke(cli, ["ocr", "--ids", "abc", "--json"])

    assert result.exit_code == 1
    assert "Invalid --ids" in result.output
