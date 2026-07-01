"""Tests for PDF ingestion."""


import fitz

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import classify_document, ingest_directory


def test_ingest_new_pdfs(tmp_path, papers_dir):
    """Ingest registers new PDFs in the database."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    stats = ingest_directory(conn, papers_dir)

    assert stats.total_pdfs == 2
    assert stats.new == 2
    assert stats.already_indexed == 0
    assert stats.changed == 0

    row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
    assert row[0] == 2
    conn.close()


def test_ingest_skip_existing(tmp_path, papers_dir):
    """Second ingest skips already-indexed files."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)
    stats = ingest_directory(conn, papers_dir)

    assert stats.new == 0
    assert stats.already_indexed == 2

    row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
    assert row[0] == 2
    conn.close()


def test_ingest_detect_changed(tmp_path, papers_dir):
    """Changed files are detected and re-indexed."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)

    # Modify a PDF
    pdf_path = papers_dir / "test_climate.pdf"
    doc = fitz.open(str(pdf_path))
    page = doc.new_page()
    page.insert_text((72, 72), "New content added")
    doc.save(str(pdf_path), incremental=True, encryption=0)
    doc.close()

    stats = ingest_directory(conn, papers_dir)
    assert stats.changed == 1
    conn.close()


def test_ingest_detect_missing(tmp_path, papers_dir):
    """Missing files are detected."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)

    # Remove a PDF
    (papers_dir / "ml_nlp_paper.pdf").unlink()

    stats = ingest_directory(conn, papers_dir)
    assert stats.missing == 1
    conn.close()


def test_ingest_dry_run(tmp_path, papers_dir):
    """Dry run doesn't write to database."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    stats = ingest_directory(conn, papers_dir, dry_run=True)

    assert stats.new == 2
    row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
    assert row[0] == 0  # Nothing written
    conn.close()


def test_ingest_page_count(populated_db):
    """Page count is extracted during ingest."""
    row = populated_db.execute(
        "SELECT page_count FROM papers WHERE filename = 'test_climate.pdf'"
    ).fetchone()
    assert row["page_count"] is not None
    assert row["page_count"] >= 1


def test_classify_digital_pdf(sample_pdf):
    """Digital PDF is classified as digital."""
    result = classify_document(sample_pdf)
    assert result == "digital"


def test_classify_scanned_pdf(scanned_pdf):
    """Scanned-like PDF is classified as scanned."""
    result = classify_document(scanned_pdf)
    assert result == "scanned"


def test_ingest_sets_scan_flag(populated_db):
    """Ingest sets is_scanned flag."""
    rows = populated_db.execute("SELECT is_scanned FROM papers").fetchall()
    for row in rows:
        assert row["is_scanned"] is not None  # Should be classified


def test_classify_encrypted_pdf_does_not_raise(encrypted_pdf):
    """A password-protected PDF opens fine in fitz but can't be paged through
    without authenticating. classify_document must not let that ValueError
    escape — it should fall back the same way it does for any other PDF it
    can't classify, not crash the caller.
    """
    result = classify_document(encrypted_pdf)
    assert result in ("digital", "scanned", "mixed")


def test_ingest_survives_encrypted_pdf(tmp_path, papers_dir, encrypted_pdf):
    """One password-protected PDF anywhere in the papers dir must not abort
    ingestion of the rest of the corpus.
    """
    import shutil

    shutil.copy(encrypted_pdf, papers_dir / "locked.pdf")

    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    stats = ingest_directory(conn, papers_dir)

    # The two normal PDFs from papers_dir plus the locked one all get
    # registered — ingestion doesn't stop partway through the directory.
    assert stats.total_pdfs == 3
    assert stats.new == 3
    row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
    assert row[0] == 3
    conn.close()


def test_ingest_skips_edeadlk(tmp_path, papers_dir, monkeypatch):
    """iCloud dataless files (EDEADLK) are skipped, not fatal."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    original_file_hash = __import__("pdf_gantry.utils", fromlist=["file_hash"]).file_hash
    call_count = 0

    def fake_hash(path, chunk_size=65536):
        nonlocal call_count
        call_count += 1
        if "test_climate" in str(path):
            err = OSError("Resource deadlock avoided")
            err.errno = 11
            raise err
        return original_file_hash(path, chunk_size)

    monkeypatch.setattr("pdf_gantry.ingest.file_hash", fake_hash)

    stats = ingest_directory(conn, papers_dir)

    assert stats.evicted == 1
    assert stats.new == 1  # the other PDF still got indexed
    row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
    assert row[0] == 1
    conn.close()
