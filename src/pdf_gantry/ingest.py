"""File scanning and registration of PDFs into the database."""

import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import fitz  # PyMuPDF

from .models import IngestStats
from .utils import file_hash, now_iso


def classify_document(pdf_path: Path, scan_threshold: float = 0.05) -> str:
    """
    Classify a PDF as 'scanned', 'digital', or 'mixed'.

    Heuristic: For each page, extract text blocks and calculate the ratio
    of text area to page area. If average ratio < scan_threshold, it's scanned.
    If above 0.5, it's digital. Otherwise mixed.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return "digital"  # default if we can't open

    try:
        if doc.page_count == 0:
            return "digital"

        # fitz.open() succeeds on a password-protected PDF without raising —
        # it only throws once page content is touched without authenticating.
        # A locked PDF (DRM'd publisher export, accidentally-encrypted
        # download) is unreadable to us either way, so treat it the same as
        # any other file we can't classify rather than letting it escape and
        # abort the whole ingest run.
        if doc.needs_pass:
            return "digital"

        ratios = []
        try:
            for page in doc:
                page_area = page.rect.width * page.rect.height
                if page_area == 0:
                    ratios.append(0.0)
                    continue
                blocks = page.get_text("blocks")
                text_area = sum(
                    (b[2] - b[0]) * (b[3] - b[1])
                    for b in blocks
                    if b[6] == 0  # type 0 = text block
                )
                ratios.append(text_area / page_area)
        except Exception:
            # Some other decode failure surfaced mid-document (corrupt page,
            # unsupported filter). Same fallback as "can't open" — don't let
            # one bad page take down the batch.
            return "digital"
    finally:
        doc.close()

    if not ratios:
        return "digital"

    avg_ratio = sum(ratios) / len(ratios)
    if avg_ratio < scan_threshold:
        return "scanned"
    elif avg_ratio > 0.15:
        return "digital"
    else:
        return "mixed"


def _get_pdf_metadata(pdf_path: Path) -> tuple[int | None, str, bool]:
    """Get page count, classification, and encrypted flag for a PDF.

    needs_pass is read off the same fitz.open() used for page_count — it's
    reliable to check before any page content is touched (unlike
    classify_document's own open, which is a separate handle), so this is a
    free byproduct rather than an extra file open.
    """
    try:
        doc = fitz.open(str(pdf_path))
        page_count = doc.page_count
        is_encrypted = bool(doc.needs_pass)
        doc.close()
    except Exception:
        page_count = None
        is_encrypted = False

    classification = classify_document(pdf_path)
    return page_count, classification, is_encrypted


def _fast_path_match(conn: sqlite3.Connection, rel_path: str, size: int, mtime: str) -> bool:
    """Check if file matches existing record by path + size + mtime (skip hash)."""
    row = conn.execute(
        "SELECT file_size, file_modified FROM papers WHERE path = ?",
        (rel_path,),
    ).fetchone()
    if row is None:
        return False
    return row["file_size"] == size and row["file_modified"] == mtime


def ingest_directory(
    conn: sqlite3.Connection,
    papers_dir: Path,
    dry_run: bool = False,
    scan_threshold: float = 0.05,
    progress_callback=None,
) -> IngestStats:
    """Scan papers_dir for PDFs and register them in the database."""
    stats = IngestStats()
    start = time.time()

    # Collect all PDF files
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    stats.total_pdfs = len(pdf_files)

    # Track existing paths for missing detection
    existing_paths = {
        row[0]
        for row in conn.execute("SELECT path FROM papers").fetchall()
    }
    seen_paths = set()

    for i, pdf_path in enumerate(pdf_files):
        rel_path = pdf_path.name  # flat directory, just filename
        seen_paths.add(rel_path)

        stat = pdf_path.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()

        # Fast path: if path+size+mtime match, skip hash
        if _fast_path_match(conn, rel_path, size, mtime):
            stats.already_indexed += 1
            if progress_callback:
                progress_callback(i + 1, stats.total_pdfs)
            continue

        # Compute hash — catch iCloud dataless files (EDEADLK)
        try:
            fhash = file_hash(pdf_path)
        except OSError as e:
            if e.errno == 11:  # EDEADLK: Resource deadlock avoided
                stats.evicted += 1
                if progress_callback:
                    progress_callback(i + 1, stats.total_pdfs)
                continue
            raise

        # Check if already in DB
        row = conn.execute(
            "SELECT id, file_hash FROM papers WHERE path = ?",
            (rel_path,),
        ).fetchone()

        if row is not None:
            if row["file_hash"] == fhash:
                # Hash matches but mtime/size changed (unlikely) - update metadata
                stats.already_indexed += 1
                if not dry_run:
                    conn.execute(
                        "UPDATE papers SET file_size = ?, file_modified = ?, updated_at = ? "
                        "WHERE id = ?",
                        (size, mtime, now_iso(), row["id"]),
                    )
            else:
                # Content changed - reset processing flags
                stats.changed += 1
                if not dry_run:
                    page_count, classification, is_encrypted = _get_pdf_metadata(pdf_path)
                    is_scanned = (
                        1 if classification == "scanned"
                        else (0 if classification == "digital" else None)
                    )
                    needs_ocr = 1 if classification in ("scanned", "mixed") else 0
                    conn.execute(
                        """UPDATE papers SET
                            file_hash = ?, file_size = ?, file_modified = ?,
                            page_count = ?, is_scanned = ?, needs_ocr = ?,
                            is_encrypted = ?,
                            has_text = 0, has_markdown = 0, has_embeddings = 0,
                            updated_at = ?
                        WHERE id = ?""",
                        (fhash, size, mtime, page_count, is_scanned, needs_ocr,
                         int(is_encrypted), now_iso(), row["id"]),
                    )
        else:
            # New file
            stats.new += 1
            if not dry_run:
                page_count, classification, is_encrypted = _get_pdf_metadata(pdf_path)
                is_scanned = (
                    1 if classification == "scanned"
                    else (0 if classification == "digital" else None)
                )
                needs_ocr = 1 if classification in ("scanned", "mixed") else 0
                now = now_iso()
                conn.execute(
                    """INSERT INTO papers (
                        path, filename, file_hash, file_size, file_modified,
                        page_count, is_scanned, needs_ocr, is_encrypted,
                        indexed_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (rel_path, pdf_path.name, fhash, size, mtime,
                     page_count, is_scanned, needs_ocr, int(is_encrypted), now, now),
                )

        if progress_callback:
            progress_callback(i + 1, stats.total_pdfs)

    # Detect missing files
    missing_paths = existing_paths - seen_paths
    stats.missing = len(missing_paths)

    if not dry_run:
        conn.commit()

    stats.elapsed_seconds = round(time.time() - start, 1)
    return stats
