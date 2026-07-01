"""OCR detection and processing for scanned PDFs using Surya."""

import sqlite3
import time
from pathlib import Path

from .models import ProcessStats
from .utils import now_iso

# Lazy-loaded predictor singletons (expensive to initialize)
_foundation_predictor = None
_recognition_predictor = None
_detection_predictor = None


def _check_surya_available():
    """Check if Surya OCR is installed."""
    try:
        import surya  # noqa: F401
        return True
    except ImportError:
        return False


def _load_predictors():
    """Load Surya predictors once and cache at module level."""
    global _foundation_predictor, _recognition_predictor, _detection_predictor
    if _recognition_predictor is None:
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        _foundation_predictor = FoundationPredictor()
        _recognition_predictor = RecognitionPredictor(_foundation_predictor)
        _detection_predictor = DetectionPredictor()
    return _recognition_predictor, _detection_predictor



def ocr_document(pdf_path: Path) -> tuple[str, str]:
    """
    Run OCR on a scanned PDF using Surya.
    Returns (raw_text, markdown).
    """
    if not _check_surya_available():
        raise ImportError("Surya OCR not installed. Run: pip install pdf-gantry[ocr]")

    import io

    import fitz
    from PIL import Image

    # Open PDF and convert pages to images
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()

    if not images:
        return "", ""

    # Load predictors (cached at module level)
    rec_predictor, det_predictor = _load_predictors()

    # Run OCR
    predictions = rec_predictor(images, det_predictor=det_predictor)

    # Extract text
    pages_text = []
    for page_result in predictions:
        lines = [line.text for line in page_result.text_lines]
        pages_text.append("\n".join(lines))

    raw_text = "\n\n".join(pages_text)
    # Basic markdown: separate pages with headers
    markdown_pages = []
    for i, text in enumerate(pages_text):
        markdown_pages.append(f"## Page {i + 1}\n\n{text}")
    markdown = "\n\n".join(markdown_pages)

    return raw_text, markdown


def process_ocr_documents(
    conn: sqlite3.Connection,
    papers_dir: Path,
    db_path: Path,
    paper_ids: list[int] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    progress_callback=None,
) -> ProcessStats:
    """Process scanned PDFs with OCR."""
    if not dry_run and not _check_surya_available():
        raise ImportError("Surya OCR not installed. Run: pip install pdf-gantry[ocr]")

    stats = ProcessStats()
    start = time.time()

    if paper_ids is not None:
        placeholders = ",".join("?" * len(paper_ids))
        rows = conn.execute(
            f"SELECT id, path FROM papers WHERE id IN ({placeholders})",
            paper_ids,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, path FROM papers WHERE needs_ocr = 1 AND ocr_completed_at IS NULL"
        ).fetchall()

    if limit:
        rows = rows[:limit]

    stats.total = len(rows)

    if dry_run:
        stats.elapsed_seconds = round(time.time() - start, 1)
        return stats

    completed = 0

    for row in rows:
        paper_id = row["id"]
        pdf_path = papers_dir / row["path"]

        try:
            raw_text, markdown = ocr_document(pdf_path)
            now = now_iso()

            paper = conn.execute(
                "SELECT filename, title, authors, abstract FROM papers WHERE id = ?",
                (paper_id,),
            ).fetchone()

            # Read the prior text BEFORE overwriting — needed to delete stale
            # postings from the contentless FTS5 index.
            old_row = conn.execute(
                "SELECT raw_text FROM paper_text WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            old_text = old_row["raw_text"] if old_row else ""

            conn.execute(
                """INSERT OR REPLACE INTO paper_text
                    (paper_id, raw_text, markdown, text_length, markdown_length)
                VALUES (?, ?, ?, ?, ?)""",
                (paper_id, raw_text, markdown, len(raw_text), len(markdown)),
            )

            # Update FTS (contentless: delete old postings before inserting new).
            existing_fts = conn.execute(
                "SELECT rowid FROM papers_fts WHERE rowid = ?", (paper_id,)
            ).fetchone()
            if existing_fts:
                conn.execute(
                    "INSERT INTO papers_fts(papers_fts, rowid, filename, title, authors, "
                    "abstract, text_content) "
                    "VALUES('delete', ?, ?, ?, ?, ?, ?)",
                    (paper_id, paper["filename"] or "", paper["title"] or "",
                     paper["authors"] or "", paper["abstract"] or "", old_text or ""),
                )
            conn.execute(
                "INSERT INTO papers_fts(rowid, filename, title, authors, abstract, text_content) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (paper_id, paper["filename"] or "", paper["title"] or "",
                 paper["authors"] or "", paper["abstract"] or "", raw_text),
            )

            # Re-derive chunks from the OCR markdown — otherwise chunks (and their
            # embeddings) keep answering with the pre-OCR garbage extraction.
            from .chunking import chunk_markdown
            raw_chunks = chunk_markdown(markdown, title=paper["title"])
            conn.execute(
                "DELETE FROM chunk_vec WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                (paper_id,),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (paper_id,))
            for i, chunk in enumerate(raw_chunks):
                conn.execute(
                    """INSERT INTO chunks
                    (doc_id, chunk_index, section_header, page_start, text, char_offset)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (paper_id, i, chunk.section_header, chunk.page_start,
                     chunk.text, chunk.char_offset),
                )

            conn.execute(
                """UPDATE papers SET
                    has_text = 1, has_markdown = 1, needs_ocr = 0,
                    has_chunk_embeddings = 0,
                    text_method = 'surya', ocr_method = 'surya',
                    text_extracted_at = ?, ocr_completed_at = ?,
                    markdown_method = 'surya', markdown_extracted_at = ?,
                    updated_at = ?
                WHERE id = ?""",
                (now, now, now, now, paper_id),
            )
            stats.succeeded += 1

        except Exception as e:
            conn.execute(
                """UPDATE papers SET
                    last_error = ?, error_count = error_count + 1,
                    last_error_at = ?, updated_at = ?
                WHERE id = ?""",
                (str(e), now_iso(), now_iso(), paper_id),
            )
            stats.failed += 1

        completed += 1
        if completed % 5 == 0:
            conn.commit()
        if progress_callback:
            progress_callback(completed, stats.total)

    conn.commit()
    stats.elapsed_seconds = round(time.time() - start, 1)
    return stats
