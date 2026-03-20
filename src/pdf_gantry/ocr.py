"""OCR detection and processing for scanned PDFs."""

import sqlite3
import time
from pathlib import Path

from .models import ProcessStats
from .utils import now_iso


def _check_surya_available():
    """Check if Surya OCR is installed."""
    try:
        import surya  # noqa: F401
        return True
    except ImportError:
        return False


def ocr_document(pdf_path: Path) -> tuple[str, str]:
    """
    Run OCR on a scanned PDF using Surya.
    Returns (raw_text, markdown).
    """
    try:
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.detection.processor import load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
        import fitz
    except ImportError:
        raise ImportError("Surya OCR not installed. Run: pip install pdf-gantry[ocr]")

    # Open PDF and convert pages to images
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()

    if not images:
        return "", ""

    # Load models
    det_model = load_det_model()
    det_processor = load_det_processor()
    rec_model = load_rec_model()
    rec_processor = load_rec_processor()

    # Run OCR
    langs = [["en"]] * len(images)
    results = run_ocr(images, langs, det_model, det_processor, rec_model, rec_processor)

    # Extract text
    pages_text = []
    for page_result in results:
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
    progress_callback=None,
) -> ProcessStats:
    """Process scanned PDFs with OCR."""
    if not _check_surya_available():
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
    completed = 0

    for row in rows:
        paper_id = row["id"]
        pdf_path = papers_dir / row["path"]

        try:
            raw_text, markdown = ocr_document(pdf_path)
            now = now_iso()

            conn.execute(
                """INSERT OR REPLACE INTO paper_text
                    (paper_id, raw_text, markdown, text_length, markdown_length)
                VALUES (?, ?, ?, ?, ?)""",
                (paper_id, raw_text, markdown, len(raw_text), len(markdown)),
            )

            # Update FTS
            paper = conn.execute(
                "SELECT filename, title, authors, abstract FROM papers WHERE id = ?",
                (paper_id,),
            ).fetchone()

            conn.execute(
                "INSERT INTO papers_fts(rowid, filename, title, authors, abstract, text_content) VALUES (?, ?, ?, ?, ?, ?)",
                (paper_id, paper["filename"] or "", paper["title"] or "",
                 paper["authors"] or "", paper["abstract"] or "", raw_text),
            )

            conn.execute(
                """UPDATE papers SET
                    has_text = 1, has_markdown = 1, needs_ocr = 0,
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
