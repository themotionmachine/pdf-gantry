"""Text extraction and markdown conversion from PDFs."""

import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm

from .models import ProcessStats
from .utils import now_iso


def extract_text_pymupdf(pdf_path: Path) -> tuple[str, str]:
    """
    Extract raw text and markdown from a PDF using PyMuPDF/PyMuPDF4LLM.
    Returns (raw_text, markdown).
    """
    # Raw text extraction
    doc = fitz.open(str(pdf_path))
    raw_pages = []
    for page in doc:
        raw_pages.append(page.get_text())
    doc.close()
    raw_text = "\n".join(raw_pages)

    # Markdown extraction — table_strategy="lines" handles booktabs-style tables
    markdown = pymupdf4llm.to_markdown(str(pdf_path), table_strategy="lines")

    return raw_text, markdown


def _process_single(
    paper_id: int,
    pdf_path: str,
    papers_dir: str,
    db_path: str,
    method: str,
) -> tuple[int, bool, str | None]:
    """
    Process a single PDF. Runs in a worker process.
    Returns (paper_id, success, error_message).
    """
    from .db import get_connection

    try:
        full_path = Path(papers_dir) / pdf_path

        if method == "pymupdf4llm":
            raw_text, markdown = extract_text_pymupdf(full_path)
        else:
            return (paper_id, False, f"Unknown method: {method}")

        conn = get_connection(db_path)
        now = now_iso()

        # Store text
        conn.execute(
            """INSERT OR REPLACE INTO paper_text
                (paper_id, raw_text, markdown, text_length, markdown_length)
            VALUES (?, ?, ?, ?, ?)""",
            (paper_id, raw_text, markdown, len(raw_text), len(markdown)),
        )

        # Get paper metadata for FTS
        row = conn.execute(
            "SELECT filename, title, authors, abstract FROM papers WHERE id = ?",
            (paper_id,),
        ).fetchone()

        # Update FTS index
        # For contentless FTS5, check if row exists before trying to delete
        existing_fts = conn.execute(
            "SELECT rowid FROM papers_fts WHERE rowid = ?", (paper_id,)
        ).fetchone()
        if existing_fts:
            # Must provide old content for contentless delete
            old_text = conn.execute(
                "SELECT raw_text FROM paper_text WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            old_content = old_text["raw_text"] if old_text else ""
            conn.execute(
                "INSERT INTO papers_fts(papers_fts, rowid, filename, title, authors, abstract, text_content) "
                "VALUES('delete', ?, ?, ?, ?, ?, ?)",
                (paper_id, row["filename"] or "", row["title"] or "",
                 row["authors"] or "", row["abstract"] or "", old_content or ""),
            )
        conn.execute(
            "INSERT INTO papers_fts(rowid, filename, title, authors, abstract, text_content) VALUES (?, ?, ?, ?, ?, ?)",
            (paper_id, row["filename"] or "", row["title"] or "", row["authors"] or "", row["abstract"] or "", raw_text),
        )

        # Generate and store chunks
        from .chunking import chunk_markdown
        raw_chunks = chunk_markdown(markdown, title=row["title"])

        # Clear old chunks (and their embeddings — vec0 has no CASCADE)
        conn.execute(
            "DELETE FROM chunk_vec WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)",
            (paper_id,),
        )
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (paper_id,))

        for i, chunk in enumerate(raw_chunks):
            conn.execute(
                """INSERT INTO chunks (doc_id, chunk_index, section_header, page_start, text, char_offset)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (paper_id, i, chunk.section_header, chunk.page_start, chunk.text, chunk.char_offset),
            )

        # Update processing flags (reset chunk embeddings since chunks changed)
        conn.execute(
            """UPDATE papers SET
                has_text = 1, has_markdown = 1, has_chunk_embeddings = 0,
                text_method = ?, text_extracted_at = ?,
                markdown_method = ?, markdown_extracted_at = ?,
                updated_at = ?
            WHERE id = ?""",
            (method, now, method, now, now, paper_id),
        )
        conn.commit()
        conn.close()
        return (paper_id, True, None)

    except Exception as e:
        try:
            conn = get_connection(db_path)
            conn.execute(
                """UPDATE papers SET
                    last_error = ?, error_count = error_count + 1,
                    last_error_at = ?, updated_at = ?
                WHERE id = ?""",
                (str(e), now_iso(), now_iso(), paper_id),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        return (paper_id, False, str(e))


def process_documents(
    conn: sqlite3.Connection,
    papers_dir: Path,
    db_path: Path,
    paper_ids: list[int] | None = None,
    method: str = "pymupdf4llm",
    workers: int = 4,
    limit: int | None = None,
    progress_callback=None,
) -> ProcessStats:
    """
    Process documents matching the given IDs (or all needing text).
    Returns processing statistics.
    """
    stats = ProcessStats()
    start = time.time()

    if paper_ids is None:
        rows = conn.execute(
            "SELECT id, path FROM papers WHERE has_text = 0 AND (is_scanned = 0 OR is_scanned IS NULL)"
        ).fetchall()
    else:
        placeholders = ",".join("?" * len(paper_ids))
        rows = conn.execute(
            f"SELECT id, path FROM papers WHERE id IN ({placeholders})",
            paper_ids,
        ).fetchall()

    if limit:
        rows = rows[:limit]

    stats.total = len(rows)
    if stats.total == 0:
        return stats

    completed = 0

    # Use workers=1 to run in-process for simplicity with small batches
    if workers <= 1:
        for row in rows:
            pid, success, error = _process_single(
                row["id"], row["path"], str(papers_dir), str(db_path), method
            )
            if success:
                stats.succeeded += 1
            else:
                stats.failed += 1
            completed += 1
            if progress_callback:
                progress_callback(completed, stats.total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_single,
                    row["id"], row["path"], str(papers_dir), str(db_path), method,
                ): row["id"]
                for row in rows
            }
            for future in as_completed(futures):
                pid, success, error = future.result()
                if success:
                    stats.succeeded += 1
                else:
                    stats.failed += 1
                completed += 1
                if progress_callback:
                    progress_callback(completed, stats.total)

    stats.elapsed_seconds = round(time.time() - start, 1)
    return stats
