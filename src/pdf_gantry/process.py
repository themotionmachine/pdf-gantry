"""Text extraction and markdown conversion from PDFs."""

import contextlib
import logging
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pymupdf as fitz  # not `import fitz`: that alias prints a deprecation line to stdout
import pymupdf4llm

from .models import ProcessStats
from .queue import DEFAULT_MAX_RETRIES, not_quarantined_condition
from .utils import now_iso

logger = logging.getLogger(__name__)


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
    with _parser_messages_captured():
        markdown = pymupdf4llm.to_markdown(str(pdf_path), table_strategy="lines")

    return raw_text, markdown


# Distinct PyMuPDF4LLM messages seen in this process since the last drain.
_PARSER_MESSAGES: list[str] = []


class _MessageSink:
    """File-like sink that keeps each distinct non-blank message line once."""

    def write(self, text: str) -> int:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("===") and line not in _PARSER_MESSAGES:
                _PARSER_MESSAGES.append(line)
        return len(text)

    def flush(self) -> None:
        pass


@contextlib.contextmanager
def _parser_messages_captured():
    """
    Keep PyMuPDF4LLM's chatter off stdout while it runs.

    to_markdown announces its OCR engine on every call, through
    pymupdf.message (stdout) after a bare print(). On stdout that corrupts
    `--json`; once per PDF it makes a working run look stuck. Both are
    collected here and reported once per run by process_documents.
    """
    previous = fitz._g_out_message
    sink = _MessageSink()
    fitz.set_messages(stream=sink)
    try:
        with contextlib.redirect_stdout(sink):
            yield
    finally:
        fitz._g_out_message = previous


def _drain_parser_messages() -> list[str]:
    messages = list(_PARSER_MESSAGES)
    _PARSER_MESSAGES.clear()
    return messages


# Module-level cache for Marker models (loaded once per worker process)
_MARKER_MODELS: dict = {}


def extract_text_marker(pdf_path: Path) -> tuple[str, str]:
    """
    Extract raw text and markdown from a PDF using Marker.
    Returns (raw_text, markdown). Marker produces high-quality markdown
    with LaTeX equations, proper tables, and layout-aware text.

    Models are cached at module level — loaded once per worker process (~8s),
    then reused for all subsequent calls.
    """
    global _MARKER_MODELS

    try:
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered
    except ImportError:
        raise ImportError("Marker not installed. Run: pip install pdf-gantry[quality]")

    if not _MARKER_MODELS:
        from marker.config.parser import ConfigParser
        from marker.models import create_model_dict
        config_parser = ConfigParser({"output_format": "markdown"})
        _MARKER_MODELS["models"] = create_model_dict()
        _MARKER_MODELS["config"] = config_parser.generate_config_dict()

    converter = PdfConverter(
        artifact_dict=_MARKER_MODELS["models"],
        config=_MARKER_MODELS["config"],
    )
    rendered = converter(str(pdf_path))
    markdown, ext, images = text_from_rendered(rendered)

    # Raw text via fitz (same as pymupdf4llm path)
    doc = fitz.open(str(pdf_path))
    raw_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    return raw_text, markdown


def _process_single(
    paper_id: int,
    pdf_path: str,
    papers_dir: str,
    db_path: str,
    method: str,
) -> tuple[int, bool, str | None, list[str]]:
    """
    Process a single PDF. Runs in a worker process.
    Returns (paper_id, success, error_message, parser_messages).
    """
    paper_id, success, error = _process_single_inner(
        paper_id, pdf_path, papers_dir, db_path, method,
    )
    return paper_id, success, error, _drain_parser_messages()


def _process_single_inner(
    paper_id: int,
    pdf_path: str,
    papers_dir: str,
    db_path: str,
    method: str,
) -> tuple[int, bool, str | None]:
    from .db import get_connection

    try:
        full_path = Path(papers_dir) / pdf_path

        if method == "pymupdf4llm":
            raw_text, markdown = extract_text_pymupdf(full_path)
        elif method == "marker":
            raw_text, markdown = extract_text_marker(full_path)
        else:
            return (paper_id, False, f"Unknown method: {method}")

        conn = get_connection(db_path)
        now = now_iso()

        # Get paper metadata for FTS
        row = conn.execute(
            "SELECT filename, title, authors, abstract FROM papers WHERE id = ?",
            (paper_id,),
        ).fetchone()

        # Read the prior text BEFORE overwriting paper_text — a legacy
        # contentless FTS5 table needs the OLD content to drop stale postings.
        old_row = conn.execute(
            "SELECT raw_text FROM paper_text WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        old_content = old_row["raw_text"] if old_row else ""

        # Store text
        conn.execute(
            """INSERT OR REPLACE INTO paper_text
                (paper_id, raw_text, markdown, text_length, markdown_length)
            VALUES (?, ?, ?, ?, ?)""",
            (paper_id, raw_text, markdown, len(raw_text), len(markdown)),
        )

        # Update FTS index. old_content only matters on a legacy (pre
        # contentless_delete) table; see fts.refresh_row.
        from .fts import refresh_row
        refresh_row(conn, paper_id, old_text=old_content or "")

        # Generate and store chunks
        from .chunking import chunk_markdown
        from .pages import pdf_page_texts

        # Per-page text lets each chunk be located on its PDF page(s). The
        # markdown itself carries no page boundaries. Best effort: a failure
        # here leaves pages NULL rather than failing the extraction.
        try:
            page_texts = pdf_page_texts(full_path)
        except Exception:
            page_texts = None
        raw_chunks = chunk_markdown(markdown, title=row["title"], page_texts=page_texts)

        # Clear old chunks (and their embeddings — vec0 has no CASCADE)
        conn.execute(
            "DELETE FROM chunk_vec WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE doc_id = ?)",
            (paper_id,),
        )
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (paper_id,))

        for i, chunk in enumerate(raw_chunks):
            conn.execute(
                """INSERT INTO chunks
                (doc_id, chunk_index, section_header, page_start, page_end,
                 text, char_offset)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (paper_id, i, chunk.section_header, chunk.page_start,
                 chunk.page_end, chunk.text, chunk.char_offset),
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
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ProcessStats:
    """
    Process documents matching the given IDs (or all needing text).

    When paper_ids is None, papers that have failed at least `max_retries` times
    are quarantined — skipped so a permanently-broken PDF isn't re-attempted on
    every run. Explicit paper_ids bypass the cap (an operator asked for them).
    Returns processing statistics.
    """
    # Marker is internally parallelized — multiple workers would OOM
    if method == "marker" and workers > 1:
        workers = 1

    stats = ProcessStats()
    start = time.time()

    if paper_ids is None:
        rows = conn.execute(
            "SELECT id, path FROM papers WHERE has_text = 0 "
            "AND (is_scanned = 0 OR is_scanned IS NULL) "
            "AND is_encrypted = 0 "
            f"AND {not_quarantined_condition(max_retries)}"
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
    succeeded_ids: list[int] = []

    # Use workers=1 to run in-process for simplicity with small batches
    if workers <= 1:
        for row in rows:
            pid, success, error, messages = _process_single(
                row["id"], row["path"], str(papers_dir), str(db_path), method
            )
            _merge_messages(stats.parser_messages, messages)
            if success:
                stats.succeeded += 1
                succeeded_ids.append(pid)
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
                pid, success, error, messages = future.result()
                _merge_messages(stats.parser_messages, messages)
                if success:
                    stats.succeeded += 1
                    succeeded_ids.append(pid)
                else:
                    stats.failed += 1
                completed += 1
                if progress_callback:
                    progress_callback(completed, stats.total)

    for message in stats.parser_messages:
        print(f"PDF parser: {message}", file=sys.stderr)

    stats.layout_titled = _populate_layout_titles(conn, papers_dir, succeeded_ids)
    stats.elapsed_seconds = round(time.time() - start, 1)
    return stats


def _merge_messages(into: list[str], messages: list[str]) -> None:
    for message in messages:
        if message not in into:
            into.append(message)


def _populate_layout_titles(conn, papers_dir: Path, paper_ids: list[int]) -> int:
    """Layout titles for freshly extracted papers. Best effort.

    Runs here, in the parent, because boilerplate is corpus-wide and a worker
    sees one paper. A failure leaves the papers pending for ``titles backfill``
    and never fails the extraction that already succeeded.
    """
    if not paper_ids:
        return 0
    from . import layout
    try:
        return layout.backfill_titles(conn, papers_dir, paper_ids=paper_ids)["titled"]
    except Exception as e:
        logger.warning("layout titles skipped: %s", e)
        return 0


def reset_errors(conn: sqlite3.Connection, paper_ids: list[int]) -> int:
    """
    Clear the error state of the given papers, re-admitting any that were
    quarantined (error_count >= cap) to default process/embed selection.

    Returns the number of rows updated.
    """
    if not paper_ids:
        return 0
    placeholders = ",".join("?" * len(paper_ids))
    cur = conn.execute(
        f"""UPDATE papers SET
            error_count = 0, last_error = NULL, updated_at = ?
        WHERE id IN ({placeholders})""",
        [now_iso(), *paper_ids],
    )
    conn.commit()
    return cur.rowcount
