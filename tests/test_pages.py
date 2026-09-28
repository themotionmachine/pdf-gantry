"""Tests for chunk page provenance: mapping chunk text back to PDF pages."""

import sqlite3

import fitz
import pytest

from pdf_gantry.db import SCHEMA_VERSION, get_connection

# Three pages of distinct prose. Each page is long enough that a small
# max_chars forces chunks to break inside a page and across a boundary.
PAGE_TEXTS = [
    (
        "Deliberative democracy asks citizens to reason together about public "
        "problems. Mini publics such as citizens assemblies select members by "
        "lottery so that the room resembles the population. Facilitators keep "
        "discussion balanced and make sure quieter participants are heard."
    ),
    (
        "Algorithmic curation shapes what people see on social platforms. "
        "Recommendation systems optimise engagement, which can amplify outrage "
        "and sensational claims. Researchers measure exposure with browsing "
        "panels and donated data rather than relying on platform reports."
    ),
    (
        "Coastal wetlands store large amounts of carbon in waterlogged soils. "
        "Restoration projects reconnect tidal flows and replant marsh grasses. "
        "Monitoring programmes track sediment accretion, salinity and the "
        "return of birds that depend on intact estuarine habitat."
    ),
]


def _write_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
        page.insert_textbox(rect, text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def three_page_pdf(tmp_path):
    return _write_pdf(tmp_path / "three_pages.pdf", PAGE_TEXTS)


# --- schema ----------------------------------------------------------------


def _chunk_columns(conn):
    return {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}


def test_fresh_schema_has_page_end(tmp_db):
    assert "page_end" in _chunk_columns(tmp_db)
    assert "page_start" in _chunk_columns(tmp_db)


def test_migration_adds_page_end_to_v6_db(tmp_path):
    """An existing v6 index gains chunks.page_end without losing chunk rows."""
    db_path = tmp_path / "old.db"
    conn = get_connection(str(db_path))
    # Rewind the fresh schema to look like v6: drop page_end, reset version.
    conn.execute("ALTER TABLE chunks DROP COLUMN page_end")
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (6, 'x')")
    conn.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at) VALUES ('a.pdf','a.pdf','h',1,'t','t','t')"
    )
    conn.execute(
        "INSERT INTO chunks (doc_id, chunk_index, text) VALUES (1, 0, 'kept')"
    )
    conn.commit()
    conn.close()

    conn = get_connection(str(db_path))
    assert "page_end" in _chunk_columns(conn)
    assert conn.execute("SELECT text FROM chunks").fetchone()[0] == "kept"
    version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert version == SCHEMA_VERSION >= 7


# --- the mapper --------------------------------------------------------------


def test_map_chunks_locates_each_page():
    from pdf_gantry.pages import map_chunks_to_pages

    chunks = [PAGE_TEXTS[0], PAGE_TEXTS[1], PAGE_TEXTS[2]]
    rows = map_chunks_to_pages(PAGE_TEXTS, chunks)
    assert [(r.page_start, r.page_end) for r in rows] == [(1, 1), (2, 2), (3, 3)]
    assert all(r.status == "located" for r in rows)


def test_map_chunks_spanning_boundary():
    """A chunk that starts on page 1 and ends on page 2 gets a 1-2 range."""
    from pdf_gantry.pages import map_chunks_to_pages

    spanning = PAGE_TEXTS[0][-120:] + "\n\n" + PAGE_TEXTS[1][:120]
    rows = map_chunks_to_pages(PAGE_TEXTS, [spanning])
    assert (rows[0].page_start, rows[0].page_end) == (1, 2)


def test_map_chunks_tolerates_markdown_and_hyphenation():
    """Markdown emphasis and a line-break hyphen don't defeat the probe."""
    from pdf_gantry.pages import map_chunks_to_pages

    md = "**Recommendation systems** optimise engage-\nment, which can amplify " \
         "outrage and sensational claims. Researchers measure exposure."
    rows = map_chunks_to_pages(PAGE_TEXTS, [md])
    assert (rows[0].page_start, rows[0].page_end) == (2, 2)


def test_map_chunks_repeated_text_resolves_in_order():
    """Text repeated on two pages maps to the occurrence after the previous chunk."""
    from pdf_gantry.pages import map_chunks_to_pages

    header = "Journal of Repeated Running Headers Volume Twelve Issue Three"
    pages = [header + " " + PAGE_TEXTS[0], header + " " + PAGE_TEXTS[1]]
    chunks = [pages[0], header + " " + PAGE_TEXTS[1][:60]]
    rows = map_chunks_to_pages(pages, chunks)
    assert rows[1].page_start == 2


def test_map_chunks_short_chunk_interpolated_from_neighbours():
    from pdf_gantry.pages import map_chunks_to_pages

    rows = map_chunks_to_pages(PAGE_TEXTS, [PAGE_TEXTS[1], "Fig. 2", PAGE_TEXTS[2]])
    assert rows[1].status == "interpolated"
    assert (rows[1].page_start, rows[1].page_end) == (2, 3)


def test_map_chunks_no_text_pages():
    from pdf_gantry.pages import map_chunks_to_pages

    rows = map_chunks_to_pages(["", ""], [PAGE_TEXTS[0]])
    assert rows[0].status == "no_text"
    assert rows[0].page_start is None


def test_map_chunks_unlocatable():
    from pdf_gantry.pages import map_chunks_to_pages

    rows = map_chunks_to_pages(
        PAGE_TEXTS, ["Completely unrelated sentence about volcanic geology of Iceland."]
    )
    assert rows[0].status == "unlocated"
    assert rows[0].page_start is None


def test_ocr_page_header_parsing():
    from pdf_gantry.pages import page_from_ocr_header

    assert page_from_ocr_header("Page 3") == 3
    assert page_from_ocr_header("Page 12") == 12
    assert page_from_ocr_header("Introduction") is None
    assert page_from_ocr_header(None) is None


# --- chunk-time population -------------------------------------------------


def test_chunk_markdown_assigns_pages_when_given_page_texts():
    from pdf_gantry.chunking import chunk_markdown

    md = "\n\n".join(PAGE_TEXTS)
    chunks = chunk_markdown(md, max_chars=300, overlap_chars=0, min_chars=50,
                            page_texts=PAGE_TEXTS)
    assert len(chunks) >= 3
    assert all(c.page_start is not None for c in chunks)
    assert all(c.page_end >= c.page_start for c in chunks)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3


def test_chunk_markdown_without_page_texts_leaves_pages_null():
    from pdf_gantry.chunking import chunk_markdown

    chunks = chunk_markdown("\n\n".join(PAGE_TEXTS), max_chars=300, min_chars=50)
    assert all(c.page_start is None and c.page_end is None for c in chunks)


def test_process_populates_chunk_pages(tmp_path, three_page_pdf):
    """End to end: gantry process writes page_start/page_end on every chunk."""
    from pdf_gantry.ingest import ingest_directory
    from pdf_gantry.process import process_documents

    papers = tmp_path / "papers"
    papers.mkdir()
    import shutil
    shutil.copy(three_page_pdf, papers / "three_pages.pdf")

    db_path = tmp_path / "t.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers)
    stats = process_documents(conn, papers, db_path, workers=1)
    assert stats.succeeded == 1

    rows = conn.execute(
        "SELECT page_start, page_end, text FROM chunks ORDER BY chunk_index"
    ).fetchall()
    assert rows
    for r in rows:
        assert r["page_start"] is not None, r["text"][:80]
        assert 1 <= r["page_start"] <= r["page_end"] <= 3
    conn.close()


def test_ocr_chunks_get_page_from_page_header():
    """OCR markdown carries '## Page N' headers; chunks inherit that page."""
    from pdf_gantry.chunking import chunk_markdown
    from pdf_gantry.pages import assign_pages_from_ocr_headers

    md = "\n\n".join(f"## Page {i + 1}\n\n{t}" for i, t in enumerate(PAGE_TEXTS))
    chunks = assign_pages_from_ocr_headers(chunk_markdown(md, min_chars=10))
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2), (3, 3)]


def test_sqlite_supports_drop_column():
    """Guard for the migration test's setup (DROP COLUMN needs SQLite >= 3.35)."""
    assert sqlite3.sqlite_version_info >= (3, 35, 0)
