"""Tests for `gantry chunks backfill-pages`: page provenance for EXISTING chunks."""

import json
import shutil
import struct

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents

from .test_pages import PAGE_TEXTS, _write_pdf


def _vec(axis: int, dim: int = 768) -> bytes:
    v = [0.0] * dim
    v[axis] = 1.0
    return struct.pack(f"{dim}f", *v)


@pytest.fixture
def indexed(tmp_path):
    """A processed 3-page paper whose chunks have embeddings but NULL pages
    (the state of every chunk in the real index before this change)."""
    papers = tmp_path / "papers"
    papers.mkdir()
    _write_pdf(papers / "three.pdf", PAGE_TEXTS)
    shutil.copy(papers / "three.pdf", papers / "copy.pdf")
    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers)
    process_documents(conn, papers, db_path, workers=1)
    expected = {
        r["chunk_id"]: (r["page_start"], r["page_end"])
        for r in conn.execute("SELECT chunk_id, page_start, page_end FROM chunks")
    }
    for i, cid in enumerate(expected):
        conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                     (cid, _vec(i % 768)))
    conn.execute("UPDATE chunks SET page_start = NULL, page_end = NULL")
    conn.commit()
    return conn, papers, db_path, expected


def _snapshot(conn):
    chunks = conn.execute(
        "SELECT chunk_id, doc_id, chunk_index, text, section_header, char_offset "
        "FROM chunks ORDER BY chunk_id").fetchall()
    vecs = conn.execute("SELECT chunk_id, embedding FROM chunk_vec ORDER BY chunk_id").fetchall()
    flags = conn.execute("SELECT id, has_chunk_embeddings FROM papers ORDER BY id").fetchall()
    return [tuple(r) for r in chunks], [tuple(r) for r in vecs], [tuple(r) for r in flags]


def test_backfill_restores_pages_without_touching_text_or_vectors(indexed):
    from pdf_gantry.pages import backfill_pages

    conn, papers, _, expected = indexed
    before = _snapshot(conn)
    report = backfill_pages(conn, papers)

    after = {
        r["chunk_id"]: (r["page_start"], r["page_end"])
        for r in conn.execute("SELECT chunk_id, page_start, page_end FROM chunks")
    }
    assert after == expected
    assert _snapshot(conn) == before
    assert report["papers"] == 2
    assert report["chunks"] == len(expected)
    assert report["assigned"] == len(expected)
    assert report["coverage"] == 1.0
    assert report["updated"] == len(expected)


def test_backfill_dry_run_writes_nothing(indexed):
    from pdf_gantry.pages import backfill_pages

    conn, papers, _, expected = indexed
    report = backfill_pages(conn, papers, dry_run=True)
    assert report["dry_run"] is True
    assert report["assigned"] == len(expected)
    assert report["updated"] == 0
    nulls = conn.execute("SELECT COUNT(*) FROM chunks WHERE page_start IS NULL").fetchone()[0]
    assert nulls == len(expected)


def test_backfill_scoped_to_ids(indexed):
    from pdf_gantry.pages import backfill_pages

    conn, papers, _, _ = indexed
    doc = conn.execute("SELECT MIN(id) FROM papers").fetchone()[0]
    report = backfill_pages(conn, papers, paper_ids=[doc])
    assert report["papers"] == 1
    other_nulls = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id != ? AND page_start IS NULL", (doc,)
    ).fetchone()[0]
    assert other_nulls > 0


def test_backfill_skips_already_paged_unless_force(indexed):
    from pdf_gantry.pages import backfill_pages

    conn, papers, _, _ = indexed
    backfill_pages(conn, papers)
    again = backfill_pages(conn, papers)
    assert again["papers"] == 0
    forced = backfill_pages(conn, papers, force=True)
    assert forced["papers"] == 2


def test_backfill_reports_missing_pdf(indexed):
    from pdf_gantry.pages import backfill_pages

    conn, papers, _, _ = indexed
    (papers / "copy.pdf").unlink()
    report = backfill_pages(conn, papers)
    assert len(report["errors"]) == 1
    assert report["errors"][0]["filename"] == "copy.pdf"
    assert report["papers"] == 2


def test_backfill_reports_ambiguous_chunks(indexed):
    """Chunks the mapper can't place are counted and listed per paper."""
    from pdf_gantry.pages import backfill_pages

    conn, papers, _, _ = indexed
    doc = conn.execute("SELECT MIN(id) FROM papers").fetchone()[0]
    conn.execute(
        "INSERT INTO chunks (doc_id, chunk_index, text) VALUES (?, 99, ?)",
        (doc, "Wholly unrelated prose about volcanic geology and basalt columns."),
    )
    conn.commit()
    report = backfill_pages(conn, papers, paper_ids=[doc])
    assert report["by_status"]["unlocated"] + report["by_status"]["interpolated"] >= 1
    amb = report["ambiguous"]
    assert amb and amb[0]["doc_id"] == doc


def test_backfill_ocr_paper_uses_page_headers_without_pdf(tmp_path):
    """An OCR'd paper (text_method=surya) is paged from '## Page N' headers;
    its PDF (a scan with no text layer) is never needed."""
    from pdf_gantry.pages import backfill_pages

    conn = get_connection(str(tmp_path / "i.db"))
    conn.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at, text_method) "
        "VALUES ('scan.pdf','scan.pdf','h',1,'t','t','t','surya')"
    )
    conn.execute("INSERT INTO chunks (doc_id, chunk_index, section_header, text) "
                  "VALUES (1, 0, 'Page 1', 'first page words')")
    conn.execute("INSERT INTO chunks (doc_id, chunk_index, section_header, text) "
                  "VALUES (1, 1, 'Page 4', 'fourth page words')")
    conn.commit()
    report = backfill_pages(conn, tmp_path / "no-such-dir")
    rows = conn.execute("SELECT page_start, page_end FROM chunks ORDER BY chunk_index").fetchall()
    assert [tuple(r) for r in rows] == [(1, 1), (4, 4)]
    assert report["errors"] == []


# --- CLI -------------------------------------------------------------------


def _config(tmp_path, papers, db_path, monkeypatch):
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(db_path.parent))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(papers))


def test_cli_chunks_backfill_pages_json(indexed, tmp_path, monkeypatch):
    conn, papers, db_path, expected = indexed
    conn.close()
    _config(tmp_path, papers, db_path, monkeypatch)
    result = CliRunner().invoke(cli, ["chunks", "backfill-pages", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["assigned"] == len(expected)
    assert data["coverage"] == 1.0


def test_cli_chunks_backfill_pages_dry_run_and_ids(indexed, tmp_path, monkeypatch):
    conn, papers, db_path, _ = indexed
    doc = conn.execute("SELECT MIN(id) FROM papers").fetchone()[0]
    conn.close()
    _config(tmp_path, papers, db_path, monkeypatch)
    result = CliRunner().invoke(
        cli, ["chunks", "backfill-pages", "--ids", str(doc), "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["papers"] == 1 and data["dry_run"] is True and data["updated"] == 0


def test_cli_chunks_backfill_pages_text_output(indexed, tmp_path, monkeypatch):
    conn, papers, db_path, _ = indexed
    conn.close()
    _config(tmp_path, papers, db_path, monkeypatch)
    result = CliRunner().invoke(cli, ["chunks", "backfill-pages"])
    assert result.exit_code == 0, result.output
    assert "100.0%" in result.output
