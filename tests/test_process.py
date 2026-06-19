"""Tests for text extraction and processing."""

import json

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import extract_text_pymupdf, process_documents


def test_extract_text_from_digital_pdf(sample_pdf):
    """Text extraction produces non-empty results."""
    raw_text, markdown = extract_text_pymupdf(sample_pdf)
    assert "climate change" in raw_text.lower()
    assert len(raw_text) > 0
    assert len(markdown) > 0


def test_extract_markdown_from_digital_pdf(sample_pdf):
    """Markdown extraction produces output."""
    raw_text, markdown = extract_text_pymupdf(sample_pdf)
    assert len(markdown) > 0


def test_process_sets_flags(tmp_path, papers_dir):
    """Processing sets has_text and has_markdown flags."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)

    stats = process_documents(
        conn, papers_dir, db_path,
        workers=1,
    )

    assert stats.succeeded > 0

    # Re-read from DB to check flags
    conn2 = get_connection(str(db_path))
    rows = conn2.execute(
        "SELECT has_text, has_markdown, text_method FROM papers WHERE has_text = 1"
    ).fetchall()
    assert len(rows) > 0
    for row in rows:
        assert row["has_text"] == 1
        assert row["has_markdown"] == 1
        assert row["text_method"] == "pymupdf4llm"
    conn2.close()
    conn.close()


def test_fts_populated_after_processing(tmp_path, papers_dir):
    """FTS5 index has content after processing."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    # Search FTS
    conn2 = get_connection(str(db_path))
    rows = conn2.execute(
        "SELECT * FROM papers_fts WHERE papers_fts MATCH 'climate'"
    ).fetchall()
    assert len(rows) >= 1
    conn2.close()
    conn.close()


def test_process_with_limit(tmp_path, papers_dir):
    """Processing respects the limit parameter."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)

    stats = process_documents(
        conn, papers_dir, db_path,
        workers=1, limit=1,
    )

    assert stats.total == 1
    assert stats.succeeded == 1
    conn.close()


def test_paper_text_stored(tmp_path, papers_dir):
    """Raw text and markdown are stored in paper_text table."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    conn2 = get_connection(str(db_path))
    rows = conn2.execute(
        "SELECT raw_text, markdown, text_length, markdown_length FROM paper_text"
    ).fetchall()
    assert len(rows) > 0
    for row in rows:
        assert row["raw_text"] is not None
        assert row["markdown"] is not None
        assert row["text_length"] > 0
        assert row["markdown_length"] > 0
    conn2.close()
    conn.close()


def test_chunks_generated_after_processing(tmp_path, papers_dir):
    """Processing generates chunks in the chunks table."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    conn2 = get_connection(str(db_path))
    chunk_count = conn2.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert chunk_count > 0

    # Each paper should have at least one chunk
    papers = conn2.execute("SELECT id FROM papers WHERE has_text = 1").fetchall()
    for paper in papers:
        count = conn2.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (paper["id"],)
        ).fetchone()[0]
        assert count >= 1

    conn2.close()
    conn.close()


# --- Re-processing correctness: --force selection + stale-artifact cleanup ---


def test_cli_process_force_reselects_processed(tmp_path, papers_dir, monkeypatch):
    """`process --force` re-selects already-processed papers.

    Without --force a fully-processed corpus has nothing to do (exit 2);
    --force overrides the default has_text=0 filter and re-runs extraction.
    """
    from click.testing import CliRunner

    from pdf_gantry.cli import cli

    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(papers_dir))
    runner = CliRunner()

    r1 = runner.invoke(cli, ["process", "--json"])
    assert r1.exit_code == 0, r1.output
    assert json.loads(r1.output)["succeeded"] >= 1

    # Everything is processed now — the default filter selects nothing.
    r2 = runner.invoke(cli, ["process", "--json"])
    assert r2.exit_code == 2, r2.output

    # --force re-selects the already-processed papers.
    r3 = runner.invoke(cli, ["process", "--force", "--json"])
    assert r3.exit_code == 0, r3.output
    data = json.loads(r3.output)
    assert data["total"] >= 1
    assert data["succeeded"] >= 1


def test_reprocess_clears_stale_chunks_and_vectors(tmp_path, papers_dir, monkeypatch):
    """Re-processing a paper deletes its prior chunks and chunk_vec rows.

    Seeds a sentinel stale chunk + embedding, re-processes, and asserts the
    sentinel is gone (no orphaned chunks, no dangling vectors — vec0 has no
    CASCADE).
    """
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    paper_id = conn.execute(
        "SELECT id FROM papers WHERE has_text = 1 ORDER BY id LIMIT 1"
    ).fetchone()["id"]

    # Seed a stale chunk + embedding, mimicking a prior (different) extraction.
    cur = conn.execute(
        """INSERT INTO chunks (doc_id, chunk_index, section_header, page_start, text, char_offset)
           VALUES (?, 99, 'STALE', 1, 'STALE GARBAGE chunk text', 0)""",
        (paper_id,),
    )
    stale_chunk_id = cur.lastrowid
    conn.execute(
        "INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
        (stale_chunk_id, json.dumps([0.0] * 768)),
    )
    conn.commit()

    process_documents(conn, papers_dir, db_path, paper_ids=[paper_id], workers=1)

    orphan_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_id = ?", (stale_chunk_id,)
    ).fetchone()[0]
    assert orphan_chunks == 0, "stale chunk should be deleted on re-process"

    dangling_vec = conn.execute(
        "SELECT COUNT(*) FROM chunk_vec WHERE chunk_id = ?", (stale_chunk_id,)
    ).fetchone()[0]
    assert dangling_vec == 0, "stale chunk embedding should be deleted on re-process"
    conn.close()


def test_reprocess_clears_stale_fts_postings(tmp_path, papers_dir, monkeypatch):
    """Re-processing clears stale FTS postings before re-inserting.

    Contentless FTS5 requires the OLD content to delete the old postings. If
    re-processing overwrites paper_text before computing the delete, the old
    terms leak and keep matching. After re-extraction yields different text,
    the old term must no longer match and the new term must.
    """
    import pdf_gantry.process as process_mod

    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    paper_id = conn.execute(
        "SELECT id FROM papers WHERE has_text = 1 ORDER BY id LIMIT 1"
    ).fetchone()["id"]

    def hits(term):
        return [r["rowid"] for r in conn.execute(
            "SELECT rowid FROM papers_fts WHERE papers_fts MATCH ?", (term,)
        ).fetchall()]

    assert paper_id in hits("climate") or paper_id in hits("machine"), \
        "sanity: original term should match the paper before re-process"
    original_term = "climate" if paper_id in hits("climate") else "machine"

    # Re-extraction yields entirely different text.
    new_text = "Totallynewword distinctive replacement content."
    new_md = "## Totallynewword\n\ndistinctive replacement content."
    monkeypatch.setattr(
        process_mod, "extract_text_pymupdf", lambda _path: (new_text, new_md)
    )

    process_documents(conn, papers_dir, db_path, paper_ids=[paper_id], workers=1)

    assert paper_id not in hits(original_term), "stale FTS posting still matches"
    assert paper_id in hits("Totallynewword"), "new FTS posting missing"
    conn.close()
