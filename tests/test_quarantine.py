"""Tests for poison-pill quarantine: cap retries on permanently-broken PDFs.

A PDF that PyMuPDF can't open (e.g. paper 103, error_count=133 on the live
index) is re-attempted on every nightly pipeline run forever. These tests pin
the behavior that caps retries: once error_count reaches the cap, the paper is
quarantined — excluded from default process/embed selection, surfaced via
`queue --is broken`, and resettable so an operator can re-try after a fix.
"""

import json

from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.embeddings import embed_documents
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents, reset_errors
from pdf_gantry.queue import DEFAULT_MAX_RETRIES, query_queue, queue_count

CAP = DEFAULT_MAX_RETRIES


def _insert_paper(conn, paper_id, *, error_count, has_text=0, is_scanned=0,
                  with_text=False):
    """Insert a minimal paper row with a controlled error_count."""
    conn.execute(
        """INSERT INTO papers
            (id, path, filename, file_hash, file_size, file_modified,
             page_count, has_text, has_embeddings, is_scanned, error_count,
             indexed_at, updated_at)
           VALUES (?, ?, ?, ?, 1, '2026-01-01', 1, ?, 0, ?, ?,
                   '2026-01-01', '2026-01-01')""",
        (paper_id, f"p{paper_id}.pdf", f"p{paper_id}.pdf", f"h{paper_id}",
         has_text, is_scanned, error_count),
    )
    if with_text:
        conn.execute(
            "INSERT INTO paper_text (paper_id, raw_text, markdown, text_length, markdown_length) "
            "VALUES (?, 'body', '', 4, 0)",
            (paper_id,),
        )
    conn.commit()


# --- (a) excluded from default process selection ---

def test_process_default_excludes_quarantined(tmp_path, papers_dir):
    """A paper at/above the retry cap is skipped by default `process` selection."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)  # two digital papers, error_count 0

    # Quarantine paper 1 (simulating the poison pill).
    conn.execute("UPDATE papers SET error_count = ? WHERE id = 1", (CAP,))
    conn.commit()

    process_documents(conn, papers_dir, db_path, workers=1)  # paper_ids=None

    conn2 = get_connection(str(db_path))
    # Quarantined paper was NOT processed...
    assert conn2.execute("SELECT has_text FROM papers WHERE id = 1").fetchone()["has_text"] == 0
    # ...the healthy one was.
    assert conn2.execute("SELECT has_text FROM papers WHERE id = 2").fetchone()["has_text"] == 1
    conn2.close()
    conn.close()


# --- (a) excluded from default embed selection ---

def test_embed_default_excludes_quarantined(tmp_path):
    """A quarantined paper with text is skipped by default `embed` selection.

    The paper has has_text=1 and has_embeddings=0, so it matches the base embed
    predicate. If the quarantine clause were missing it would be selected and
    embed_documents would try to load the (absent) model and raise ImportError.
    A clean total==0 proves the exclusion.
    """
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    _insert_paper(conn, 1, error_count=CAP, has_text=1, with_text=True)

    stats = embed_documents(conn, db_path)  # paper_ids=None

    assert stats.total == 0
    conn.close()


# --- (b) queue --is broken surfaces quarantined and nothing else ---

def test_queue_is_broken_surfaces_only_quarantined(tmp_path):
    """`queue --is broken` isolates papers at/above the cap."""
    conn = get_connection(str(tmp_path / "test.db"))
    _insert_paper(conn, 1, error_count=CAP)        # broken (== cap)
    _insert_paper(conn, 2, error_count=CAP + 50)   # broken (well above)
    _insert_paper(conn, 3, error_count=CAP - 1)    # one short of the cap
    _insert_paper(conn, 4, error_count=0)          # healthy

    rows = query_queue(conn, is_prop=["broken"])
    assert sorted(r["id"] for r in rows) == [1, 2]
    assert queue_count(conn, is_prop=["broken"]) == 2
    conn.close()


# --- (c) reset brings a quarantined paper back into selection ---

def test_reset_errors_brings_paper_back_into_process_selection(tmp_path, papers_dir):
    """Zeroing error_count re-admits a quarantined paper to default selection."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    conn.execute("UPDATE papers SET error_count = ? WHERE id = 1", (CAP,))
    conn.commit()

    # Quarantined: not selected.
    process_documents(conn, papers_dir, db_path, workers=1)
    conn_a = get_connection(str(db_path))
    assert conn_a.execute("SELECT has_text FROM papers WHERE id = 1").fetchone()["has_text"] == 0
    conn_a.close()

    # Reset clears the error count.
    reset_errors(conn, [1])
    ec = conn.execute("SELECT error_count FROM papers WHERE id = 1").fetchone()["error_count"]
    assert ec == 0

    # Now it is processed by default selection.
    process_documents(conn, papers_dir, db_path, workers=1)
    conn_b = get_connection(str(db_path))
    assert conn_b.execute("SELECT has_text FROM papers WHERE id = 1").fetchone()["has_text"] == 1
    conn_b.close()
    conn.close()


# --- CLI surface ---

def _index_with_quarantined(tmp_path, monkeypatch):
    """Index DB with one quarantined paper among healthy ones; wire env."""
    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))
    _insert_paper(conn, 1, error_count=CAP, has_text=0)   # quarantined
    _insert_paper(conn, 2, error_count=0, has_text=0)     # healthy
    conn.close()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))
    return db_path


def test_cli_queue_is_broken(tmp_path, monkeypatch):
    """`gantry queue --is broken` is an accepted choice and isolates the pill."""
    _index_with_quarantined(tmp_path, monkeypatch)

    result = CliRunner().invoke(cli, ["queue", "--is", "broken", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["count"] == 1
    assert data["documents"][0]["id"] == 1


def test_cli_retry_ids_resets_quarantined(tmp_path, monkeypatch):
    """`gantry retry --ids` zeroes the error count, un-quarantining the paper."""
    db_path = _index_with_quarantined(tmp_path, monkeypatch)

    result = CliRunner().invoke(cli, ["retry", "--ids", "1", "--json"])

    # The reset zeroes the count; the reprocess attempt on the (missing) PDF then
    # bumps it to 1 — still below the cap, so the paper is no longer quarantined.
    assert result.exit_code in (0, 1, 3), result.output
    conn = get_connection(str(db_path))
    count = conn.execute("SELECT error_count FROM papers WHERE id = 1").fetchone()["error_count"]
    assert count < CAP
    conn.close()
