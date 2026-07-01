"""Tests for pruning ghost entries from the database."""

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents
from pdf_gantry.prune import prune_missing
from pdf_gantry.search import fts_search

# ---------------------------------------------------------------------------
# Hostile-input cases the original author never pictured
# ---------------------------------------------------------------------------

def test_prune_nonexistent_papers_dir_raises(tmp_path, papers_dir):
    """prune_missing raises FileNotFoundError when papers_dir doesn't exist.

    The silent killer: if papers_dir is absent (drive unmounted, iCloud sync
    stalled, misconfigured path), every file_path.exists() call returns False.
    All papers land in missing[], the loop runs, conn.commit() seals it, and
    the entire index is silently destroyed.  A 3-line guard prevents this.
    """
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    count_before = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert count_before == 2  # sanity: two papers ingested

    vanished_dir = tmp_path / "drive_not_mounted"
    assert not vanished_dir.exists()  # must not exist

    with pytest.raises(FileNotFoundError, match="papers_dir"):
        prune_missing(conn, vanished_dir)

    # The DB must be completely intact — no records lost.
    count_after = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert count_after == 2, (
        f"prune_missing deleted {count_before - count_after} record(s) "
        "because it treated every file as 'missing' in a nonexistent directory"
    )
    conn.close()


def test_prune_dry_run_nonexistent_papers_dir_also_raises(tmp_path, papers_dir):
    """dry_run=True offers no protection against the absent-directory bug.

    Without an early guard, dry_run still reports every paper as 'missing'
    (pruned==2), even though it doesn't commit the deletes.  The caller gets
    a false alarm that could cascade into incorrect automation decisions.
    The fix: raise before scanning, regardless of dry_run.
    """
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    vanished_dir = tmp_path / "not_here"
    assert not vanished_dir.exists()

    with pytest.raises(FileNotFoundError, match="papers_dir"):
        prune_missing(conn, vanished_dir, dry_run=True)

    conn.close()


def test_prune_empty_db_noop(tmp_path):
    """prune_missing on an empty database returns zero stats and is a no-op.

    Characterization lock: empty DB is a valid starting state (e.g. first run
    before any ingest).  prune should return cleanly with pruned==0.
    """
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()

    stats = prune_missing(conn, papers_dir)

    assert stats["pruned"] == 0
    assert stats["remaining"] == 0
    assert stats["pruned_files"] == []
    conn.close()


def test_prune_removes_missing_files(tmp_path, papers_dir):
    """Prune removes DB entries for files no longer on disk."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    # Delete a file from disk
    (papers_dir / "ml_nlp_paper.pdf").unlink()

    stats = prune_missing(conn, papers_dir)
    assert stats["pruned"] == 1
    assert stats["remaining"] > 0

    # Verify it's gone from DB
    row = conn.execute("SELECT COUNT(*) FROM papers WHERE filename = 'ml_nlp_paper.pdf'").fetchone()
    assert row[0] == 0
    conn.close()


def test_prune_noop_when_all_present(tmp_path, papers_dir):
    """Prune is a no-op when all files exist."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    stats = prune_missing(conn, papers_dir)
    assert stats["pruned"] == 0
    assert stats["remaining"] == 2
    conn.close()


def test_prune_dry_run(tmp_path, papers_dir):
    """Dry run reports but doesn't delete."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    (papers_dir / "ml_nlp_paper.pdf").unlink()

    stats = prune_missing(conn, papers_dir, dry_run=True)
    assert stats["pruned"] == 1

    # File should still be in DB
    row = conn.execute("SELECT COUNT(*) FROM papers WHERE filename = 'ml_nlp_paper.pdf'").fetchone()
    assert row[0] == 1
    conn.close()


def test_prune_cleans_related_tables(tmp_path, papers_dir):
    """Prune removes chunks, text, FTS, and vec entries for missing files."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    # Get the paper ID before deleting
    conn2 = get_connection(str(db_path))
    paper = conn2.execute("SELECT id FROM papers WHERE filename = 'ml_nlp_paper.pdf'").fetchone()
    paper_id = paper["id"]

    # Verify related data exists
    assert conn2.execute(
        "SELECT COUNT(*) FROM paper_text WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0] > 0
    assert conn2.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (paper_id,)
    ).fetchone()[0] > 0

    # Delete file and prune
    (papers_dir / "ml_nlp_paper.pdf").unlink()
    prune_missing(conn2, papers_dir)

    # All related data should be gone
    assert conn2.execute(
        "SELECT COUNT(*) FROM paper_text WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0] == 0
    assert conn2.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (paper_id,)
    ).fetchone()[0] == 0
    conn2.close()
    conn.close()


def test_prune_returns_filenames(tmp_path, papers_dir):
    """Prune returns the list of pruned filenames."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    (papers_dir / "ml_nlp_paper.pdf").unlink()

    stats = prune_missing(conn, papers_dir)
    assert "pruned_files" in stats
    assert "ml_nlp_paper.pdf" in stats["pruned_files"]
    conn.close()


# ---------------------------------------------------------------------------
# FTS orphan audit
# test_prune_cleans_related_tables's docstring says "FTS" is cleaned but
# never asserts it.  The two tests below lock that behavior by proving:
#   (1) papers_fts has zero postings for the pruned rowid after prune, and
#   (2) fts_search returns zero results for terms unique to the pruned paper
#       (the full user path: process → delete → prune → search → no ghost).
# ---------------------------------------------------------------------------


def test_prune_cleans_fts_postings(tmp_path, papers_dir):
    """Prune removes papers_fts postings for deleted papers — no orphaned entries.

    The contentless-FTS5 'delete' command in prune_missing
    must pass the exact original content to correctly remove all term postings.
    This test verifies two things:
      - the FTS rowid is absent after prune (papers_fts_content is clean), AND
      - a direct FTS MATCH query (no JOIN with papers) returns 0 hits for a term
        that is unique to the pruned paper (inverted index is clean too).
    If either assertion fails, orphaned postings have accumulated and could
    affect BM25 ranking statistics across the live corpus.
    """
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    # Re-open for a clean connection so we see the committed state
    conn.close()
    conn = get_connection(str(db_path))

    paper = conn.execute(
        "SELECT id FROM papers WHERE filename = 'ml_nlp_paper.pdf'"
    ).fetchone()
    paper_id = paper["id"]

    # Sanity: FTS has the rowid before pruning
    fts_before = conn.execute(
        "SELECT rowid FROM papers_fts WHERE rowid = ?", (paper_id,)
    ).fetchone()
    assert fts_before is not None, "papers_fts should have rowid before prune"

    # 'neural' is unique to ml_nlp_paper.pdf in the fixture corpus
    raw_match_before = conn.execute(
        "SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH 'neural'"
    ).fetchone()[0]
    assert raw_match_before >= 1, "FTS should match 'neural' before prune"

    # Delete file and prune
    (papers_dir / "ml_nlp_paper.pdf").unlink()
    prune_missing(conn, papers_dir)

    # FTS rowid must be gone — no ghost rowid in papers_fts_content
    fts_after = conn.execute(
        "SELECT rowid FROM papers_fts WHERE rowid = ?", (paper_id,)
    ).fetchone()
    assert fts_after is None, (
        f"papers_fts still has rowid={paper_id} after prune — "
        "orphaned entry survives in FTS shadow table"
    )

    # Direct FTS MATCH query (bypasses the papers JOIN) must also return 0 —
    # confirming the inverted index itself is clean, not just filtered by JOIN
    raw_match_after = conn.execute(
        "SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH 'neural'"
    ).fetchone()[0]
    assert raw_match_after == 0, (
        f"Orphaned FTS postings: 'neural' still matches {raw_match_after} "
        "FTS entries after prune — inverted index is dirty"
    )

    conn.close()


def test_no_ghost_fts_hits_after_prune(tmp_path, papers_dir):
    """User path: process → file vanishes → prune → search returns 0 hits.

    User-visible: after prune removes a processed paper,
    fts_search must return zero results for terms that were unique to that paper.
    A ghost hit here means the user (or agent) follows a search result to a
    paper that no longer exists — a dead end that wastes tokens and context.

    'neural' appears in ml_nlp_paper.pdf and not in test_climate.pdf.
    """
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)

    conn.close()
    conn = get_connection(str(db_path))

    # Sanity: paper is findable before pruning
    results_before = fts_search(conn, "neural")
    assert any(r.filename == "ml_nlp_paper.pdf" for r in results_before), (
        "ml_nlp_paper.pdf should appear in search results before prune"
    )

    # Delete file and prune
    (papers_dir / "ml_nlp_paper.pdf").unlink()
    prune_missing(conn, papers_dir)

    # The pruned paper must not appear in search results — no ghost
    results_after = fts_search(conn, "neural")
    ghost_hits = [r for r in results_after if r.filename == "ml_nlp_paper.pdf"]
    assert ghost_hits == [], (
        f"Ghost result! fts_search returned {ghost_hits} for a pruned paper — "
        "the user's search would lead to a dead end"
    )

    conn.close()
