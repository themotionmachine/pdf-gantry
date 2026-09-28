"""Keyword index follows metadata edits (fts-refresh).

``papers_fts`` is contentless and was written only at extraction time, which
runs before ``enrich``. Titles, authors and abstracts added later never
reached keyword search. Worse, a contentless FTS5 'delete' must be given the
exact values that were indexed; process/ocr/prune passed the CURRENT values,
so re-processing an enriched paper corrupted the posting lists.

The fix: ``contentless_delete=1`` (delete by rowid), one helper that every
writer goes through, and ``gantry fts rebuild`` to convert an existing index.
"""

import json
import logging
import sqlite3

import pytest
from click.testing import CliRunner

from pdf_gantry import fts, openalex
from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.meta import clear_metadata, normalize_metadata, set_metadata
from pdf_gantry.metadata import enrich_documents
from pdf_gantry.process import process_documents
from pdf_gantry.prune import prune_missing
from pdf_gantry.search import fts_search

from .search_helpers import add_paper, new_db

LEGACY_DDL = (
    "CREATE VIRTUAL TABLE papers_fts USING fts5("
    "filename, title, authors, abstract, text_content, "
    "content='', tokenize='porter unicode61')"
)


def _ids(conn, query):
    return [r.id for r in fts_search(conn, query, snippets=False)]


def _integrity_ok(conn):
    # Raises sqlite3.DatabaseError ("database disk image is malformed") on a
    # corrupted index.
    conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('integrity-check')")
    return True


def _make_legacy(conn):
    """Swap papers_fts for the pre-fix table (no contentless_delete)."""
    conn.execute("DROP TABLE papers_fts")
    conn.execute(LEGACY_DDL)
    conn.commit()


@pytest.fixture
def db(tmp_path):
    conn, db_path = new_db(tmp_path)
    add_paper(conn, 1, "a.pdf", text="coastal adaptation policy")
    add_paper(conn, 2, "b.pdf", text="neural language models")
    yield conn, db_path
    conn.close()


@pytest.fixture
def cli_db(db, monkeypatch):
    conn, db_path = db
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(db_path.parent))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(db_path.parent))
    return conn, db_path


def _run(*args):
    return CliRunner().invoke(cli, list(args))


# --- schema ------------------------------------------------------------------


def test_new_db_uses_contentless_delete(db):
    conn, _ = db
    assert fts.has_contentless_delete(conn) is True


def test_legacy_table_detected(db):
    conn, _ = db
    _make_legacy(conn)
    assert fts.has_contentless_delete(conn) is False


# --- metadata writers --------------------------------------------------------


def test_meta_set_title_reaches_keyword_search(cli_db):
    conn, _ = cli_db
    res = _run("meta", "set", "--id", "1", "--title", "Zyxwvut Glacier Studies", "--json")
    assert res.exit_code == 0, res.output
    res = _run("search", "Zyxwvut", "--fts", "--json")
    assert res.exit_code == 0, res.output
    assert [r["id"] for r in json.loads(res.output)["results"]] == [1]
    assert _integrity_ok(conn)


def test_meta_set_replaces_old_title_terms(db):
    conn, _ = db
    set_metadata(conn, 1, title="Firstword Title")
    set_metadata(conn, 1, title="Secondword Title")
    assert _ids(conn, "Firstword") == []
    assert _ids(conn, "Secondword") == [1]
    assert _integrity_ok(conn)


def test_meta_set_authors_and_abstract_indexed(db):
    conn, _ = db
    set_metadata(conn, 2, authors="Quintessa Marbleworth", abstract="hydroponics overview")
    assert _ids(conn, "Marbleworth") == [2]
    assert _ids(conn, "hydroponics") == [2]


def test_meta_clear_removes_title_terms(db):
    conn, _ = db
    set_metadata(conn, 1, title="Vanishingword")
    assert _ids(conn, "Vanishingword") == [1]
    clear_metadata(conn, [1])
    assert _ids(conn, "Vanishingword") == []
    # Text postings survive.
    assert _ids(conn, "coastal") == [1]
    assert _integrity_ok(conn)


def test_meta_normalize_refreshes_authors(db):
    conn, _ = db
    conn.execute("UPDATE papers SET authors = 'Ann Oddsurname; Bo Other' WHERE id = 2")
    conn.commit()
    fts.rebuild(conn)
    normalize_metadata(conn)
    assert _ids(conn, "Oddsurname") == [2]
    assert _integrity_ok(conn)


def test_meta_set_on_paper_without_text_adds_no_row(db):
    """A paper gets an FTS row iff it has extracted text (same rule as rebuild)."""
    conn, _ = db
    conn.execute(
        "INSERT INTO papers (id, path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at) VALUES (3, '/p/c.pdf', 'c.pdf', 'h3', 1, '2026', '2026', '2026')"
    )
    conn.commit()
    set_metadata(conn, 3, title="Untextedword")
    assert _ids(conn, "Untextedword") == []


def test_enrich_title_reaches_keyword_search(db, monkeypatch):
    conn, _ = db
    monkeypatch.setattr(openalex, "fetch_by_doi", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_filename", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_title", lambda *a, **k: {
        "title": "Coastal Enrichedword Adaptation", "authors": ["Pemberly Quarx"],
        "year": 2020, "abstract": "abstractonlyterm", "doi": None, "source_id": "W",
    })
    stats = enrich_documents(conn, paper_ids=[1], rate_limit=0)
    assert stats.matched == 1
    assert _ids(conn, "Enrichedword") == [1]
    assert _ids(conn, "Quarx") == [1]
    assert _ids(conn, "abstractonlyterm") == [1]
    assert _integrity_ok(conn)


# --- extraction writers ------------------------------------------------------


def test_reprocess_enriched_paper_keeps_index_intact(tmp_path, papers_dir, monkeypatch):
    import pdf_gantry.process as process_mod

    db_path = tmp_path / "idx.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)
    pid = conn.execute(
        "SELECT id FROM papers WHERE filename = 'test_climate.pdf'"
    ).fetchone()["id"]

    set_metadata(conn, pid, title="Enrichedtitleword", authors="Zed Authorname",
                 abstract="abstractterm")

    monkeypatch.setattr(
        process_mod, "extract_text_pymupdf",
        lambda _p: ("Replacementword body text.", "## Replacementword\n\nbody text."),
    )
    process_documents(conn, papers_dir, db_path, paper_ids=[pid], workers=1)

    assert _integrity_ok(conn)
    assert _ids(conn, "coastal") == []  # old text gone ("climate" is in the filename)
    assert _ids(conn, "Replacementword") == [pid]
    assert _ids(conn, "Enrichedtitleword") == [pid]
    assert _ids(conn, "Authorname") == [pid]
    conn.close()


def test_prune_deletes_by_rowid_after_enrichment(tmp_path, papers_dir):
    db_path = tmp_path / "idx.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)
    pid = conn.execute(
        "SELECT id FROM papers WHERE filename = 'ml_nlp_paper.pdf'"
    ).fetchone()["id"]
    set_metadata(conn, pid, title="Prunedtitleword")
    (papers_dir / "ml_nlp_paper.pdf").unlink()

    prune_missing(conn, papers_dir)

    assert _integrity_ok(conn)
    assert _ids(conn, "neural") == []
    assert _ids(conn, "Prunedtitleword") == []
    assert conn.execute(
        "SELECT COUNT(*) FROM papers_fts WHERE rowid = ?", (pid,)
    ).fetchone()[0] == 0
    conn.close()


# --- rebuild ------------------------------------------------------------------


def test_rebuild_converts_legacy_table_and_indexes_metadata(db):
    conn, _ = db
    _make_legacy(conn)
    # The legacy table holds text only; metadata was set afterwards.
    for pid, text in ((1, "coastal adaptation policy"), (2, "neural language models")):
        conn.execute(
            "INSERT INTO papers_fts(rowid, filename, title, authors, abstract, text_content) "
            "VALUES (?, '', '', '', '', ?)", (pid, text),
        )
    conn.execute("UPDATE papers SET title = 'Laterword Title' WHERE id = 2")
    conn.commit()
    assert _ids(conn, "Laterword") == []

    report = fts.rebuild(conn)

    assert report["rows"] == 2
    assert report["previous_contentless_delete"] is False
    assert report["contentless_delete"] is True
    assert "elapsed_seconds" in report
    assert fts.has_contentless_delete(conn)
    assert _ids(conn, "Laterword") == [2]
    assert _ids(conn, "coastal") == [1]
    assert _integrity_ok(conn)
    # No stray temp table left behind.
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert fts.TEMP_TABLE not in names


def test_rebuild_is_idempotent(db):
    conn, _ = db
    set_metadata(conn, 1, title="Stableword")
    first = fts.rebuild(conn)
    before = (_ids(conn, "Stableword"), _ids(conn, "neural"))
    second = fts.rebuild(conn)
    assert first["rows"] == second["rows"] == 2
    assert (_ids(conn, "Stableword"), _ids(conn, "neural")) == before == ([1], [2])
    assert _integrity_ok(conn)


def test_rebuild_dry_run_writes_nothing(db):
    conn, _ = db
    _make_legacy(conn)
    report = fts.rebuild(conn, dry_run=True)
    assert report["dry_run"] is True
    assert report["rows"] == 2
    assert fts.has_contentless_delete(conn) is False


def test_search_works_during_rebuild_until_swap(tmp_path):
    """A reader on another connection keeps seeing the old index until commit."""
    conn, db_path = new_db(tmp_path)
    add_paper(conn, 1, "a.pdf", text="coastal adaptation")
    conn.execute("UPDATE papers SET title = 'Swapword' WHERE id = 1")
    conn.commit()
    reader = get_connection(str(db_path))
    seen = {}

    def during_build():
        seen["coastal"] = _ids(reader, "coastal")
        seen["Swapword"] = _ids(reader, "Swapword")

    fts.rebuild(conn, _before_commit=during_build)
    assert seen == {"coastal": [1], "Swapword": []}
    assert _ids(reader, "Swapword") == [1]
    reader.close()
    conn.close()


def test_rebuild_rolls_back_on_failure(db):
    conn, _ = db
    _make_legacy(conn)

    def boom():
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        fts.rebuild(conn, _before_commit=boom)
    assert fts.has_contentless_delete(conn) is False
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert "papers_fts" in names and fts.TEMP_TABLE not in names


def test_fts_rebuild_cli_json(cli_db):
    conn, _ = cli_db
    _make_legacy(conn)
    conn.close()
    res = _run("fts", "rebuild", "--json")
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["rows"] == 2
    assert data["contentless_delete"] is True
    res = _run("status", "--json")
    assert json.loads(res.output)["fts_contentless_delete"] is True


def test_fts_rebuild_cli_dry_run(cli_db):
    conn, _ = cli_db
    _make_legacy(conn)
    conn.close()
    res = _run("fts", "rebuild", "--dry-run", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["dry_run"] is True
    res = _run("status", "--json")
    assert json.loads(res.output)["fts_contentless_delete"] is False


# --- legacy fallback ----------------------------------------------------------


def test_legacy_metadata_write_skips_index_and_warns_once(db, caplog):
    conn, _ = db
    _make_legacy(conn)
    conn.execute(
        "INSERT INTO papers_fts(rowid, filename, title, authors, abstract, text_content) "
        "VALUES (1, 'a.pdf', '', '', '', 'coastal adaptation policy')"
    )
    conn.commit()
    fts._warned = False
    with caplog.at_level(logging.WARNING, logger="pdf_gantry.fts"):
        set_metadata(conn, 1, title="Legacyword")
        set_metadata(conn, 1, title="Legacyword Again")
    # Metadata writers can't safely delete on a legacy table: index untouched.
    assert _ids(conn, "Legacyword") == []
    assert _ids(conn, "coastal") == [1]
    warnings = [r for r in caplog.records if "gantry fts rebuild" in r.getMessage()]
    assert len(warnings) == 1
    assert _integrity_ok(conn)


def test_legacy_refresh_with_old_text_replaces_postings(db):
    """process/ocr on a legacy table keep today's delete-with-values path."""
    conn, _ = db
    _make_legacy(conn)
    conn.execute(
        "INSERT INTO papers_fts(rowid, filename, title, authors, abstract, text_content) "
        "VALUES (1, 'a.pdf', '', '', '', 'coastal adaptation policy')"
    )
    conn.execute("UPDATE paper_text SET raw_text = 'fresh replacement' WHERE paper_id = 1")
    conn.commit()
    fts.refresh_row(conn, 1, old_text="coastal adaptation policy")
    conn.commit()
    assert _ids(conn, "coastal") == []
    assert _ids(conn, "fresh") == [1]
    assert _integrity_ok(conn)


def test_legacy_delete_row_uses_current_values(db):
    conn, _ = db
    _make_legacy(conn)
    conn.execute(
        "INSERT INTO papers_fts(rowid, filename, title, authors, abstract, text_content) "
        "VALUES (2, 'b.pdf', '', '', '', 'neural language models')"
    )
    conn.commit()
    fts.delete_row(conn, 2)
    conn.commit()
    assert _ids(conn, "neural") == []
    assert _integrity_ok(conn)


def test_delete_row_missing_is_noop(db):
    conn, _ = db
    fts.delete_row(conn, 999)
    conn.commit()
    assert _integrity_ok(conn)


def test_status_reports_contentless_delete(cli_db):
    res = _run("status", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["fts_contentless_delete"] is True


def test_integrity_check_detects_corruption_on_legacy(db):
    """Guard for the harness: a wrong-values delete really is detected."""
    conn, _ = db
    _make_legacy(conn)
    conn.execute(
        "INSERT INTO papers_fts(rowid, filename, title, authors, abstract, text_content) "
        "VALUES (1, 'a.pdf', '', '', '', 'coastal adaptation policy')"
    )
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(
            "INSERT INTO papers_fts(papers_fts, rowid, filename, title, authors, abstract, "
            "text_content) VALUES('delete', 1, 'a.pdf', 'Wrong Title', '', '', 'coastal')"
        )
        _integrity_ok(conn)


def test_status_text_hints_rebuild_on_legacy(cli_db):
    conn, _ = cli_db
    _make_legacy(conn)
    res = _run("status")
    assert res.exit_code == 0, res.output
    assert "gantry fts rebuild" in res.output
