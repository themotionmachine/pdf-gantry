"""`gantry titles backfill`, new-paper population, and the FTS title fallback."""

import json

import pymupdf
import pytest
from click.testing import CliRunner

from pdf_gantry import layout
from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents

BODY = ("The ordinary body text of this paper keeps running along the page so that "
        "extraction and classification both see a plain digital document here.")


def write_pdf(path, title="Zebrafish Governance in Northern Cities"):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((60, 120), title, fontsize=20)
    for i in range(30):
        page.insert_text((60, 420 + i * 12), BODY[:80], fontsize=10)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def lib(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    write_pdf(papers / "a.pdf")
    write_pdf(papers / "b.pdf", "Quokka Migration and Municipal Budgets")
    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers)
    return conn, papers, db_path


@pytest.fixture
def unstamped(lib):
    """Processed papers from before this feature: no layout attempt recorded."""
    conn, papers, db_path = lib
    process_documents(conn, papers, db_path, workers=1)
    conn.execute("UPDATE papers SET title_from_layout = NULL, "
                 "title_layout_confidence = NULL, title_layout_at = NULL")
    conn.commit()
    return conn, papers, db_path


def _config(tmp_path, papers, db_path, monkeypatch):
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(db_path.parent))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(papers))


def _row(conn, name):
    return conn.execute("SELECT * FROM papers WHERE filename = ?", (name,)).fetchone()


def test_backfill_writes_only_layout_columns(unstamped):
    conn, papers, _ = unstamped
    before = dict(_row(conn, "a.pdf"))
    report = layout.backfill_titles(conn, papers)
    after = dict(_row(conn, "a.pdf"))
    assert report["titled"] == 2 and report["written"] == 2
    assert after["title_from_layout"] == "Zebrafish Governance in Northern Cities"
    assert after["title_layout_confidence"] == "high"
    assert after["title_layout_at"]
    changed = {k for k in after if after[k] != before[k]}
    assert changed == {"title_from_layout", "title_layout_confidence", "title_layout_at"}
    assert after["title"] is None


def test_backfill_dry_run_writes_nothing(unstamped):
    conn, papers, _ = unstamped
    report = layout.backfill_titles(conn, papers, dry_run=True)
    assert report["titled"] == 2 and report["written"] == 0
    assert _row(conn, "a.pdf")["title_layout_at"] is None


def test_no_result_is_stamped_and_not_retried(unstamped, monkeypatch):
    conn, papers, _ = unstamped
    monkeypatch.setattr(layout, "extract_layout_title",
                        lambda *a, **k: layout.LayoutTitle())
    assert layout.backfill_titles(conn, papers)["empty"] == 2
    assert _row(conn, "a.pdf")["title_layout_at"]
    assert layout.backfill_titles(conn, papers)["selected"] == 0


def test_missing_file_is_an_error_and_stays_pending(unstamped):
    conn, papers, _ = unstamped
    (papers / "a.pdf").unlink()
    report = layout.backfill_titles(conn, papers)
    assert [e["filename"] for e in report["errors"]] == ["a.pdf"]
    assert _row(conn, "a.pdf")["title_layout_at"] is None
    assert _row(conn, "b.pdf")["title_layout_at"]


def test_force_ids_limit(unstamped):
    conn, papers, _ = unstamped
    a = _row(conn, "a.pdf")["id"]
    assert layout.backfill_titles(conn, papers, limit=1)["selected"] == 1
    assert layout.backfill_titles(conn, papers, limit=None)["selected"] == 1
    assert layout.backfill_titles(conn, papers)["selected"] == 0
    assert layout.backfill_titles(conn, papers, paper_ids=[a])["selected"] == 0
    assert layout.backfill_titles(conn, papers, paper_ids=[a], force=True)["selected"] == 1


def test_boilerplate_computed_once_per_run(unstamped, monkeypatch):
    conn, papers, _ = unstamped
    calls = []
    real = layout.corpus_boilerplate
    monkeypatch.setattr(layout, "corpus_boilerplate",
                        lambda c: calls.append(1) or real(c))
    layout.backfill_titles(conn, papers)
    assert len(calls) == 1


def test_process_populates_in_parent_and_keyword_search_finds_it(lib):
    conn, papers, db_path = lib
    stats = process_documents(conn, papers, db_path, workers=1)
    assert stats.succeeded == 2 and stats.layout_titled == 2
    assert _row(conn, "a.pdf")["title_from_layout"].startswith("Zebrafish")


def test_bad_pdf_does_not_fail_processing(lib, monkeypatch):
    conn, papers, db_path = lib

    def boom(*a, **k):
        raise RuntimeError("layout exploded")

    monkeypatch.setattr(layout, "backfill_titles", boom)
    stats = process_documents(conn, papers, db_path, workers=1)
    assert stats.succeeded == 2 and stats.failed == 0


def test_changed_file_resets_the_layout_attempt(unstamped):
    conn, papers, _ = unstamped
    layout.backfill_titles(conn, papers)
    write_pdf(papers / "a.pdf", "A Completely Different Paper About Otters")
    ingest_directory(conn, papers)
    row = _row(conn, "a.pdf")
    assert row["title_from_layout"] is None and row["title_layout_at"] is None


def test_fts_indexes_layout_title_only_when_no_metadata_title(unstamped):
    conn, papers, _ = unstamped
    conn.execute("UPDATE papers SET title_from_layout = 'Capybara Fiscal Federalism', "
                 "title_layout_at = 'x' WHERE filename = 'a.pdf'")
    from pdf_gantry.fts import rebuild, refresh_row
    pid = _row(conn, "a.pdf")["id"]
    refresh_row(conn, pid)
    conn.commit()
    rows = conn.execute("SELECT rowid FROM papers_fts WHERE papers_fts MATCH "
                        "'title : capybara'").fetchall()
    assert [r[0] for r in rows] == [pid]

    conn.execute("UPDATE papers SET title = 'Verified Metadata Title' WHERE id = ?", (pid,))
    refresh_row(conn, pid)
    conn.commit()
    assert not conn.execute("SELECT rowid FROM papers_fts WHERE papers_fts MATCH "
                            "'title : capybara'").fetchall()

    conn.execute("UPDATE papers SET title = NULL WHERE id = ?", (pid,))
    conn.commit()
    rebuild(conn)
    assert conn.execute("SELECT rowid FROM papers_fts WHERE papers_fts MATCH "
                        "'title : capybara'").fetchall()


def test_backfill_refreshes_fts_for_untitled_papers(unstamped):
    conn, papers, _ = unstamped
    layout.backfill_titles(conn, papers)
    assert conn.execute("SELECT rowid FROM papers_fts WHERE papers_fts MATCH "
                        "'title : zebrafish'").fetchall()


def test_cli_titles_backfill_json_and_status(unstamped, tmp_path, monkeypatch):
    conn, papers, db_path = unstamped
    _config(tmp_path, papers, db_path, monkeypatch)
    conn.close()
    runner = CliRunner()
    before = json.loads(runner.invoke(cli, ["status", "--json"]).output)
    assert before["needs_layout_title"] == 2
    assert before["layout_titled"] == 0 and before["display_untitled"] == 2

    dry = runner.invoke(cli, ["titles", "backfill", "--dry-run", "--json"])
    assert dry.exit_code == 0, dry.output
    assert json.loads(dry.output)["written"] == 0

    result = runner.invoke(cli, ["titles", "backfill", "--json", "--limit", "1"])
    assert json.loads(result.output)["written"] == 1
    result = runner.invoke(cli, ["titles", "backfill"])
    assert result.exit_code == 0 and "1 written" in result.output

    after = json.loads(runner.invoke(cli, ["status", "--json"]).output)
    assert after["needs_layout_title"] == 0
    assert after["layout_titled"] == 2 and after["display_untitled"] == 0


def test_cli_ids_reports_not_found(unstamped, tmp_path, monkeypatch):
    conn, papers, db_path = unstamped
    _config(tmp_path, papers, db_path, monkeypatch)
    conn.close()
    result = CliRunner().invoke(cli, ["titles", "backfill", "--ids", "99999", "--json"])
    assert json.loads(result.output)["not_found"] == [99999]
