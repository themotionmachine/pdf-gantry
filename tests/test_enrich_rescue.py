"""Enrich title tier: layout query, gated rescue on --retry-misses, titled_ids."""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry import openalex
from pdf_gantry.cli import cli
from pdf_gantry.metadata import enrich_documents

from .search_helpers import add_paper, new_db, point_cli_at

TEXT = "Masthead Line\nZebrafish Governance Systems in Practice\nBody about zebrafish."
GOOD = {"title": "Zebrafish Governance Systems in Practice", "authors": ["A B"], "year": 2021,
        "abstract": "x", "doi": "10.1/z", "source_id": "W1"}
WRONG_SUBJECT = {**GOOD, "title": "Practice Systems of Zebrafish Governance Reconsidered Anew"}
UNRELATED = {**GOOD, "title": "Walrus Migration in Polar Seas"}


@pytest.fixture
def db(tmp_path):
    conn, _ = new_db(tmp_path)
    return conn


def _paper(conn, layout="Zebrafish Governance Systems in Practice",
           source="none:openalex", enriched="2026-01-01T00:00:00+00:00", pid=1):
    add_paper(conn, pid, f"{pid}.pdf", text=TEXT)
    conn.execute("UPDATE papers SET title_from_layout = ?, metadata_source = ?, "
                 "metadata_enriched_at = ? WHERE id = ?", (layout, source, enriched, pid))
    conn.commit()


def _provider(monkeypatch, result):
    queries = []
    monkeypatch.setattr(openalex, "fetch_by_doi", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_filename", lambda *a, **k: None)

    def by_title(title, **k):
        queries.append(title)
        return result

    monkeypatch.setattr(openalex, "fetch_by_title", by_title)
    return queries


def _row(conn, pid=1):
    return conn.execute("SELECT * FROM papers WHERE id = ?", (pid,)).fetchone()


def test_title_tier_queries_with_layout_title(db, monkeypatch):
    _paper(db, source=None, enriched=None)
    queries = _provider(monkeypatch, None)
    enrich_documents(db, rate_limit=0)
    assert queries == ["Zebrafish Governance Systems in Practice"]


def test_title_tier_falls_back_to_text_guess(db, monkeypatch):
    _paper(db, layout=None, source=None, enriched=None)
    queries = _provider(monkeypatch, None)
    enrich_documents(db, rate_limit=0)
    assert len(queries) == 1 and "Zebrafish" in queries[0]


def test_default_enrich_still_accepts_and_flags_a_bad_hit(db, monkeypatch):
    _paper(db, source=None, enriched=None)
    _provider(monkeypatch, UNRELATED)
    stats = enrich_documents(db, rate_limit=0)
    assert stats.matched == 1 and stats.suspect == 1
    assert _row(db)["title"] == UNRELATED["title"]


def test_rescue_accepts_a_contained_similar_title(db, monkeypatch):
    _paper(db)
    _provider(monkeypatch, GOOD)
    stats = enrich_documents(db, rate_limit=0, retry_misses=True)
    row = _row(db)
    assert row["title"] == GOOD["title"] and row["metadata_source"] == "openalex_title"
    assert row["metadata_suspect"] == 0
    assert stats.titled_ids == [1] and stats.rejected_title == 0


@pytest.mark.parametrize("hit", [UNRELATED, WRONG_SUBJECT])
def test_rescue_rejects_and_writes_nothing(db, monkeypatch, hit):
    _paper(db)
    before = dict(_row(db))
    _provider(monkeypatch, hit)
    stats = enrich_documents(db, rate_limit=0, retry_misses=True)
    assert dict(_row(db)) == before
    assert stats.rejected_title == 1 and stats.no_match == 1 and stats.titled_ids == []


def test_rescue_rejects_without_a_layout_title(db, monkeypatch):
    _paper(db, layout=None)
    _provider(monkeypatch, GOOD)
    assert enrich_documents(db, rate_limit=0, retry_misses=True).rejected_title == 1
    assert _row(db)["title"] is None


def test_rejected_never_attempted_paper_is_stamped_a_miss(db, monkeypatch):
    _paper(db, source=None, enriched=None)
    _provider(monkeypatch, UNRELATED)
    enrich_documents(db, rate_limit=0, retry_misses=True)
    assert _row(db)["metadata_source"] == "none:openalex" and _row(db)["title"] is None


def test_doi_tier_is_not_gated(db, monkeypatch):
    _paper(db, layout=None)
    db.execute("UPDATE papers SET doi = '10.1/z' WHERE id = 1")
    db.commit()
    _provider(monkeypatch, None)
    monkeypatch.setattr(openalex, "fetch_by_doi", lambda *a, **k: UNRELATED)
    stats = enrich_documents(db, rate_limit=0, retry_misses=True)
    assert stats.matched == 1 and _row(db)["title"] == UNRELATED["title"]


def test_cli_retry_misses_applies_the_gate_and_reports_titled_ids(tmp_path, monkeypatch):
    conn, db_path = new_db(tmp_path)
    _paper(conn, pid=1)
    _paper(conn, layout="Some Other Layout Heading Entirely", pid=2)
    point_cli_at(tmp_path, monkeypatch, db_path)
    _provider(monkeypatch, GOOD)
    out = CliRunner().invoke(cli, ["enrich", "--retry-misses", "--json"]).output
    report = json.loads(out)
    assert report["titled_ids"] == [1] and report["rejected_title"] == 1
