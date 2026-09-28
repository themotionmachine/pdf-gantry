"""`gantry meta set|clear|normalize` -- hand metadata edits done in code.

Agents made 16 raw-SQL writes to set metadata by hand. Those writes produced
`;`-separated author strings (B7), `YYYY-MM-DD HH:MM:SS` timestamps next to
ISO ones, and orphaned abstracts left behind when a wrong match was cleared
without its abstract (B8). These verbs make the same edits in tested code.
"""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.meta import (
    clear_metadata,
    normalize_authors,
    normalize_metadata,
    normalize_timestamp,
    set_metadata,
)
from pdf_gantry.metadata import enrich_documents

ISO_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$"


def _seed(conn, paper_id, **cols):
    base = {
        "id": paper_id, "path": f"/p/p{paper_id}.pdf", "filename": f"p{paper_id}.pdf",
        "file_hash": f"h{paper_id}", "file_size": 1, "file_modified": "2026-01-01",
        "indexed_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    base.update(cols)
    keys = ", ".join(base)
    marks = ", ".join("?" * len(base))
    conn.execute(f"INSERT INTO papers ({keys}) VALUES ({marks})", list(base.values()))
    conn.commit()


def _row(conn, paper_id):
    return dict(conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone())


@pytest.fixture
def db(tmp_path):
    conn = get_connection(str(tmp_path / "index.db"))
    yield conn
    conn.close()


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))
    yield conn
    conn.close()


def _run(*args):
    return CliRunner().invoke(cli, list(args))


# --- normalize_authors ------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    ("A One; B Two", ["A One", "B Two"]),
    ("A One;B Two; ", ["A One", "B Two"]),
    ('["A One", "B Two"]', ["A One", "B Two"]),
    (["A One", " B Two "], ["A One", "B Two"]),
    ("Kai Xiang Teo", ["Kai Xiang Teo"]),
    ("", []),
    (None, None),
])
def test_normalize_authors(value, expected):
    assert normalize_authors(value) == expected


def test_normalize_timestamp():
    assert normalize_timestamp("2026-08-19 19:13:09") == "2026-08-19T19:13:09+00:00"
    iso = "2026-09-10T19:03:38.585826+00:00"
    assert normalize_timestamp(iso) == iso
    assert normalize_timestamp(None) is None


# --- set --------------------------------------------------------------------


def test_set_writes_manual_source_json_authors_iso(db):
    _seed(db, 1, metadata_suspect=1, metadata_verify_score=0.1)
    result = set_metadata(db, 1, title="Mind Games", authors="Ada L; Bo K",
                          year=2021, citekey="ada2021", by="brev")
    row = _row(db, 1)
    assert row["title"] == "Mind Games"
    assert json.loads(row["authors"]) == ["Ada L", "Bo K"]
    assert row["year"] == 2021
    assert row["citekey"] == "ada2021"
    assert row["citekey_source"] == "manual:brev"
    assert row["metadata_source"] == "manual:brev"
    assert row["metadata_suspect"] == 0
    import re
    assert re.match(ISO_RE, row["metadata_enriched_at"])
    assert re.match(ISO_RE, row["updated_at"])
    assert result["changes"]["title"] == {"old": None, "new": "Mind Games"}


def test_set_leaves_unspecified_fields_alone(db):
    _seed(db, 1, title="Old", abstract="Keep me", year=1999)
    set_metadata(db, 1, year=2000, by="x")
    row = _row(db, 1)
    assert row["title"] == "Old" and row["abstract"] == "Keep me" and row["year"] == 2000


def test_set_dry_run_writes_nothing(db):
    _seed(db, 1)
    result = set_metadata(db, 1, title="T", by="x", dry_run=True)
    assert result["changes"]["title"]["new"] == "T"
    assert _row(db, 1)["title"] is None


def test_set_normalizes_doi_url(db):
    _seed(db, 1)
    set_metadata(db, 1, doi="https://doi.org/10.1234/ABC", by="x")
    assert _row(db, 1)["doi"] == "10.1234/ABC"


def test_set_requires_a_field(db):
    _seed(db, 1)
    with pytest.raises(ValueError):
        set_metadata(db, 1, by="x")


def test_set_unknown_id_raises(db):
    with pytest.raises(LookupError):
        set_metadata(db, 42, title="T", by="x")


def test_manual_metadata_survives_enrich(db, monkeypatch):
    from pdf_gantry import openalex
    _seed(db, 1)
    set_metadata(db, 1, title="Hand Title", by="x")
    monkeypatch.setattr(openalex, "fetch_by_filename", lambda *a, **k: {
        "title": "Wrong", "authors": [], "year": 1, "abstract": None,
        "doi": None, "source_id": "W"})
    enrich_documents(db, paper_ids=[1])
    assert _row(db, 1)["title"] == "Hand Title"


def test_cli_meta_set_json(cli_env):
    _seed(cli_env, 1)
    r = _run("meta", "set", "--id", "1", "--title", "T", "--authors", '["A", "B"]',
             "--by", "ryan", "--json")
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["id"] == 1
    assert data["metadata_source"] == "manual:ryan"
    assert data["changes"]["authors"]["new"] == ["A", "B"]
    assert data["dry_run"] is False


def test_cli_meta_set_missing_id_exit_2(cli_env):
    r = _run("meta", "set", "--id", "9", "--title", "T", "--json")
    assert r.exit_code == 2
    assert json.loads(r.output)["not_found"] == [9]


def test_cli_meta_set_no_fields_exit_1(cli_env):
    _seed(cli_env, 1)
    r = _run("meta", "set", "--id", "1", "--json")
    assert r.exit_code == 1
    assert "error" in json.loads(r.output)


# --- clear ------------------------------------------------------------------


def _seed_matched(conn, paper_id):
    _seed(conn, paper_id, title="Wrong", authors='["X"]', year=2001, doi="10.1/w",
          abstract="Daily Show abstract", semantic_scholar_id="ss",
          metadata_source="openalex_title",
          metadata_enriched_at="2026-09-10T00:00:00+00:00",
          metadata_suspect=1, metadata_verify_score=0.2,
          metadata_verified_at="2026-09-10T00:00:00+00:00",
          citekey="keep2001", citekey_source="bib")


def test_clear_nulls_metadata_and_resets_state(db):
    _seed_matched(db, 1)
    clear_metadata(db, [1])
    row = _row(db, 1)
    for col in ("title", "authors", "year", "doi", "abstract", "semantic_scholar_id",
                "metadata_source", "metadata_enriched_at", "metadata_verify_score",
                "metadata_verified_at"):
        assert row[col] is None, col
    assert row["metadata_suspect"] == 0
    assert row["citekey"] == "keep2001"


def test_clear_makes_paper_eligible_for_default_enrich(db):
    from pdf_gantry.metadata import default_enrich_ids
    _seed_matched(db, 1)
    assert default_enrich_ids(db) == []
    clear_metadata(db, [1])
    assert default_enrich_ids(db) == [1]


def test_clear_citekey_only_when_asked(db):
    _seed_matched(db, 1)
    clear_metadata(db, [1], citekey=True)
    row = _row(db, 1)
    assert row["citekey"] is None and row["citekey_source"] is None


def test_clear_dry_run(db):
    _seed_matched(db, 1)
    result = clear_metadata(db, [1, 7], dry_run=True)
    assert result["cleared"] == [1]
    assert result["not_found"] == [7]
    assert _row(db, 1)["title"] == "Wrong"


def test_cli_meta_clear_json(cli_env):
    _seed_matched(cli_env, 1)
    r = _run("meta", "clear", "--ids", "1,5", "--json")
    assert r.exit_code == 3, r.output  # partial: 5 not found
    data = json.loads(r.output)
    assert data["cleared"] == [1] and data["not_found"] == [5]
    assert "abstract" in data["fields"]


# --- normalize --------------------------------------------------------------


def _seed_messy(conn):
    _seed(conn, 1, authors="Ann A; Ben B", title="T1")
    _seed(conn, 2, authors='["Already", "Json"]', title="T2")
    _seed(conn, 3, title="T3", metadata_enriched_at="2026-08-19 19:13:09",
          updated_at="2026-08-19 19:13:09", metadata_verified_at="2026-09-02 15:38:53")
    # orphaned abstract from a hand-cleared wrong match
    _seed(conn, 4, abstract="Daily Show", metadata_suspect=1)
    # abstract-without-title that is NOT a cleared suspect: report only
    _seed(conn, 5, abstract="Maybe real", metadata_suspect=0)
    # legacy miss: stamped, no source, no title
    _seed(conn, 6, metadata_enriched_at="2026-09-10T19:29:00+00:00")


def test_normalize_dry_run_counts_and_writes_nothing(db):
    _seed_messy(db)
    report = normalize_metadata(db, dry_run=True)
    assert report["authors_normalized"] == 1
    assert report["authors_ids"] == [1]
    assert report["timestamps_normalized"] == {
        "metadata_enriched_at": 1, "metadata_verified_at": 1, "updated_at": 1,
    }
    assert report["orphan_abstracts_cleared"] == 1
    assert report["orphan_abstract_ids"] == [4]
    assert report["other_orphan_abstract_ids"] == [5]
    assert report["legacy_misses_tagged"] == 1
    assert _row(db, 1)["authors"] == "Ann A; Ben B"
    assert _row(db, 4)["abstract"] == "Daily Show"


def test_normalize_applies_and_is_idempotent(db):
    _seed_messy(db)
    normalize_metadata(db)
    assert json.loads(_row(db, 1)["authors"]) == ["Ann A", "Ben B"]
    assert _row(db, 2)["authors"] == '["Already", "Json"]'
    assert _row(db, 3)["metadata_enriched_at"] == "2026-08-19T19:13:09+00:00"
    assert _row(db, 3)["updated_at"] == "2026-08-19T19:13:09+00:00"
    assert _row(db, 4)["abstract"] is None
    assert _row(db, 5)["abstract"] == "Maybe real"
    assert _row(db, 6)["metadata_source"] == "none:legacy"

    again = normalize_metadata(db)
    assert again["authors_normalized"] == 0
    assert sum(again["timestamps_normalized"].values()) == 0
    assert again["orphan_abstracts_cleared"] == 0
    assert again["legacy_misses_tagged"] == 0


def test_normalize_all_orphans(db):
    _seed_messy(db)
    report = normalize_metadata(db, all_orphans=True)
    assert report["orphan_abstract_ids"] == [4, 5]
    assert _row(db, 5)["abstract"] is None


def test_cli_meta_normalize_json(cli_env):
    _seed_messy(cli_env)
    r = _run("meta", "normalize", "--dry-run", "--json")
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["dry_run"] is True
    assert data["authors_normalized"] == 1
