"""Tests for metadata completeness gaps (`gantry gaps`).

The enrichment pipeline (metadata.py::enrich_documents) marks a paper as
"done" the moment `metadata_enriched_at` is set -- whether or not the
provider actually returned a title, an abstract, or a DOI. `queue --needs
metadata` only looks at `doi IS NULL OR metadata_enriched_at IS NULL`, so a
paper that was matched by title (no DOI exists, e.g. a lot of preprints and
book chapters) or whose provider simply has no abstract on file stays
invisible forever: enriched-but-incomplete looks identical to
enriched-and-complete to every existing command. These tests pin down a new
`field_completeness()` / `gap_ids()` pair (and the `gantry gaps` CLI command
built on them) that makes that silent gap visible and actionable.
"""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.metadata import field_completeness, gap_ids
from pdf_gantry.queue import query_queue


def _seed(conn, paper_id, *, doi=None, title=None, authors=None, year=None,
          abstract=None, metadata_enriched_at=None, filename=None):
    filename = filename or f"p{paper_id}.pdf"
    conn.execute(
        "INSERT INTO papers (id, path, filename, file_hash, file_size, "
        "file_modified, indexed_at, updated_at, doi, title, authors, year, "
        "abstract, metadata_enriched_at) "
        "VALUES (?, ?, ?, ?, 1, '2026-01-01', '2026-01-01', '2026-01-01', "
        "?, ?, ?, ?, ?, ?)",
        (paper_id, f"/p/{filename}", filename, f"h{paper_id}",
         doi, title, authors, year, abstract, metadata_enriched_at),
    )
    conn.commit()


@pytest.fixture
def gaps_db(tmp_path):
    conn = get_connection(str(tmp_path / "index.db"))
    yield conn
    conn.close()


def _seed_mixed_corpus(conn):
    """3 papers spanning the three completeness states this feature exists to tell apart."""
    # 1: never enriched at all -- the ordinary, expected gap.
    _seed(conn, 1, filename="never.pdf")
    # 2: enrichment ran and *matched by DOI* (doi and title/authors/year all
    #    populated) but the provider had no abstract_inverted_index for this
    #    work -- common on OpenAlex for older/closed-access papers. Because
    #    doi is non-NULL and metadata_enriched_at is non-NULL, this paper is
    #    indistinguishable from a fully-complete one to every existing
    #    command (`queue --needs metadata` will never re-surface it).
    _seed(conn, 2, filename="silent_gap.pdf", doi="10.1/silent", title="A Paper",
          authors="[]", year=2019, abstract=None,
          metadata_enriched_at="2026-01-02")
    # 3: fully enriched and complete.
    _seed(conn, 3, filename="complete.pdf", doi="10.1/x", title="Full Paper",
          authors='["A. Author"]', year=2020, abstract="An abstract.",
          metadata_enriched_at="2026-01-02")
    return conn


# --- metadata.field_completeness ---


def test_field_completeness_reports_total(gaps_db):
    _seed_mixed_corpus(gaps_db)
    result = field_completeness(gaps_db)
    assert result["total"] == 3


def test_field_completeness_splits_never_attempted_vs_attempted_incomplete(gaps_db):
    _seed_mixed_corpus(gaps_db)
    result = field_completeness(gaps_db)
    abstract = result["fields"]["abstract"]
    assert abstract["missing"] == 2
    assert abstract["never_attempted"] == 1       # paper 1
    assert abstract["attempted_incomplete"] == 1   # paper 2 -- the invisible one
    assert abstract["complete"] == 1               # paper 3


def test_field_completeness_doi_is_fine_once_matched(gaps_db):
    """Paper 2's DOI was found and written -- the persistent gap is
    elsewhere (abstract), proving completeness is genuinely per-field,
    not a single enriched/not-enriched bit."""
    _seed_mixed_corpus(gaps_db)
    result = field_completeness(gaps_db)
    doi = result["fields"]["doi"]
    assert doi["missing"] == 1  # paper 1 only
    assert doi["attempted_incomplete"] == 0


def test_field_completeness_treats_empty_json_author_list_as_missing(gaps_db):
    """authors='[]' is truthy text but zero authors -- must count as a gap,
    not slip through a naive `authors IS NULL` check."""
    _seed_mixed_corpus(gaps_db)
    result = field_completeness(gaps_db)
    assert result["fields"]["authors"]["missing"] == 2  # papers 1 and 2


def test_field_completeness_scoped_to_requested_fields(gaps_db):
    _seed_mixed_corpus(gaps_db)
    result = field_completeness(gaps_db, fields=["abstract"])
    assert set(result["fields"].keys()) == {"abstract"}


def test_field_completeness_unknown_field_raises(gaps_db):
    with pytest.raises(ValueError):
        field_completeness(gaps_db, fields=["bogus"])


# --- metadata.gap_ids ---


def test_gap_ids_returns_all_missing_sorted(gaps_db):
    _seed_mixed_corpus(gaps_db)
    assert gap_ids(gaps_db, "abstract") == [1, 2]


def test_gap_ids_attempted_only_isolates_the_silent_failure(gaps_db):
    """This is the sharp end: scoped to papers enrichment already ran on and
    still couldn't fill in -- re-running `enrich` on the same provider won't
    help these, they need a different provider or a manual look."""
    _seed_mixed_corpus(gaps_db)
    assert gap_ids(gaps_db, "abstract", attempted_only=True) == [2]


def test_gap_ids_unknown_field_raises(gaps_db):
    with pytest.raises(ValueError):
        gap_ids(gaps_db, "bogus")


def test_attempted_incomplete_paper_is_invisible_to_existing_queue_filter(gaps_db):
    """Confirms the actual blind spot: `queue --needs metadata` never re-surfaces
    paper 2 (doi found, metadata_enriched_at set), yet it is missing an abstract
    forever. gap_ids() is the only place that gap is visible."""
    _seed_mixed_corpus(gaps_db)
    needs_metadata_ids = {r["id"] for r in query_queue(gaps_db, needs=["metadata"])}
    assert 2 not in needs_metadata_ids
    assert 2 in gap_ids(gaps_db, "abstract", attempted_only=True)


# --- CLI: gantry gaps ---


def test_cli_gaps_json_summary(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_mixed_corpus(conn)
    conn.close()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["gaps", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total"] == 3
    assert data["fields"]["abstract"]["attempted_incomplete"] == 1
    assert data["fields"]["doi"]["missing"] == 1


def test_cli_gaps_field_ids_only_is_pipeable(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_mixed_corpus(conn)
    conn.close()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["gaps", "--field", "abstract", "--ids-only"])
    assert result.exit_code == 0
    assert result.output.strip().split("\n") == ["1", "2"]


def test_cli_gaps_field_attempted_only_ids_only(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_mixed_corpus(conn)
    conn.close()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(
        cli, ["gaps", "--field", "abstract", "--attempted-only", "--ids-only"]
    )
    assert result.exit_code == 0
    assert result.output.strip() == "2"


def test_cli_gaps_field_no_matches_is_exit_code_2(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed(conn, 1, doi="10.1/x", title="T", authors="[]", year=2020,
          abstract="abstract text", metadata_enriched_at="2026-01-02")
    conn.close()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["gaps", "--field", "abstract", "--ids-only"])
    assert result.exit_code == 2
    assert result.output.strip() == ""


def test_cli_gaps_no_db_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["gaps", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert "error" in data
