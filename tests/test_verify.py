"""Tests for gantry verify — auditing title-search-sourced metadata matches.

enrich_documents()'s title-search fallback (metadata.py) accepts the top API
result unconditionally, with no similarity check against the PDF's actual
title. verify_documents() closes that gap: it recomputes a similarity score
between the stored (API-sourced) title and the title extracted from the raw
PDF text, and flags low-similarity matches as suspect.
"""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.metadata import title_similarity, verify_documents

# --- title_similarity ---

def test_title_similarity_identical():
    assert title_similarity("Deep Learning for NLP", "Deep Learning for NLP") == 1.0


def test_title_similarity_case_and_whitespace_insensitive():
    """Trivial formatting differences shouldn't tank the score."""
    score = title_similarity("Deep Learning for NLP", "  deep   learning FOR nlp  ")
    assert score > 0.95


def test_title_similarity_completely_different():
    score = title_similarity(
        "Deep Learning for Natural Language Processing",
        "Introduction to Statistical Mechanics",
    )
    assert score < 0.4


def test_title_similarity_none_inputs():
    assert title_similarity(None, "Something") == 0.0
    assert title_similarity("Something", None) == 0.0
    assert title_similarity(None, None) == 0.0


def test_title_similarity_empty_strings():
    assert title_similarity("", "Something") == 0.0


# --- verify_documents ---

def _seed_paper(conn, paper_id, *, title, metadata_source, raw_text=""):
    conn.execute(
        "INSERT INTO papers (id, path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at, title, metadata_source) "
        "VALUES (?, ?, ?, ?, 1, '2026-01-01', '2026-01-01', '2026-01-01', ?, ?)",
        (paper_id, f"/p/p{paper_id}.pdf", f"p{paper_id}.pdf", f"h{paper_id}",
         title, metadata_source),
    )
    if raw_text:
        conn.execute(
            "INSERT INTO paper_text (paper_id, raw_text, markdown, text_length, "
            "markdown_length) VALUES (?, ?, '', ?, 0)",
            (paper_id, raw_text, len(raw_text)),
        )
    conn.commit()


@pytest.fixture
def verify_db(tmp_path):
    conn = get_connection(str(tmp_path / "index.db"))
    yield conn
    conn.close()


def test_verify_flags_mismatched_title_match(verify_db):
    """A title-sourced match whose stored title doesn't resemble the PDF's own title is suspect."""
    _seed_paper(
        verify_db, 1,
        title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Introduction to Statistical Mechanics\nSome intro body text follows here.",
    )

    results = verify_documents(verify_db)
    assert len(results) == 1
    r = results[0]
    assert r.paper_id == 1
    assert r.suspect is True
    assert r.similarity < 0.4


def test_verify_passes_good_title_match(verify_db):
    """A title-sourced match whose title matches the PDF's own title is not suspect."""
    _seed_paper(
        verify_db, 1,
        title="Deep Learning for Natural Language Processing",
        metadata_source="semantic_scholar_title",
        raw_text="Deep Learning for Natural Language Processing\nAbstract: we present...",
    )

    results = verify_documents(verify_db)
    assert len(results) == 1
    assert results[0].suspect is False
    assert results[0].similarity > 0.55


def test_verify_skips_doi_sourced_metadata(verify_db):
    """DOI-matched metadata is exact by construction — verify only checks title-sourced matches."""
    _seed_paper(
        verify_db, 1,
        title="Anything At All",
        metadata_source="openalex",
        raw_text="Completely different text on the page.",
    )

    results = verify_documents(verify_db)
    assert results == []


def test_verify_flags_unverifiable_when_no_raw_text(verify_db):
    """No extracted text means the match can't be confirmed — treat as suspect."""
    _seed_paper(
        verify_db, 1,
        title="Some Title",
        metadata_source="openalex_title",
        raw_text="",
    )

    results = verify_documents(verify_db)
    assert len(results) == 1
    assert results[0].suspect is True
    assert results[0].similarity == 0.0


def test_verify_persists_verdict_to_db(verify_db):
    """verify_documents writes metadata_suspect/metadata_verify_score/metadata_verified_at."""
    _seed_paper(
        verify_db, 1,
        title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Introduction to Statistical Mechanics\nbody",
    )

    verify_documents(verify_db)

    row = verify_db.execute(
        "SELECT metadata_suspect, metadata_verify_score, metadata_verified_at "
        "FROM papers WHERE id = 1"
    ).fetchone()
    assert row["metadata_suspect"] == 1
    assert row["metadata_verify_score"] < 0.4
    assert row["metadata_verified_at"] is not None


def test_verify_scoped_to_ids(verify_db):
    """paper_ids restricts which title-sourced papers get checked."""
    _seed_paper(
        verify_db, 1, title="Match One", metadata_source="openalex_title",
        raw_text="Totally Different First Line\nbody",
    )
    _seed_paper(
        verify_db, 2, title="Match Two", metadata_source="openalex_title",
        raw_text="Totally Different First Line\nbody",
    )

    results = verify_documents(verify_db, paper_ids=[1])
    assert [r.paper_id for r in results] == [1]


def test_verify_worst_matches_first(verify_db):
    """Results are ordered by ascending similarity so the worst offenders surface first."""
    _seed_paper(
        verify_db, 1, title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Deep Learning for Natural Language Processing\nabstract",
    )
    _seed_paper(
        verify_db, 2, title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Nothing Resembling That Title At All\nbody",
    )

    results = verify_documents(verify_db)
    assert [r.paper_id for r in results] == [2, 1]


# --- CLI ---

def test_cli_verify_reports_suspects_and_exits_partial(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_paper(
        conn, 1, title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Introduction to Statistical Mechanics\nbody",
    )
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["verify", "--json"])
    assert result.exit_code == 3  # EXIT_PARTIAL — corpus has a real data-quality issue
    data = json.loads(result.output)
    assert data["total"] == 1
    assert data["suspect_count"] == 1
    assert data["results"][0]["paper_id"] == 1
    assert data["results"][0]["suspect"] is True


def test_cli_verify_clean_corpus_exits_zero(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_paper(
        conn, 1, title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Deep Learning for Natural Language Processing\nabstract",
    )
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["verify", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["suspect_count"] == 0


def test_cli_verify_no_title_sourced_papers_exits_no_results(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_paper(conn, 1, title="Anything", metadata_source="openalex", raw_text="x")
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["verify", "--json"])
    assert result.exit_code == 2


def test_cli_verify_ids_only_pipes_suspect_ids(tmp_path, monkeypatch):
    """--ids-only emits bare suspect paper IDs, one per line — composable with queue/enrich/info."""
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_paper(
        conn, 1, title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Introduction to Statistical Mechanics\nbody",
    )
    _seed_paper(
        conn, 2, title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Deep Learning for Natural Language Processing\nabstract",
    )
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["verify", "--ids-only"])
    assert result.output.strip() == "1"


def test_cli_verify_scoped_by_ids_flag(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_paper(
        conn, 1, title="Match One", metadata_source="openalex_title",
        raw_text="Totally Different\nbody",
    )
    _seed_paper(
        conn, 2, title="Match Two", metadata_source="openalex_title",
        raw_text="Totally Different\nbody",
    )
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["verify", "--ids", "1", "--json"])
    data = json.loads(result.output)
    assert data["total"] == 1
    assert data["results"][0]["paper_id"] == 1


# --- CLI: --ids reports IDs that don't exist vs. exist but aren't title-sourced ---
#
# verify --ids is doubly ambiguous compared to info/ocr/retry's --ids: a requested
# id that doesn't show up in results might not exist at all (not_found), or it might
# exist but be DOI/filename-sourced and so outside verify's scope by design
# (skipped_ineligible) -- not a data problem. Collapsing the two would misreport a
# working filter as a corpus defect.


def test_cli_verify_ids_reports_not_found(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_paper(
        conn, 1, title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Deep Learning for Natural Language Processing\nAbstract: we present...",
    )
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["verify", "--ids", "1,999", "--json"])
    data = json.loads(result.output)
    assert data["not_found"] == [999]
    assert data["skipped_ineligible"] == []
    assert data["total"] == 1
    assert result.exit_code == 3  # EXIT_PARTIAL: real results, but one requested id missing


def test_cli_verify_ids_reports_skipped_ineligible(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_paper(
        conn, 1, title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Deep Learning for Natural Language Processing\nAbstract: we present...",
    )
    _seed_paper(
        conn, 2, title="DOI Sourced", metadata_source="openalex_doi",
        raw_text="DOI Sourced\nbody",
    )
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["verify", "--ids", "1,2", "--json"])
    data = json.loads(result.output)
    assert data["not_found"] == []
    assert data["skipped_ineligible"] == [2]
    assert data["total"] == 1
    assert result.exit_code == 3


def test_cli_verify_ids_all_unresolved_exits_no_results(tmp_path, monkeypatch):
    """A requested id that's DOI-sourced and one that doesn't exist -- zero checked."""
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_paper(
        conn, 2, title="DOI Sourced", metadata_source="openalex_doi",
        raw_text="DOI Sourced\nbody",
    )
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["verify", "--ids", "2,999", "--json"])
    assert result.exit_code == 2  # EXIT_NO_RESULTS
    data = json.loads(result.output)
    assert data["not_found"] == [999]
    assert data["skipped_ineligible"] == [2]


def test_cli_verify_ids_all_resolved_no_diagnostic_fields_populated(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_paper(
        conn, 1, title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Deep Learning for Natural Language Processing\nAbstract: we present...",
    )
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["verify", "--ids", "1", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["not_found"] == []
    assert data["skipped_ineligible"] == []


def test_cli_verify_ids_only_reports_unresolved_to_stderr(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_paper(
        conn, 1, title="Deep Learning for Natural Language Processing",
        metadata_source="openalex_title",
        raw_text="Introduction to Statistical Mechanics\nbody",
    )
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["verify", "--ids", "1,999", "--ids-only"])
    assert result.stdout.strip() == "1"
    assert "999" in result.stderr
    assert result.exit_code == 3
