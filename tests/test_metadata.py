"""Tests for metadata enrichment."""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry import openalex
from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.metadata import enrich_documents, extract_doi


def test_extract_doi_standard():
    """Standard DOI pattern is extracted."""
    text = "Available at https://doi.org/10.1234/test.paper.2024"
    doi = extract_doi(text)
    assert doi == "10.1234/test.paper.2024"


def test_extract_doi_in_text():
    """DOI embedded in text is extracted."""
    text = "The paper (doi: 10.1038/nature12373) discusses..."
    doi = extract_doi(text)
    assert doi == "10.1038/nature12373"


def test_extract_doi_none():
    """No DOI returns None."""
    text = "This is a document without any DOI"
    doi = extract_doi(text)
    assert doi is None


def test_extract_doi_trailing_punctuation():
    """Trailing punctuation is stripped from DOI."""
    text = "See 10.1234/test.paper."
    doi = extract_doi(text)
    assert doi == "10.1234/test.paper"


def test_extract_doi_with_parenthesis():
    """DOI followed by closing paren is cleaned."""
    text = "(10.1234/test.paper)"
    doi = extract_doi(text)
    assert doi == "10.1234/test.paper"


# --- provider dispatch (issue #20) ---


def _seed(conn, paper_id, *, doi=None, raw_text="", filename=None):
    filename = filename or f"p{paper_id}.pdf"
    conn.execute(
        "INSERT INTO papers (id, path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at, doi) "
        "VALUES (?, ?, ?, ?, 1, '2026-01-01', '2026-01-01', '2026-01-01', ?)",
        (paper_id, f"/p/{filename}", filename, f"h{paper_id}", doi),
    )
    if raw_text:
        conn.execute(
            "INSERT INTO paper_text (paper_id, raw_text, markdown, text_length, "
            "markdown_length) VALUES (?, ?, '', ?, 0)",
            (paper_id, raw_text, len(raw_text)),
        )
    conn.commit()


@pytest.fixture
def enrich_db(tmp_path):
    conn = get_connection(str(tmp_path / "index.db"))
    yield conn
    conn.close()


def test_enrich_openalex_is_default(enrich_db, monkeypatch):
    """With no provider arg, enrich uses OpenAlex by DOI and writes metadata."""
    _seed(enrich_db, 1, doi="10.1234/x")

    def fake_by_doi(doi, mailto=None, rate_limit=0.0):
        return {
            "title": "OA Title", "authors": ["Ada Lovelace"], "year": 2024,
            "abstract": "An abstract", "doi": doi, "source_id": "W1",
        }

    monkeypatch.setattr(openalex, "fetch_by_doi", fake_by_doi)

    stats = enrich_documents(enrich_db, paper_ids=[1])
    assert stats.total == 1

    row = enrich_db.execute(
        "SELECT title, authors, year, metadata_source FROM papers WHERE id = 1"
    ).fetchone()
    assert row["title"] == "OA Title"
    assert json.loads(row["authors"]) == ["Ada Lovelace"]
    assert row["year"] == 2024
    assert row["metadata_source"] == "openalex"


def test_enrich_openalex_title_fallback(enrich_db, monkeypatch):
    """No DOI -> filename miss -> title search succeeds."""
    _seed(enrich_db, 1, raw_text="Some Distinctive Paper Title\nbody",
          filename="weird.pdf")

    monkeypatch.setattr(openalex, "fetch_by_doi", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_filename", lambda *a, **k: None)
    monkeypatch.setattr(
        openalex, "fetch_by_title",
        lambda title, mailto=None, rate_limit=0.0: {
            "title": "Matched", "authors": [], "year": 2020,
            "abstract": None, "doi": None, "source_id": "W2",
        },
    )

    stats = enrich_documents(enrich_db, paper_ids=[1])
    assert stats.matched_by_title == 1
    row = enrich_db.execute("SELECT metadata_source FROM papers WHERE id = 1").fetchone()
    assert row["metadata_source"] == "openalex_title"


def test_enrich_openalex_mailto_passed(enrich_db, monkeypatch):
    """The configured mailto reaches the OpenAlex fetcher."""
    _seed(enrich_db, 1, doi="10.1234/x")
    seen = {}

    def fake_by_doi(doi, mailto=None, rate_limit=0.0):
        seen["mailto"] = mailto
        return {"title": "T", "authors": [], "year": None,
                "abstract": None, "doi": doi, "source_id": "W"}

    monkeypatch.setattr(openalex, "fetch_by_doi", fake_by_doi)
    enrich_documents(enrich_db, paper_ids=[1], mailto="ryan@example.com")
    assert seen["mailto"] == "ryan@example.com"


def test_enrich_semantic_scholar_still_works(enrich_db, monkeypatch):
    """provider='semantic-scholar' routes to the SS fetchers."""
    _seed(enrich_db, 1, doi="10.1234/x")

    import pdf_gantry.metadata as md
    monkeypatch.setattr(
        md, "_fetch_by_doi",
        lambda doi, rate_limit=0.1: {
            "title": "SS Title",
            "authors": [{"name": "Grace Hopper"}],
            "year": 1999,
            "abstract": "ss abstract",
            "paperId": "ss123",
            "externalIds": {"DOI": doi},
        },
    )

    stats = enrich_documents(enrich_db, paper_ids=[1], provider="semantic-scholar")
    assert stats.total == 1
    row = enrich_db.execute(
        "SELECT title, authors, semantic_scholar_id, metadata_source "
        "FROM papers WHERE id = 1"
    ).fetchone()
    assert row["title"] == "SS Title"
    assert json.loads(row["authors"]) == ["Grace Hopper"]
    assert row["semantic_scholar_id"] == "ss123"
    assert row["metadata_source"] == "semantic_scholar"


def test_cli_enrich_defaults_to_openalex(tmp_path, monkeypatch):
    """`gantry enrich` with no flags uses OpenAlex and writes metadata."""
    conn = get_connection(str(tmp_path / "index.db"))
    _seed(conn, 1, doi="10.1234/x")
    conn.close()

    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))
    monkeypatch.setattr(
        openalex, "fetch_by_doi",
        lambda doi, mailto=None, rate_limit=0.0: {
            "title": "OA CLI", "authors": ["A B"], "year": 2021,
            "abstract": None, "doi": doi, "source_id": "W",
        },
    )

    result = CliRunner().invoke(cli, ["enrich", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total"] == 1

    conn = get_connection(str(tmp_path / "index.db"))
    row = conn.execute(
        "SELECT title, metadata_source FROM papers WHERE id = 1"
    ).fetchone()
    conn.close()
    assert row["title"] == "OA CLI"
    assert row["metadata_source"] == "openalex"
