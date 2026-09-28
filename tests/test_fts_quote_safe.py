"""Quote-safe FTS (B1): user text is literal terms by default.

Pasting a paper title is the most common "do I have X" query, and titles
contain colons, parentheses and apostrophes that FTS5 treats as syntax.
These inputs all used to exit 1 with ``no such column`` / ``syntax error``.
"""

import json
import sqlite3

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.search import _sanitize_fts_query, fts_search, hybrid_search, search_count

from .search_helpers import add_paper, new_db, point_cli_at

CRASHERS = [
    "Deepfakes and Social Media: Implications for the 2020 Election",
    "Social Media: Implications",
    "co-occurrence (network)",
    "C++ OR",
    'what\'s "new',
    'what\'s "new"',
    "AND",
    "OR NOT",
    "NEAR(a b)",
    "title:climate",
    "^climate",
    "climate +change",
    "()",
    "*",
]


@pytest.fixture
def fts_db(tmp_path):
    """A tiny DB with FTS rows (no PDFs needed)."""
    conn, db_path = new_db(tmp_path)
    add_paper(conn, 1, "a.pdf",
              title="Deepfakes and Social Media: Implications for the 2020 Election",
              text="social media deepfakes election implications co-occurrence network")
    add_paper(conn, 2, "b.pdf", title="Climate change adaptation",
              text="climate change adaptation policy what's new in C++")
    return conn, db_path


@pytest.mark.parametrize("query", CRASHERS)
def test_fts_search_literal_never_raises(fts_db, query):
    conn, _ = fts_db
    results = fts_search(conn, query)
    assert isinstance(results, list)
    assert isinstance(search_count(conn, query), int)


def test_title_with_colon_finds_paper(fts_db):
    conn, _ = fts_db
    results = fts_search(conn, "Deepfakes and Social Media: Implications for the 2020 Election")
    assert [r.id for r in results] == [1]


def test_parenthesised_and_hyphenated_terms_match(fts_db):
    conn, _ = fts_db
    assert [r.id for r in fts_search(conn, "co-occurrence (network)")] == [1]


def test_infix_or_keeps_or_semantics(fts_db):
    """An OR between two terms is still an operator (backward compatible)."""
    conn, _ = fts_db
    ids = {r.id for r in fts_search(conn, "deepfakes OR adaptation")}
    assert ids == {1, 2}


def test_dangling_operator_is_literal(fts_db):
    conn, _ = fts_db
    # "C++ OR" — trailing OR has no right operand, so it's quoted as a term.
    assert _sanitize_fts_query("C++ OR").count('"') >= 4


def test_prefix_star_preserved(fts_db):
    conn, _ = fts_db
    assert [r.id for r in fts_search(conn, "adapt*")] == [2]


def test_phrase_preserved(fts_db):
    conn, _ = fts_db
    assert [r.id for r in fts_search(conn, '"climate change"')] == [2]
    assert fts_search(conn, '"change climate"') == []


def test_fts_syntax_mode_passes_raw(fts_db):
    conn, _ = fts_db
    # Column filter is FTS5 syntax: literal mode would search the word "title".
    assert [r.id for r in fts_search(conn, "title:climate", syntax=True)] == [2]
    with pytest.raises(sqlite3.OperationalError):
        fts_search(conn, "Social Media: Implications", syntax=True)


def test_hybrid_falls_back_to_vector_on_fts_error(fts_db, monkeypatch):
    conn, _ = fts_db
    from pdf_gantry import search as search_mod
    from pdf_gantry.models import SearchResult

    monkeypatch.setattr(
        search_mod, "semantic_search",
        lambda *a, **k: [SearchResult(id=2, filename="b.pdf", path="/p/b.pdf", score=0.5)],
    )
    status: dict = {}
    results = hybrid_search(
        conn, "Social Media: Implications", b"v", syntax=True, status=status,
    )
    assert [r.id for r in results] == [2]
    assert status["mode"] == "vector_fallback"
    assert "fts_error" in status


_cli_env = point_cli_at


@pytest.mark.parametrize("query", CRASHERS)
def test_cli_search_fts_does_not_crash(fts_db, tmp_path, monkeypatch, query):
    conn, db_path = fts_db
    conn.close()
    _cli_env(tmp_path, monkeypatch, db_path)
    result = CliRunner().invoke(cli, ["search", query, "--fts", "--json"])
    assert result.exit_code in (0, 2), result.output
    payload = json.loads(result.stdout)
    assert "results" in payload


def test_cli_search_hybrid_title_with_colon(fts_db, tmp_path, monkeypatch):
    conn, db_path = fts_db
    conn.close()
    _cli_env(tmp_path, monkeypatch, db_path)
    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", lambda m, q: b"\x00" * 3072)
    monkeypatch.setattr("pdf_gantry.search.semantic_search", lambda *a, **k: [])
    result = CliRunner().invoke(cli, ["search", "Social Media: Implications", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "hybrid"
    assert [r["id"] for r in payload["results"]] == [1]


def test_cli_hybrid_fts_syntax_error_reports_vector_fallback(fts_db, tmp_path, monkeypatch):
    conn, db_path = fts_db
    conn.close()
    _cli_env(tmp_path, monkeypatch, db_path)
    from pdf_gantry.models import SearchResult
    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", lambda m, q: b"\x00" * 3072)
    monkeypatch.setattr(
        "pdf_gantry.search.semantic_search",
        lambda *a, **k: [SearchResult(id=2, filename="b.pdf", path="/p/b.pdf", score=0.5)],
    )
    result = CliRunner().invoke(
        cli, ["search", "Social Media: Implications", "--fts-syntax", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "vector_fallback"
    assert [r["id"] for r in payload["results"]] == [2]


def test_cli_fts_syntax_error_in_fts_only_mode_is_clean_error(fts_db, tmp_path, monkeypatch):
    conn, db_path = fts_db
    conn.close()
    _cli_env(tmp_path, monkeypatch, db_path)
    result = CliRunner().invoke(
        cli, ["search", "Social Media: Implications", "--fts", "--fts-syntax", "--json"]
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "FTS5 syntax" in payload["error"]
