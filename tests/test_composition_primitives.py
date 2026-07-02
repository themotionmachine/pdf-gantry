"""Tests for composable agent primitives: --ids-only and --restrict-to-ids.

These shape how an agent chains retrieval steps without parsing full JSON
between them (`--ids-only`) and how it scopes a search to a candidate doc set
(`--restrict-to-ids`) — "which of THESE papers discuss X?" in one call.
"""

import json
import struct

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection


def _vec(axis: int, dim: int = 768) -> bytes:
    """A unit vector along a single axis, serialized for sqlite-vec."""
    v = [0.0] * dim
    v[axis] = 1.0
    return struct.pack(f"{dim}f", *v)


def _add_paper(conn, filename: str, text: str, axis: int) -> int:
    """Insert a paper with FTS content, raw text, and a doc-level embedding."""
    cur = conn.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at, has_text, has_markdown, has_embeddings) "
        "VALUES (?, ?, 'hash', 100, '2024-01-01', '2024-01-01', '2024-01-01', 1, 1, 1)",
        (f"/papers/{filename}", filename),
    )
    pid = cur.lastrowid
    conn.execute(
        "INSERT INTO paper_text (paper_id, raw_text, text_length) VALUES (?, ?, ?)",
        (pid, text, len(text)),
    )
    conn.execute(
        "INSERT INTO papers_fts (rowid, filename, title, authors, abstract, text_content) "
        "VALUES (?, ?, '', '', '', ?)",
        (pid, filename, text),
    )
    conn.execute(
        "INSERT INTO paper_embeddings (paper_id, embedding) VALUES (?, ?)",
        (pid, _vec(axis)),
    )
    return pid


@pytest.fixture
def corpus(tmp_path):
    """Three papers all matching 'climate' on FTS, on distinct embedding axes."""
    conn = get_connection(str(tmp_path / "test.db"))
    a = _add_paper(conn, "alpha.pdf", "climate adaptation in coastal regions", 0)
    b = _add_paper(conn, "beta.pdf", "climate policy and neural machine models", 1)
    c = _add_paper(conn, "gamma.pdf", "climate risk and machine learning methods", 2)
    conn.commit()
    return conn, a, b, c


# --- search.py: restrict_ids on the search functions ---


def test_fts_search_restrict_ids_scopes(corpus):
    from pdf_gantry.search import fts_search

    conn, a, b, c = corpus
    results = fts_search(conn, "climate", restrict_ids=[a])
    assert {r.id for r in results} == {a}


def test_fts_search_restrict_ids_excludes_nonmembers(corpus):
    from pdf_gantry.search import fts_search

    conn, a, b, c = corpus
    results = fts_search(conn, "climate", restrict_ids=[b, c])
    assert {r.id for r in results} == {b, c}
    assert a not in {r.id for r in results}


def test_fts_search_restrict_empty_returns_empty(corpus):
    from pdf_gantry.search import fts_search

    conn, a, b, c = corpus
    assert fts_search(conn, "climate", restrict_ids=[]) == []


def test_semantic_search_restrict_ids_scopes(corpus):
    from pdf_gantry.search import semantic_search

    conn, a, b, c = corpus
    # Query nearest to axis 0 (doc a), but restrict away from it.
    results = semantic_search(conn, _vec(0), restrict_ids=[b, c])
    assert {r.id for r in results} == {b, c}


def test_semantic_search_restrict_empty_returns_empty(corpus):
    from pdf_gantry.search import semantic_search

    conn, a, b, c = corpus
    assert semantic_search(conn, _vec(0), restrict_ids=[]) == []


def test_hybrid_search_restrict_ids_scopes(corpus):
    from pdf_gantry.search import hybrid_search

    conn, a, b, c = corpus
    results = hybrid_search(conn, "climate", _vec(0), restrict_ids=[a])
    assert {r.id for r in results} == {a}


# --- CLI: --restrict-to-ids and --ids-only ---


@pytest.fixture
def cli_corpus(tmp_path, monkeypatch):
    """A configured DB (index_dir/index.db) for CLI tests; query along axis 0."""
    conn = get_connection(str(tmp_path / "index.db"))
    a = _add_paper(conn, "alpha.pdf", "climate adaptation in coastal regions", 0)
    b = _add_paper(conn, "beta.pdf", "climate policy and neural machine models", 1)
    c = _add_paper(conn, "gamma.pdf", "climate risk and machine learning methods", 2)
    conn.commit()
    conn.close()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)
    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", lambda model, q: _vec(0))
    return tmp_path, a, b, c


def test_search_ids_only_emits_bare_ids(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "climate", "--fts", "--ids-only"])
    assert result.exit_code == 0
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    # Every line is a bare integer id, nothing else (no scores, snippets, headers).
    ids = [int(ln) for ln in lines]
    assert set(ids) == {a, b, c}


def test_search_ids_only_preserves_rank_order(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    runner = CliRunner()
    # Hybrid query along axis 0 should rank doc a (exact match) first.
    result = runner.invoke(cli, ["search", "climate", "--ids-only"])
    assert result.exit_code == 0
    ids = [int(ln) for ln in result.output.splitlines() if ln.strip()]
    assert ids[0] == a


def test_search_restrict_to_ids(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    runner = CliRunner()
    result = runner.invoke(
        cli, ["search", "climate", "--fts", "--restrict-to-ids", f"{b},{c}", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert {r["id"] for r in payload["results"]} == {b, c}


def test_search_restrict_and_ids_only_compose(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["search", "climate", "--fts", "--restrict-to-ids", f"{a},{b}", "--ids-only"],
    )
    assert result.exit_code == 0
    ids = {int(ln) for ln in result.output.splitlines() if ln.strip()}
    assert ids == {a, b}


def test_search_restrict_empty_intersection_exit_2(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    runner = CliRunner()
    # Restrict to an id that exists but won't match the FTS query.
    result = runner.invoke(
        cli, ["search", "nonexistentxyz", "--fts", "--restrict-to-ids", str(a), "--json"]
    )
    assert result.exit_code == 2


def test_search_restrict_works_in_hybrid(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    runner = CliRunner()
    result = runner.invoke(
        cli, ["search", "climate", "--restrict-to-ids", str(b), "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert {r["id"] for r in payload["results"]} == {b}


def test_semantic_ids_only_and_restrict(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    runner = CliRunner()
    result = runner.invoke(
        cli, ["semantic", "climate", "--doc-only", "--restrict-to-ids", f"{b},{c}", "--ids-only"]
    )
    assert result.exit_code == 0
    ids = {int(ln) for ln in result.output.splitlines() if ln.strip()}
    assert ids == {b, c}


# --- CLI: --restrict-to-ids reports unresolved IDs instead of silently dropping them ---
#
# A bad ID in --restrict-to-ids used to just shrink the search's candidate set with
# no signal -- indistinguishable from "your scope was fine but nothing matched."
# resolve_ids() (utils.py) now splits the requested set into found/not_found up
# front, mirroring the not_found convention info/ocr/retry already use for --ids.


def test_search_restrict_to_ids_reports_not_found(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    bogus = max(a, b, c) + 1000
    runner = CliRunner()
    result = runner.invoke(
        cli, ["search", "climate", "--fts", "--restrict-to-ids", f"{a},{bogus}", "--json"]
    )
    payload = json.loads(result.output)
    assert payload["not_found"] == [bogus]
    assert {r["id"] for r in payload["results"]} == {a}
    assert result.exit_code == 3  # EXIT_PARTIAL: real results, but scope had a bad id


def test_search_restrict_to_ids_all_valid_not_found_empty(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    runner = CliRunner()
    result = runner.invoke(
        cli, ["search", "climate", "--fts", "--restrict-to-ids", f"{a},{b}", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["not_found"] == []


def test_search_restrict_to_ids_not_found_plain_text(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    bogus = max(a, b, c) + 1000
    runner = CliRunner()
    result = runner.invoke(
        cli, ["search", "climate", "--fts", "--restrict-to-ids", f"{a},{bogus}"]
    )
    assert "Not in index" in result.output
    assert str(bogus) in result.output
    assert result.exit_code == 3


def test_search_restrict_to_ids_all_missing_exits_no_results(cli_corpus):
    """When every restrict-to-ids id is bogus, it's a no-results run, not a partial one."""
    tmp_path, a, b, c = cli_corpus
    bogus = max(a, b, c) + 1000
    runner = CliRunner()
    result = runner.invoke(
        cli, ["search", "climate", "--fts", "--restrict-to-ids", str(bogus), "--json"]
    )
    assert result.exit_code == 2  # EXIT_NO_RESULTS
    payload = json.loads(result.output)
    assert payload["not_found"] == [bogus]


def test_search_ids_only_reports_not_found_to_stderr(cli_corpus):
    """--ids-only is a pure stdout pipe; not_found goes to stderr, not stdout."""
    tmp_path, a, b, c = cli_corpus
    bogus = max(a, b, c) + 1000
    runner = CliRunner()
    result = runner.invoke(
        cli, ["search", "climate", "--fts", "--restrict-to-ids", f"{a},{bogus}", "--ids-only"]
    )
    ids = [int(ln) for ln in result.stdout.splitlines() if ln.strip()]
    assert ids == [a]
    assert str(bogus) in result.stderr
    assert result.exit_code == 3


def test_semantic_restrict_to_ids_reports_not_found(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    bogus = max(a, b, c) + 1000
    runner = CliRunner()
    result = runner.invoke(
        cli, ["semantic", "climate", "--doc-only", "--restrict-to-ids", f"{a},{bogus}", "--json"]
    )
    payload = json.loads(result.output)
    assert payload["not_found"] == [bogus]
    assert {r["id"] for r in payload["results"]} == {a}
    assert result.exit_code == 3


def test_semantic_ids_only_reports_not_found_to_stderr(cli_corpus):
    tmp_path, a, b, c = cli_corpus
    bogus = max(a, b, c) + 1000
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["semantic", "climate", "--doc-only", "--restrict-to-ids", f"{a},{bogus}", "--ids-only"],
    )
    ids = [int(ln) for ln in result.stdout.splitlines() if ln.strip()]
    assert ids == [a]
    assert str(bogus) in result.stderr
    assert result.exit_code == 3
