"""Tests for batch context retrieval: best chunk per document for a query."""

import json
import struct

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection


def _vec(axis: int, dim: int = 768) -> bytes:
    """A unit vector pointing along a single axis, serialized for sqlite-vec."""
    v = [0.0] * dim
    v[axis] = 1.0
    return struct.pack(f"{dim}f", *v)


def _add_paper(conn, filename: str) -> int:
    cur = conn.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at, has_chunk_embeddings) "
        "VALUES (?, ?, 'hash', 100, '2024-01-01', '2024-01-01', '2024-01-01', 1)",
        (f"/papers/{filename}", filename),
    )
    return cur.lastrowid


def _add_chunk(conn, doc_id: int, index: int, text: str, axis: int) -> int:
    cur = conn.execute(
        "INSERT INTO chunks (doc_id, chunk_index, section_header, page_start, text) "
        "VALUES (?, ?, ?, ?, ?)",
        (doc_id, index, f"sec{index}", index, text),
    )
    chunk_id = cur.lastrowid
    conn.execute(
        "INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, _vec(axis)),
    )
    return chunk_id


@pytest.fixture
def chunked_db(tmp_path):
    """Two papers, each with three chunks aligned to distinct embedding axes."""
    conn = get_connection(str(tmp_path / "test.db"))
    doc_a = _add_paper(conn, "alpha.pdf")
    doc_b = _add_paper(conn, "beta.pdf")
    # doc_a chunks on axes 0,1,2 ; doc_b chunks on axes 3,4,5
    _add_chunk(conn, doc_a, 0, "alpha chunk zero", 0)
    a1 = _add_chunk(conn, doc_a, 1, "alpha chunk one", 1)
    _add_chunk(conn, doc_a, 2, "alpha chunk two", 2)
    _add_chunk(conn, doc_b, 0, "beta chunk zero", 3)
    b1 = _add_chunk(conn, doc_b, 1, "beta chunk one", 4)
    _add_chunk(conn, doc_b, 2, "beta chunk two", 5)
    conn.commit()
    return conn, doc_a, doc_b, a1, b1


def test_best_chunk_per_doc_one_per_doc(chunked_db):
    """Returns exactly one chunk per requested document."""
    from pdf_gantry.search import best_chunk_per_doc

    conn, doc_a, doc_b, _, _ = chunked_db
    best = best_chunk_per_doc(conn, _vec(1), [doc_a, doc_b])
    assert set(best.keys()) == {doc_a, doc_b}


def test_best_chunk_per_doc_picks_closest(chunked_db):
    """The chunk whose embedding matches the query is selected for its doc."""
    from pdf_gantry.search import best_chunk_per_doc

    conn, doc_a, doc_b, a1, _ = chunked_db
    # Query along axis 1 -> doc_a's best chunk is the axis-1 chunk (cosine distance 0)
    best = best_chunk_per_doc(conn, _vec(1), [doc_a, doc_b])
    assert best[doc_a].chunk_id == a1
    assert best[doc_a].score == pytest.approx(1.0, abs=1e-3)


def test_best_chunk_per_doc_is_scoped(chunked_db):
    """Documents outside the requested set are never returned."""
    from pdf_gantry.search import best_chunk_per_doc

    conn, doc_a, doc_b, _, _ = chunked_db
    best = best_chunk_per_doc(conn, _vec(4), [doc_b])
    assert set(best.keys()) == {doc_b}


def test_best_chunk_per_doc_empty_ids(chunked_db):
    """Empty id list yields an empty result without touching the DB."""
    from pdf_gantry.search import best_chunk_per_doc

    conn, _, _, _, _ = chunked_db
    assert best_chunk_per_doc(conn, _vec(0), []) == {}


def test_best_chunk_per_doc_skips_unchunked(chunked_db):
    """Docs with no chunk embeddings are omitted, not errored."""
    from pdf_gantry.search import best_chunk_per_doc

    conn, doc_a, _, _, _ = chunked_db
    bare = _add_paper(conn, "gamma.pdf")  # no chunks
    conn.commit()
    best = best_chunk_per_doc(conn, _vec(0), [doc_a, bare])
    assert bare not in best
    assert doc_a in best


# --- CLI: info --query (batch top-chunk retrieval) ---


@pytest.fixture
def cli_chunked_db(tmp_path, monkeypatch):
    """A configured DB (at index_dir/index.db) with chunked papers for CLI tests."""
    conn = get_connection(str(tmp_path / "index.db"))
    doc_a = _add_paper(conn, "alpha.pdf")
    doc_b = _add_paper(conn, "beta.pdf")
    _add_chunk(conn, doc_a, 0, "alpha chunk zero", 0)
    _add_chunk(conn, doc_a, 1, "alpha chunk one", 1)
    _add_chunk(conn, doc_b, 0, "beta chunk zero", 3)
    _add_chunk(conn, doc_b, 1, "beta chunk one", 4)
    conn.commit()
    conn.close()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)
    # Avoid loading the real embedding model: query along axis 1.
    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", lambda model, q: _vec(1))
    return tmp_path, doc_a, doc_b


def test_info_query_attaches_top_chunk(cli_chunked_db):
    tmp_path, doc_a, doc_b = cli_chunked_db
    runner = CliRunner()
    result = runner.invoke(
        cli, ["info", "--ids", f"{doc_a},{doc_b}", "--query", "anything", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    by_id = {p["id"]: p for p in payload["papers"]}
    assert "top_chunk" in by_id[doc_a]
    assert by_id[doc_a]["top_chunk"]["text"] == "alpha chunk one"
    assert by_id[doc_a]["top_chunk"]["score"] == pytest.approx(1.0, abs=1e-3)


def test_info_without_query_has_no_top_chunk(cli_chunked_db):
    tmp_path, doc_a, doc_b = cli_chunked_db
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--ids", str(doc_a), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "top_chunk" not in payload["papers"][0]


def test_info_query_respects_fields(cli_chunked_db):
    tmp_path, doc_a, _ = cli_chunked_db
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["info", "--ids", str(doc_a), "--query", "x", "--fields", "id,top_chunk", "--json"],
    )
    assert result.exit_code == 0
    paper = json.loads(result.output)["papers"][0]
    assert set(paper.keys()) == {"id", "top_chunk"}


# --- CLI: info --query --context (context-expanded batch retrieval) ---


def test_info_query_context_expands_window(cli_chunked_db):
    """--context N combines best_chunk_per_doc with get_chunk_context in one call.

    Proves the chain: search --ids-only | info --ids --query X --context N
    returns context-expanded excerpts for all N papers in two CLI calls total
    instead of N+2.  The fixture has two chunks per doc; with max_chars=5000
    both neighbours should appear in the expanded context.
    """
    tmp_path, doc_a, doc_b = cli_chunked_db
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["info", "--ids", f"{doc_a},{doc_b}", "--query", "anything",
         "--context", "5000", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    by_id = {p["id"]: p for p in payload["papers"]}

    # doc_a: query on axis 1 → best chunk is "alpha chunk one" (index 1).
    tc_a = by_id[doc_a]["top_chunk"]
    assert "context" in tc_a
    # Context must contain the target chunk's own text.
    assert tc_a["text"] in tc_a["context"]
    # With max_chars=5000 both chunks easily fit; the window expands to include index 0.
    assert "alpha chunk zero" in tc_a["context"]
    assert "alpha chunk one" in tc_a["context"]
    # total_chunks lets the agent know the doc's chunk count without a second call.
    assert tc_a["total_chunks"] == 2


def test_info_query_without_context_unchanged(cli_chunked_db):
    """Without --context the top_chunk shape is identical to the prior contract."""
    tmp_path, doc_a, _ = cli_chunked_db
    runner = CliRunner()
    result = runner.invoke(
        cli, ["info", "--ids", str(doc_a), "--query", "anything", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    tc = payload["papers"][0]["top_chunk"]
    assert "context" not in tc
    assert "total_chunks" not in tc


def test_info_context_without_query_is_ignored(cli_chunked_db):
    """--context without --query does not crash and produces no top_chunk."""
    tmp_path, doc_a, _ = cli_chunked_db
    runner = CliRunner()
    result = runner.invoke(
        cli, ["info", "--ids", str(doc_a), "--context", "1000", "--json"],
    )
    assert result.exit_code == 0
    paper = json.loads(result.output)["papers"][0]
    assert "top_chunk" not in paper
