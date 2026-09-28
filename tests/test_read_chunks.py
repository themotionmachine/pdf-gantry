"""`gantry read` chunk addressing.

B2: `read 53 --chunk 5` returned chunk 5 of doc 1 with exit 0 -- the
identifier was ignored, so an agent could attribute a quote to the wrong
paper. The identifier is now resolved and a mismatched chunk is an error.

E8: `read ID --index N` / `--index A-B` addresses chunks by their per-paper
chunk_index, replacing the `info --chunks` + filter-in-Python pattern.
"""

import json

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.reader import chunks_by_index, parse_index_range


def _paper(conn, filename, citekey=None):
    cur = conn.execute(
        "INSERT INTO papers "
        "(path, filename, file_hash, file_size, file_modified, indexed_at, updated_at,"
        " citekey, title) "
        "VALUES (?, ?, 'h', 1, '2024', '2024', '2024', ?, ?)",
        (f"/papers/{filename}", filename, citekey, filename.upper()),
    )
    return cur.lastrowid


def _chunks(conn, doc_id, n):
    ids = []
    offset = 0
    for i in range(n):
        text = f"doc{doc_id} chunk{i} body"
        cur = conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, section_header, page_start, text,"
            " char_offset) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, i, f"S{i}", i + 1, text, offset),
        )
        offset += len(text) + 2
        ids.append(cur.lastrowid)
    return ids


@pytest.fixture
def env(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    a = _paper(conn, "a.pdf", citekey="alpha2020")
    b = _paper(conn, "b.pdf")
    a_chunks = _chunks(conn, a, 4)
    b_chunks = _chunks(conn, b, 6)
    conn.commit()
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg_file)
    yield conn, a, b, a_chunks, b_chunks
    conn.close()


def _run(*args):
    return CliRunner().invoke(cli, list(args))


# --- B2: --chunk must belong to the identified paper ------------------------

def test_read_chunk_of_other_paper_is_error(env):
    _, a, b, a_chunks, b_chunks = env
    result = _run("read", str(b), "--chunk", str(a_chunks[1]), "--json")
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert "belongs to paper" in payload["error"]


def test_read_chunk_of_other_paper_with_context_is_error(env):
    _, a, b, a_chunks, _ = env
    result = _run("read", str(b), "--chunk", str(a_chunks[1]), "--context", "200", "--json")
    assert result.exit_code == 1


def test_read_chunk_of_same_paper_ok_and_has_ids(env):
    _, a, _, a_chunks, _ = env
    result = _run("read", str(a), "--chunk", str(a_chunks[2]), "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["doc_id"] == a
    assert payload["paper_id"] == a
    assert payload["id"] == a
    assert payload["chunk_index"] == 2


def test_read_chunk_unknown_paper_is_error(env):
    _, _, _, a_chunks, _ = env
    result = _run("read", "9999", "--chunk", str(a_chunks[0]), "--json")
    assert result.exit_code == 1
    assert "not found" in json.loads(result.output)["error"].lower()


def test_read_accepts_citekey(env):
    _, a, _, a_chunks, _ = env
    result = _run("read", "@alpha2020", "--chunk", str(a_chunks[0]), "--json")
    assert result.exit_code == 0, result.output


# --- id alongside paper_id --------------------------------------------------

def test_read_chunks_list_has_id(env):
    _, a, _, _, _ = env
    payload = json.loads(_run("read", str(a), "--chunks", "--json").output)
    assert payload["id"] == payload["paper_id"] == a


# --- E8: --index -------------------------------------------------------------

def test_parse_index_range():
    assert parse_index_range("5") == (5, 5)
    assert parse_index_range("2-4") == (2, 4)
    assert parse_index_range(" 2 - 4 ") == (2, 4)
    for bad in ("", "a", "4-2", "-1", "1-", "1-2-3"):
        with pytest.raises(ValueError):
            parse_index_range(bad)


def test_chunks_by_index_returns_ordered_range_with_bounds(env):
    conn, _, b, _, _ = env
    rows = chunks_by_index(conn, b, 1, 3)
    assert [r["chunk_index"] for r in rows] == [1, 2, 3]
    r = rows[0]
    assert r["text"] == f"doc{b} chunk1 body"
    assert r["char_end"] == r["char_offset"] + len(r["text"])
    assert r["page_start"] == 2
    assert "chunk_id" in r and "section_header" in r


def test_read_index_single(env):
    _, a, _, _, _ = env
    result = _run("read", str(a), "--index", "2", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["id"] == payload["paper_id"] == a
    assert payload["total_chunks"] == 4
    assert [c["chunk_index"] for c in payload["chunks"]] == [2]
    assert payload["missing_indices"] == []


def test_read_index_range(env):
    _, _, b, _, _ = env
    payload = json.loads(_run("read", str(b), "--index", "1-4", "--json").output)
    assert [c["chunk_index"] for c in payload["chunks"]] == [1, 2, 3, 4]


def test_read_index_partially_out_of_range_is_partial(env):
    _, a, _, _, _ = env
    result = _run("read", str(a), "--index", "2-6", "--json")
    assert result.exit_code == 3
    payload = json.loads(result.output)
    assert [c["chunk_index"] for c in payload["chunks"]] == [2, 3]
    assert payload["missing_indices"] == [4, 5, 6]


def test_read_index_fully_out_of_range_is_error(env):
    _, a, _, _, _ = env
    result = _run("read", str(a), "--index", "10-12", "--json")
    assert result.exit_code == 1
    assert "4 chunks" in json.loads(result.output)["error"]


def test_read_index_bad_value_is_usage_error(env):
    _, a, _, _, _ = env
    result = _run("read", str(a), "--index", "x-y", "--json")
    assert result.exit_code == 64


def test_read_index_and_chunk_are_exclusive(env):
    _, a, _, a_chunks, _ = env
    result = _run("read", str(a), "--index", "1", "--chunk", str(a_chunks[0]))
    assert result.exit_code == 64


def test_read_index_plain_text(env):
    _, a, _, _, _ = env
    result = _run("read", str(a), "--index", "0-1")
    assert result.exit_code == 0
    assert f"doc{a} chunk0 body" in result.output
    assert f"doc{a} chunk1 body" in result.output
