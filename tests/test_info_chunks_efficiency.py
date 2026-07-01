"""Tests for info --chunks: correctness and O(1) query-count efficiency.

RED (before fix): the inner loop in the `info` command issues one SELECT per
paper, so N papers → N+1 DB queries. This test asserts ≤2 queries, which is
RED with the old code and GREEN after the single-JOIN fix.
"""

import json

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection


def _add_paper(conn, filename: str) -> int:
    cur = conn.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at) "
        "VALUES (?, ?, 'hash', 100, '2024-01-01', '2024-01-01', '2024-01-01')",
        (f"/papers/{filename}", filename),
    )
    return cur.lastrowid


def _add_chunk(conn, doc_id: int, index: int, text: str, header: str = "sec") -> int:
    cur = conn.execute(
        "INSERT INTO chunks (doc_id, chunk_index, section_header, page_start, text) "
        "VALUES (?, ?, ?, ?, ?)",
        (doc_id, index, header, index, text),
    )
    return cur.lastrowid


@pytest.fixture
def multi_paper_db(tmp_path, monkeypatch):
    """n papers, each with multiple chunks. CLI-configured via monkeypatched config."""
    n = 5
    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))

    paper_ids = []
    for i in range(n):
        pid = _add_paper(conn, f"paper_{i}.pdf")
        paper_ids.append(pid)
        for j in range(3):
            _add_chunk(conn, pid, j, f"paper {i} chunk {j} text", header=f"Section {j}")
    conn.commit()
    conn.close()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)

    return tmp_path, paper_ids


# ---------------------------------------------------------------------------
# Correctness: --chunks returns the right data
# ---------------------------------------------------------------------------


def test_info_chunks_returns_all_chunks(multi_paper_db):
    """info --chunks includes every chunk for each requested paper."""
    tmp_path, paper_ids = multi_paper_db
    runner = CliRunner()
    ids_str = ",".join(str(pid) for pid in paper_ids[:2])

    result = runner.invoke(cli, ["info", "--ids", ids_str, "--chunks", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    by_id = {p["id"]: p for p in payload["papers"]}

    for pid in paper_ids[:2]:
        assert pid in by_id, f"Paper {pid} missing from response"
        chunks = by_id[pid]["chunks"]
        assert len(chunks) == 3, f"Expected 3 chunks for paper {pid}, got {len(chunks)}"
        texts = [c["text"] for c in chunks]
        for j in range(3):
            assert f"chunk {j}" in " ".join(texts)


def test_info_chunks_ordering(multi_paper_db):
    """Chunks are returned in chunk_index order."""
    tmp_path, paper_ids = multi_paper_db
    runner = CliRunner()

    result = runner.invoke(
        cli, ["info", "--ids", str(paper_ids[0]), "--chunks", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    chunks = payload["papers"][0]["chunks"]
    indices = [c["chunk_index"] for c in chunks]
    assert indices == sorted(indices), "Chunks should be ordered by chunk_index"


def test_info_without_chunks_flag_omits_chunks(multi_paper_db):
    """Without --chunks, the chunks key is absent from the response."""
    tmp_path, paper_ids = multi_paper_db
    runner = CliRunner()

    result = runner.invoke(
        cli, ["info", "--ids", str(paper_ids[0]), "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "chunks" not in payload["papers"][0]


# ---------------------------------------------------------------------------
# Efficiency: O(1) queries regardless of N
# ---------------------------------------------------------------------------


def test_info_chunks_query_count_is_constant(tmp_path, monkeypatch):
    """info --chunks issues ≤2 DB queries regardless of paper count (not N+1).

    This was the RED test before the fix: with num_papers=10 papers the old code
    issued 11 queries (1 for papers + one per paper for chunks). The fixed code
    issues exactly 2 (1 for papers + 1 for all chunks via IN).
    """
    num_papers = 10
    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))

    paper_ids = []
    for i in range(num_papers):
        pid = _add_paper(conn, f"paper_{i}.pdf")
        paper_ids.append(pid)
        for j in range(5):
            _add_chunk(conn, pid, j, f"p{i} c{j}")
    conn.commit()
    conn.close()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)

    # Intercept get_connection to wrap it with a query counter
    import pdf_gantry.cli as cli_mod
    import pdf_gantry.db as db_mod

    query_log: list[str] = []

    real_get_connection = db_mod.get_connection

    class _CountingConn:
        """Thin wrapper counting every execute() call on the underlying conn."""
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, params=()):
            query_log.append(sql.strip().split("\n")[0])
            return self._inner.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def patched_get_connection(path):
        return _CountingConn(real_get_connection(path))

    monkeypatch.setattr(cli_mod, "get_connection", patched_get_connection)

    runner = CliRunner()
    ids_str = ",".join(str(pid) for pid in paper_ids)
    result = runner.invoke(cli, ["info", "--ids", ids_str, "--chunks", "--json"])
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    payload = json.loads(result.output)
    # Verify correctness: all papers present with 5 chunks each
    assert payload["count"] == num_papers
    for paper in payload["papers"]:
        assert len(paper["chunks"]) == 5

    # --- THE EFFICIENCY INVARIANT ---
    # Filter to SELECT queries only (ignore PRAGMA, BEGIN etc.)
    select_queries = [q for q in query_log if q.upper().startswith("SELECT")]
    assert len(select_queries) <= 2, (
        f"Expected ≤2 SELECT queries for {num_papers} papers with --chunks; "
        f"got {len(select_queries)}.\n"
        f"Queries issued:\n" + "\n".join(f"  {q}" for q in select_queries)
    )
