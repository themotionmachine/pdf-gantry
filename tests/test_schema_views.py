"""Stable read surface for raw-SQL users (E11).

Agents made 141 schema probes and hit 47 `no such column` errors guessing
`chunks.id`, `paper_id` and `paper_text.text`. Two views give them stable
names (`v_papers`, `v_chunks`, both keyed by `paper_id`), and `gantry
schema` prints tables, columns and the common joins in one call.
"""

import json

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import SCHEMA_VERSION, describe_schema, get_connection, get_schema_version
from tests.test_db import _make_v1_db


def _cols(conn, name):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({name})").fetchall()]


def _views(conn):
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'view'"
    ).fetchall()}


def test_fresh_db_has_views(tmp_db):
    assert {"v_papers", "v_chunks"} <= _views(tmp_db)


def test_v_chunks_columns(tmp_db):
    cols = _cols(tmp_db, "v_chunks")
    for c in ("paper_id", "chunk_id", "chunk_index", "text", "page_start",
              "section_header", "char_offset"):
        assert c in cols


def test_v_papers_columns(tmp_db):
    cols = _cols(tmp_db, "v_papers")
    for c in ("paper_id", "id", "title", "authors", "year", "doi", "citekey", "filename"):
        assert c in cols


def test_views_return_rows_keyed_by_paper_id(tmp_db):
    pid = tmp_db.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified,"
        " indexed_at, updated_at, title) VALUES ('/p/a.pdf','a.pdf','h',1,'x','x','x','T')"
    ).lastrowid
    tmp_db.execute(
        "INSERT INTO chunks (doc_id, chunk_index, text, page_start) VALUES (?, 0, 'hello', 3)",
        (pid,),
    )
    row = tmp_db.execute(
        "SELECT p.title, c.text, c.page_start FROM v_papers p "
        "JOIN v_chunks c ON c.paper_id = p.paper_id"
    ).fetchone()
    assert tuple(row) == ("T", "hello", 3)


def test_views_pick_up_later_page_columns(tmp_db):
    """A later migration adding e.g. chunks.page_end shows up in v_chunks."""
    tmp_db.execute("ALTER TABLE chunks ADD COLUMN page_end INTEGER")
    assert "page_end" in _cols(tmp_db, "v_chunks")


def test_views_do_not_bump_schema_version(tmp_db):
    assert get_schema_version(tmp_db) == SCHEMA_VERSION == 6


def test_reopen_is_idempotent(tmp_path):
    p = tmp_path / "x.db"
    get_connection(str(p)).close()
    conn = get_connection(str(p))
    assert {"v_papers", "v_chunks"} <= _views(conn)
    conn.close()


def test_migrated_v1_db_gets_views(tmp_path):
    p = tmp_path / "old.db"
    _make_v1_db(p)
    conn = get_connection(str(p))
    assert {"v_papers", "v_chunks"} <= _views(conn)
    assert "paper_id" in _cols(conn, "v_chunks")
    conn.close()


def test_existing_current_db_without_views_gets_them(tmp_path):
    p = tmp_path / "cur.db"
    conn = get_connection(str(p))
    conn.execute("DROP VIEW v_papers")
    conn.execute("DROP VIEW v_chunks")
    conn.commit()
    conn.close()
    conn = get_connection(str(p))
    assert {"v_papers", "v_chunks"} <= _views(conn)
    conn.close()


# --- describe_schema / gantry schema ----------------------------------------

def test_describe_schema_lists_tables_views_and_joins(tmp_db):
    info = describe_schema(tmp_db)
    names = {t["name"] for t in info["tables"]}
    assert {"papers", "chunks", "paper_text", "papers_fts", "chunk_vec"} <= names
    # sqlite-vec / FTS5 shadow tables are noise for a reader
    assert not any(n.startswith("chunk_vec_") or n.startswith("papers_fts_") for n in names)
    chunks = next(t for t in info["tables"] if t["name"] == "chunks")
    assert "doc_id" in [c["name"] for c in chunks["columns"]]
    assert {"v_papers", "v_chunks"} <= {v["name"] for v in info["views"]}
    rels = {(r["from"], r["to"]) for r in info["relationships"]}
    assert ("chunks.doc_id", "papers.id") in rels
    assert ("paper_text.paper_id", "papers.id") in rels
    assert info["common_joins"]
    assert info["schema_version"] == SCHEMA_VERSION


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    get_connection(str(tmp_path / "index.db")).close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg)


def test_schema_cli_json(cli_env):
    result = CliRunner().invoke(cli, ["schema", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"tables", "views", "relationships", "common_joins"} <= set(payload)


def test_schema_cli_text(cli_env):
    result = CliRunner().invoke(cli, ["schema"])
    assert result.exit_code == 0, result.output
    assert "v_chunks" in result.output
    assert "chunks.doc_id -> papers.id" in result.output


def test_schema_cli_no_db(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"index_dir": str(tmp_path / "nowhere")}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg)
    result = CliRunner().invoke(cli, ["schema", "--json"])
    assert result.exit_code == 1
