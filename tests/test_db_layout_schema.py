"""Schema v8: layout-title columns."""

import sqlite3

from pdf_gantry.db import SCHEMA_VERSION, get_connection, get_schema_version


LAYOUT_COLUMNS = ("title_from_layout", "title_layout_confidence", "title_layout_at")


def _cols(conn):
    return {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}


def test_fresh_db_has_layout_columns(tmp_db):
    assert SCHEMA_VERSION == 8
    assert set(LAYOUT_COLUMNS) <= _cols(tmp_db)


def test_v7_db_migrates_to_v8_keeping_rows(tmp_path):
    path = tmp_path / "old.db"
    conn = get_connection(path)
    for col in LAYOUT_COLUMNS:
        conn.execute(f"ALTER TABLE papers DROP COLUMN {col}")
    conn.execute("DELETE FROM schema_version WHERE version = 8")
    conn.execute("INSERT INTO schema_version VALUES (7, 'then')")
    conn.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at) VALUES ('a.pdf', 'a.pdf', 'h', 1, 'm', 'n', 'n')"
    )
    conn.commit()
    conn.close()

    conn = get_connection(path)
    assert get_schema_version(conn) == 8
    assert set(LAYOUT_COLUMNS) <= _cols(conn)
    row = conn.execute("SELECT filename, title_from_layout FROM papers").fetchone()
    assert row["filename"] == "a.pdf" and row["title_from_layout"] is None
    # v_papers sees the new column without being recreated.
    assert "title_from_layout" in {
        r[1] for r in conn.execute("PRAGMA table_info(v_papers)").fetchall()
    }


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "x.db"
    get_connection(path).close()
    conn = get_connection(path)
    assert get_schema_version(conn) == 8
    assert isinstance(conn, sqlite3.Connection)
