"""Tests for database connection, schema, and migrations."""

import sqlite3

import pytest

from pdf_gantry.db import get_connection, get_schema_version, SCHEMA_VERSION


def test_schema_creation(tmp_path):
    """Schema is created on fresh database."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    # Check papers table exists
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = [t["name"] for t in tables]
    assert "papers" in table_names
    assert "paper_text" in table_names
    assert "schema_version" in table_names

    conn.close()


def test_sqlite_vec_loaded(tmp_db):
    """sqlite-vec extension is loaded and functional."""
    # vec0 virtual table should exist
    row = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_embeddings'"
    ).fetchone()
    assert row is not None


def test_wal_mode(tmp_db):
    """WAL journal mode is active."""
    row = tmp_db.execute("PRAGMA journal_mode").fetchone()
    assert row[0] == "wal"


def test_foreign_keys_on(tmp_db):
    """Foreign keys are enabled."""
    row = tmp_db.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1


def test_schema_version(tmp_db):
    """Schema version is set after initialization."""
    version = get_schema_version(tmp_db)
    assert version == SCHEMA_VERSION


def test_fts5_table_exists(tmp_db):
    """FTS5 virtual table is created and queryable."""
    # Insert a test row
    tmp_db.execute(
        "INSERT INTO papers_fts(rowid, filename, title, authors, abstract, text_content) "
        "VALUES (1, 'test.pdf', 'Test Title', 'Author', 'Abstract text', 'Full text content')"
    )
    tmp_db.commit()

    # Query it
    rows = tmp_db.execute(
        "SELECT * FROM papers_fts WHERE papers_fts MATCH 'test'"
    ).fetchall()
    assert len(rows) >= 1


def test_vec0_table_exists(tmp_db):
    """vec0 virtual table is created."""
    # Just verify the table exists by trying a no-op query
    row = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE name='paper_embeddings'"
    ).fetchone()
    assert row is not None


def test_chunks_table_exists(tmp_db):
    """Chunks table is created in schema v2."""
    row = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone()
    assert row is not None


def test_chunk_vec_table_exists(tmp_db):
    """chunk_vec virtual table is created."""
    row = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE name='chunk_vec'"
    ).fetchone()
    assert row is not None


def test_has_chunk_embeddings_column(tmp_db):
    """papers table has has_chunk_embeddings column."""
    tmp_db.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, indexed_at, updated_at) "
        "VALUES ('test.pdf', 'test.pdf', 'abc123', 1000, '2024-01-01', '2024-01-01', '2024-01-01')"
    )
    tmp_db.commit()
    row = tmp_db.execute("SELECT has_chunk_embeddings FROM papers WHERE path = 'test.pdf'").fetchone()
    assert row["has_chunk_embeddings"] == 0


def test_chunks_foreign_key_cascade(tmp_db):
    """Deleting a paper cascades to its chunks."""
    tmp_db.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, indexed_at, updated_at) "
        "VALUES ('test.pdf', 'test.pdf', 'abc123', 1000, '2024-01-01', '2024-01-01', '2024-01-01')"
    )
    paper_id = tmp_db.execute("SELECT id FROM papers WHERE path = 'test.pdf'").fetchone()["id"]
    tmp_db.execute(
        "INSERT INTO chunks (doc_id, chunk_index, text, char_offset) VALUES (?, 0, 'some text', 0)",
        (paper_id,),
    )
    tmp_db.commit()
    assert tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1

    tmp_db.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
    tmp_db.commit()
    assert tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


def test_idempotent_connection(tmp_path):
    """Connecting to an existing database doesn't re-create schema."""
    db_path = tmp_path / "test.db"
    conn1 = get_connection(str(db_path))
    conn1.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, indexed_at, updated_at) "
        "VALUES ('test.pdf', 'test.pdf', 'abc123', 1000, '2024-01-01', '2024-01-01', '2024-01-01')"
    )
    conn1.commit()
    conn1.close()

    conn2 = get_connection(str(db_path))
    row = conn2.execute("SELECT COUNT(*) FROM papers").fetchone()
    assert row[0] == 1
    conn2.close()
