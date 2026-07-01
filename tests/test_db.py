"""Tests for database connection, schema, and migrations."""

import os
import sqlite3

import sqlite_vec

from pdf_gantry.db import SCHEMA_VERSION, get_connection, get_schema_version

assert SCHEMA_VERSION == 4, "Update tests if schema version changes"


# ---------------------------------------------------------------------------
# Helpers for migration-convergence tests
# ---------------------------------------------------------------------------

def _make_v1_db(path):
    """Construct a schema v1 database (pre-chunks) for migration-path testing.

    Simulates a real user DB from before chunk embeddings landed: papers table
    without has_chunk_embeddings, old-style paper_embeddings vec0 (no cosine
    metric), no chunks table, no chunk_vec.  schema_version is 1.
    """
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # v1 papers table — every column that existed before has_chunk_embeddings
    conn.executescript("""
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_modified TEXT NOT NULL,
            page_count INTEGER,
            has_text INTEGER NOT NULL DEFAULT 0,
            has_markdown INTEGER NOT NULL DEFAULT 0,
            has_embeddings INTEGER NOT NULL DEFAULT 0,
            needs_ocr INTEGER NOT NULL DEFAULT 0,
            is_scanned INTEGER,
            text_method TEXT,
            text_extracted_at TEXT,
            markdown_method TEXT,
            markdown_extracted_at TEXT,
            embedding_model TEXT,
            embedding_model_version TEXT,
            embedding_computed_at TEXT,
            ocr_method TEXT,
            ocr_completed_at TEXT,
            title TEXT,
            authors TEXT,
            year INTEGER,
            doi TEXT,
            abstract TEXT,
            semantic_scholar_id TEXT,
            metadata_source TEXT,
            metadata_enriched_at TEXT,
            vault_note_path TEXT,
            vault_checked_at TEXT,
            indexed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_error TEXT,
            error_count INTEGER NOT NULL DEFAULT 0,
            last_error_at TEXT
        );
        CREATE VIRTUAL TABLE papers_fts USING fts5(
            filename, title, authors, abstract, text_content,
            content='', tokenize='porter unicode61'
        );
        CREATE TABLE paper_text (
            paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
            raw_text TEXT,
            markdown TEXT,
            text_length INTEGER,
            markdown_length INTEGER
        );
        CREATE TABLE schema_version (
            version INTEGER NOT NULL,
            applied_at TEXT NOT NULL
        );
        INSERT INTO schema_version VALUES (1, '2024-01-01T00:00:00Z');
    """)
    # Old-style paper_embeddings without cosine metric (the bug v3 fixes)
    conn.execute("""CREATE VIRTUAL TABLE paper_embeddings USING vec0(
        paper_id INTEGER PRIMARY KEY,
        embedding FLOAT[768]
    )""")
    conn.commit()
    conn.close()


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
        "INSERT INTO papers (path, filename, file_hash, file_size, "
        "file_modified, indexed_at, updated_at) "
        "VALUES ('test.pdf', 'test.pdf', 'abc123', 1000, '2024-01-01', '2024-01-01', '2024-01-01')"
    )
    tmp_db.commit()
    row = tmp_db.execute(
        "SELECT has_chunk_embeddings FROM papers WHERE path = 'test.pdf'"
    ).fetchone()
    assert row["has_chunk_embeddings"] == 0


def test_chunks_foreign_key_cascade(tmp_db):
    """Deleting a paper cascades to its chunks."""
    tmp_db.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, "
        "file_modified, indexed_at, updated_at) "
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


def test_citekey_column_exists(tmp_db):
    """papers table has citekey and citekey_source columns (schema v4)."""
    tmp_db.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, "
        "file_modified, indexed_at, updated_at) "
        "VALUES ('test.pdf', 'test.pdf', 'abc123', 1000, '2024-01-01', '2024-01-01', '2024-01-01')"
    )
    tmp_db.commit()
    row = tmp_db.execute(
        "SELECT citekey, citekey_source FROM papers WHERE path = 'test.pdf'"
    ).fetchone()
    assert row["citekey"] is None
    assert row["citekey_source"] is None


def test_citekey_index_exists(tmp_db):
    """Index on citekey column exists (schema v4)."""
    row = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_papers_citekey'"
    ).fetchone()
    assert row is not None


def test_idempotent_connection(tmp_path):
    """Connecting to an existing database doesn't re-create schema."""
    db_path = tmp_path / "test.db"
    conn1 = get_connection(str(db_path))
    conn1.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, "
        "file_modified, indexed_at, updated_at) "
        "VALUES ('test.pdf', 'test.pdf', 'abc123', 1000, '2024-01-01', '2024-01-01', '2024-01-01')"
    )
    conn1.commit()
    conn1.close()

    conn2 = get_connection(str(db_path))
    row = conn2.execute("SELECT COUNT(*) FROM papers").fetchone()
    assert row[0] == 1
    conn2.close()


# ---------------------------------------------------------------------------
# Collapse/convergence tests — init path and migration path must agree
# ---------------------------------------------------------------------------

def test_ddl_constants_exist():
    """Canonical DDL constants for chunks tables are exported from db module.

    RED before the refactor (no such attributes), GREEN after.  The constants
    are the single source of truth used by both SCHEMA_SQL and migrate().
    """
    from pdf_gantry import db
    assert hasattr(db, "_CHUNKS_TABLE_DDL"), (
        "_CHUNKS_TABLE_DDL not found — DDL constants have not been extracted"
    )
    assert hasattr(db, "_CHUNK_VEC_DDL"), (
        "_CHUNK_VEC_DDL not found — DDL constants have not been extracted"
    )
    assert "char_offset" in db._CHUNKS_TABLE_DDL, (
        "char_offset column missing from _CHUNKS_TABLE_DDL"
    )
    assert "distance_metric=cosine" in db._CHUNK_VEC_DDL, (
        "cosine metric missing from _CHUNK_VEC_DDL"
    )


def test_schema_convergence_chunks(tmp_path):
    """Init path and migration path from v1 produce identical chunks column structure.

    Pins the invariant that the two code paths agree.  Before the DDL constants
    were extracted, this was true by coincidence; after extraction it's
    structurally guaranteed (one string, two call sites).
    """
    # Fresh-init database
    fresh_db = tmp_path / "fresh.db"
    conn_fresh = get_connection(str(fresh_db))
    fresh_cols = {
        row["name"]: (row["type"], row["notnull"], row["dflt_value"], row["pk"])
        for row in conn_fresh.execute("PRAGMA table_info(chunks)")
    }
    conn_fresh.close()

    # Database built from a v1 starting point, migrated to current version
    v1_db = tmp_path / "v1.db"
    _make_v1_db(v1_db)
    conn_migrated = get_connection(str(v1_db))
    migrated_cols = {
        row["name"]: (row["type"], row["notnull"], row["dflt_value"], row["pk"])
        for row in conn_migrated.execute("PRAGMA table_info(chunks)")
    }
    conn_migrated.close()

    assert fresh_cols == migrated_cols, (
        "Init path and migration path produced different chunks schemas — "
        "DDL is out of sync.\n"
        f"Fresh:    {sorted(fresh_cols)}\n"
        f"Migrated: {sorted(migrated_cols)}"
    )


# ---------------------------------------------------------------------------
# Security: DB file permissions
# ---------------------------------------------------------------------------

def test_new_db_is_owner_only(tmp_path):
    """A freshly-created DB file must be mode 0600 (owner-read/write only).

    sqlite3.connect() respects the process umask; the macOS default umask of
    022 yields 0644, which means any local user can read the entire research
    corpus (full text, abstracts, vault note paths, authors).  get_connection()
    must tighten this to 0600 immediately after the file is created, before any
    schema or data is written.
    """
    db_path = tmp_path / "new.db"
    assert not db_path.exists(), "precondition: file must not exist yet"
    conn = get_connection(str(db_path))
    conn.close()

    mode = os.stat(str(db_path)).st_mode & 0o777
    assert mode == 0o600, (
        f"DB was created with mode {oct(mode)}, expected 0o600 (owner-only). "
        "World-readable and group-readable bits expose the entire corpus."
    )


def test_existing_db_reopened_stays_owner_only(tmp_path):
    """Re-opening an existing 0600 DB does not widen its permissions.

    Reopening a correctly-restricted DB must not accidentally chmod it back to
    a broader mode.
    """
    db_path = tmp_path / "existing.db"
    # Create it the correct way (now 0600)
    conn = get_connection(str(db_path))
    conn.close()

    # Reopen
    conn2 = get_connection(str(db_path))
    conn2.close()

    mode = os.stat(str(db_path)).st_mode & 0o777
    assert mode == 0o600, (
        f"Reopening the DB changed mode to {oct(mode)} — expected 0o600."
    )
