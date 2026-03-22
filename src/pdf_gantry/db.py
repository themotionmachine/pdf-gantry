"""Database connection, schema creation, and migrations."""

import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = 3

SCHEMA_SQL = """
-- Core papers table
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_modified TEXT NOT NULL,
    page_count INTEGER,

    -- Processing state flags
    has_text INTEGER NOT NULL DEFAULT 0,
    has_markdown INTEGER NOT NULL DEFAULT 0,
    has_embeddings INTEGER NOT NULL DEFAULT 0,
    has_chunk_embeddings INTEGER NOT NULL DEFAULT 0,
    needs_ocr INTEGER NOT NULL DEFAULT 0,
    is_scanned INTEGER,

    -- Processing metadata
    text_method TEXT,
    text_extracted_at TEXT,
    markdown_method TEXT,
    markdown_extracted_at TEXT,
    embedding_model TEXT,
    embedding_model_version TEXT,
    embedding_computed_at TEXT,
    ocr_method TEXT,
    ocr_completed_at TEXT,

    -- External metadata
    title TEXT,
    authors TEXT,
    year INTEGER,
    doi TEXT,
    abstract TEXT,
    semantic_scholar_id TEXT,
    metadata_source TEXT,
    metadata_enriched_at TEXT,

    -- Vault integration
    vault_note_path TEXT,
    vault_checked_at TEXT,

    -- Timestamps
    indexed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    -- Error tracking
    last_error TEXT,
    error_count INTEGER NOT NULL DEFAULT 0,
    last_error_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_papers_hash ON papers(file_hash);
CREATE INDEX IF NOT EXISTS idx_papers_has_text ON papers(has_text);
CREATE INDEX IF NOT EXISTS idx_papers_has_markdown ON papers(has_markdown);
CREATE INDEX IF NOT EXISTS idx_papers_has_embeddings ON papers(has_embeddings);
CREATE INDEX IF NOT EXISTS idx_papers_needs_ocr ON papers(needs_ocr);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);

-- Full-text search virtual table
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    filename,
    title,
    authors,
    abstract,
    text_content,
    content='',
    tokenize='porter unicode61'
);

-- Extracted text storage
CREATE TABLE IF NOT EXISTS paper_text (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    raw_text TEXT,
    markdown TEXT,
    text_length INTEGER,
    markdown_length INTEGER
);

-- Embeddings via sqlite-vec (cosine distance)
CREATE VIRTUAL TABLE IF NOT EXISTS paper_embeddings USING vec0(
    paper_id INTEGER PRIMARY KEY,
    embedding FLOAT[768] distance_metric=cosine
);

-- Document chunks for chunk-level embeddings
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    section_header TEXT,
    page_start INTEGER,
    text TEXT NOT NULL,
    char_offset INTEGER NOT NULL DEFAULT 0,
    UNIQUE(doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

-- Chunk-level embeddings via sqlite-vec (cosine distance)
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[768] distance_metric=cosine
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL
);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with sqlite-vec, WAL mode, and schema initialized."""
    db_path = str(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Load sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Set pragmas
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    # Initialize schema if needed
    migrate(conn)

    return conn


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return current schema version, 0 if no schema_version table."""
    try:
        row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        return row[0] if row and row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables, indices, and virtual tables."""
    conn.executescript(SCHEMA_SQL)
    from .utils import now_iso
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, now_iso()),
    )
    conn.commit()


def migrate(conn: sqlite3.Connection) -> None:
    """Apply any pending schema migrations."""
    version = get_schema_version(conn)
    if version == 0:
        init_schema(conn)
        return

    from .utils import now_iso

    if version < 2:
        conn.execute(
            "ALTER TABLE papers ADD COLUMN has_chunk_embeddings INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            section_header TEXT,
            page_start INTEGER,
            text TEXT NOT NULL,
            char_offset INTEGER NOT NULL DEFAULT 0,
            UNIQUE(doc_id, chunk_index)
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        # v2 tables created with cosine metric (skipping L2 intermediate)
        conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[768] distance_metric=cosine
        )""")
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (2, now_iso()),
        )
        conn.commit()
        version = 2

    if version < 3:
        # Recreate vec0 tables with cosine distance metric.
        # vec0 tables can't be ALTERed — must drop and recreate.
        # Preserve existing vectors by reading them out first.

        # Migrate paper_embeddings
        existing_doc_vecs = conn.execute(
            "SELECT paper_id, embedding FROM paper_embeddings"
        ).fetchall()
        conn.execute("DROP TABLE IF EXISTS paper_embeddings")
        conn.execute("""CREATE VIRTUAL TABLE paper_embeddings USING vec0(
            paper_id INTEGER PRIMARY KEY,
            embedding FLOAT[768] distance_metric=cosine
        )""")
        for row in existing_doc_vecs:
            conn.execute(
                "INSERT INTO paper_embeddings (paper_id, embedding) VALUES (?, ?)",
                (row[0], row[1]),
            )

        # Migrate chunk_vec
        existing_chunk_vecs = conn.execute(
            "SELECT chunk_id, embedding FROM chunk_vec"
        ).fetchall()
        conn.execute("DROP TABLE IF EXISTS chunk_vec")
        conn.execute("""CREATE VIRTUAL TABLE chunk_vec USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[768] distance_metric=cosine
        )""")
        for row in existing_chunk_vecs:
            conn.execute(
                "INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                (row[0], row[1]),
            )

        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (3, now_iso()),
        )
        conn.commit()
