"""Database connection, schema creation, and migrations."""

import os
import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = 6

# ---------------------------------------------------------------------------
# Canonical DDL for tables that are created both by init_schema (fresh DB) and
# by migrate() (upgrading DB).  A single constant means the two paths can't
# silently drift apart — changing the DDL here changes both call sites at once.
# ---------------------------------------------------------------------------

_CHUNKS_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    section_header TEXT,
    page_start INTEGER,
    text TEXT NOT NULL,
    char_offset INTEGER NOT NULL DEFAULT 0,
    UNIQUE(doc_id, chunk_index)
)\
"""

_CHUNK_VEC_DDL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[768] distance_metric=cosine
)\
"""

SCHEMA_SQL = f"""
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

    -- Set at ingest time when fitz reports needs_pass on the file (a locked
    -- DRM'd export or accidentally-encrypted download). classify_document()
    -- still reports these as "digital" (Round 2) so is_scanned/needs_ocr
    -- selection is unaffected, but is_encrypted makes the "will never
    -- extract" fact queryable (`queue --is encrypted`) and lets the default
    -- process/pipeline sweeps skip them instead of burning quarantine
    -- retries rediscovering it.
    is_encrypted INTEGER NOT NULL DEFAULT 0,

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

    -- Metadata verification: title-search-sourced metadata is accepted from
    -- the provider with no confidence check at enrich time (top result wins
    -- unconditionally). metadata_suspect flags matches whose stored title
    -- doesn't resemble the title actually printed on the PDF, so a silently
    -- wrong match becomes a loud, queryable, pipeable fact instead of an
    -- invisible one. Set by verify_documents() in metadata.py.
    metadata_suspect INTEGER NOT NULL DEFAULT 0,
    metadata_verify_score REAL,
    metadata_verified_at TEXT,

    -- Vault integration
    vault_note_path TEXT,
    vault_checked_at TEXT,

    -- Bibliography linking
    citekey TEXT,
    citekey_source TEXT,

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
CREATE INDEX IF NOT EXISTS idx_papers_citekey ON papers(citekey);
CREATE INDEX IF NOT EXISTS idx_papers_metadata_suspect ON papers(metadata_suspect);

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
{_CHUNKS_TABLE_DDL};

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

-- Chunk-level embeddings via sqlite-vec (cosine distance)
{_CHUNK_VEC_DDL};

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL
);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with sqlite-vec, WAL mode, and schema initialized."""
    db_path = str(db_path)
    _is_new = not Path(db_path).exists()
    conn = sqlite3.connect(db_path)
    # sqlite3.connect() respects the process umask; on macOS the default umask
    # (022) yields mode 0644 — world-readable.  The DB holds the user's entire
    # research corpus (full text, abstracts, vault note paths).  Restrict it to
    # owner-only (0600) immediately after creation, before any data is written.
    if _is_new:
        os.chmod(db_path, 0o600)
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
        conn.execute(_CHUNKS_TABLE_DDL)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        conn.execute(_CHUNK_VEC_DDL)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (2, now_iso()),
        )
        conn.commit()
        version = 2

    if version < 3:
        # Recreate paper_embeddings with cosine distance metric.
        # vec0 tables can't be ALTERed — must drop and recreate.
        # Preserve existing vectors by reading them out first.
        #
        # Note: chunk_vec was introduced in v2 already with distance_metric=cosine
        # (_CHUNK_VEC_DDL), so it does NOT need to be touched here.
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

        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (3, now_iso()),
        )
        conn.commit()
        version = 3

    if version < 4:
        conn.execute("ALTER TABLE papers ADD COLUMN citekey TEXT")
        conn.execute("ALTER TABLE papers ADD COLUMN citekey_source TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_citekey ON papers(citekey)"
        )
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (4, now_iso()),
        )
        conn.commit()
        version = 4

    if version < 5:
        conn.execute(
            "ALTER TABLE papers ADD COLUMN metadata_suspect INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute("ALTER TABLE papers ADD COLUMN metadata_verify_score REAL")
        conn.execute("ALTER TABLE papers ADD COLUMN metadata_verified_at TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_metadata_suspect "
            "ON papers(metadata_suspect)"
        )
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (5, now_iso()),
        )
        conn.commit()
        version = 5

    if version < 6:
        conn.execute(
            "ALTER TABLE papers ADD COLUMN is_encrypted INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (6, now_iso()),
        )
        conn.commit()
