"""The keyword index (``papers_fts``): one writer path and an explicit rebuild.

``papers_fts`` is a contentless FTS5 table over filename, title, authors,
abstract and extracted text. It used to be written only at extraction time,
which runs before ``enrich``, so metadata added later never reached keyword
search. And a contentless FTS5 'delete' must be handed the exact values that
were indexed: callers passed the *current* values, which after enrichment
differ, so re-processing an enriched paper corrupted the posting lists.

The fix is ``contentless_delete=1`` (SQLite 3.43+), which deletes by rowid
with no values at all. Every writer goes through ``refresh_row`` /
``delete_row`` here. A paper has an FTS row iff it has a ``paper_text`` row.

Databases created before this carry the legacy table. Converting one is an
explicit ``gantry fts rebuild`` rather than an automatic migration, because
rebuilding ~2000 papers' text on first connect would stall a read-only call.
Until then the helpers fall back to the old behaviour and log once.
"""

import logging
import re
import sqlite3
import time

logger = logging.getLogger(__name__)

TABLE = "papers_fts"
TEMP_TABLE = "gantry_fts_rebuild"
COLUMNS = ("filename", "title", "authors", "abstract", "text_content")


def table_ddl(name: str = TABLE, *, if_not_exists: bool = False) -> str:
    """DDL for the keyword index. Same columns and tokenizer as ever."""
    guard = "IF NOT EXISTS " if if_not_exists else ""
    return (
        f"CREATE VIRTUAL TABLE {guard}{name} USING fts5(\n"
        "    filename,\n"
        "    title,\n"
        "    authors,\n"
        "    abstract,\n"
        "    text_content,\n"
        "    content='',\n"
        "    contentless_delete=1,\n"
        "    tokenize='porter unicode61'\n"
        ")"
    )


_CONTENTLESS_DELETE_RE = re.compile(r"contentless_delete\s*=\s*'?1'?", re.I)
_warned = False


def has_contentless_delete(conn: sqlite3.Connection) -> bool:
    """True if ``papers_fts`` was created with ``contentless_delete=1``."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (TABLE,)
    ).fetchone()
    return bool(row and row[0] and _CONTENTLESS_DELETE_RE.search(row[0]))


def _warn_legacy() -> None:
    global _warned
    if not _warned:
        logger.warning(
            "papers_fts is a legacy contentless table: metadata edits are not "
            "reaching keyword search. Run `gantry fts rebuild` to fix."
        )
        _warned = True


def _current_values(conn: sqlite3.Connection, paper_id: int):
    """(filename, title, authors, abstract, text) now, or None if no text row."""
    row = conn.execute(
        "SELECT p.filename, p.title, p.authors, p.abstract, pt.raw_text "
        "FROM papers p JOIN paper_text pt ON pt.paper_id = p.id WHERE p.id = ?",
        (paper_id,),
    ).fetchone()
    if row is None:
        return None
    return tuple(v or "" for v in row)


def _metadata_values(conn: sqlite3.Connection, paper_id: int):
    row = conn.execute(
        "SELECT filename, title, authors, abstract FROM papers WHERE id = ?", (paper_id,)
    ).fetchone()
    return tuple(v or "" for v in row) if row else ("", "", "", "")


def _row_exists(conn: sqlite3.Connection, paper_id: int) -> bool:
    return conn.execute(
        f"SELECT rowid FROM {TABLE} WHERE rowid = ?", (paper_id,)
    ).fetchone() is not None


def _legacy_delete(conn, paper_id, text: str) -> None:
    """The pre-fix delete: current metadata plus the given text."""
    if not _row_exists(conn, paper_id):
        return
    conn.execute(
        f"INSERT INTO {TABLE}({TABLE}, rowid, {', '.join(COLUMNS)}) "
        "VALUES('delete', ?, ?, ?, ?, ?, ?)",
        (paper_id, *_metadata_values(conn, paper_id), text or ""),
    )


def _insert(conn, paper_id) -> None:
    values = _current_values(conn, paper_id)
    if values is None:
        return
    conn.execute(
        f"INSERT INTO {TABLE}(rowid, {', '.join(COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?)",
        (paper_id, *values),
    )


def refresh_row(
    conn: sqlite3.Connection, paper_id: int, *, old_text: str | None = None
) -> None:
    """Make ``paper_id``'s FTS row match its current papers/paper_text values.

    Does not commit. ``old_text`` is the text that was indexed before a text
    rewrite (process/ocr pass it, read before overwriting ``paper_text``);
    metadata writers leave it ``None``.

    Legacy table: a text rewrite keeps the old delete-with-values path. A
    metadata-only refresh is skipped, since the indexed values are unknown
    and a wrong-values delete corrupts the index.
    """
    if has_contentless_delete(conn):
        conn.execute(f"DELETE FROM {TABLE} WHERE rowid = ?", (paper_id,))
        _insert(conn, paper_id)
        return
    _warn_legacy()
    if old_text is None:
        return
    _legacy_delete(conn, paper_id, old_text)
    _insert(conn, paper_id)


def refresh_rows(conn: sqlite3.Connection, paper_ids) -> None:
    """``refresh_row`` for several metadata-only changes. Does not commit."""
    for pid in paper_ids:
        refresh_row(conn, pid)


def delete_row(conn: sqlite3.Connection, paper_id: int) -> None:
    """Remove ``paper_id``'s FTS row (for prune). Does not commit.

    Call it before the ``papers`` row is deleted: the legacy fallback reads
    the current values from it.
    """
    if has_contentless_delete(conn):
        conn.execute(f"DELETE FROM {TABLE} WHERE rowid = ?", (paper_id,))
        return
    _warn_legacy()
    text = conn.execute(
        "SELECT raw_text FROM paper_text WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    _legacy_delete(conn, paper_id, text[0] if text else "")


_POPULATE_SQL = (
    "INSERT INTO {name}(rowid, " + ", ".join(COLUMNS) + ") "
    "SELECT p.id, COALESCE(p.filename, ''), COALESCE(p.title, ''), "
    "COALESCE(p.authors, ''), COALESCE(p.abstract, ''), COALESCE(pt.raw_text, '') "
    "FROM papers p JOIN paper_text pt ON pt.paper_id = p.id"
)


def rebuild(conn: sqlite3.Connection, *, dry_run: bool = False,
            _before_commit=None) -> dict:
    """Rebuild ``papers_fts`` from current papers + paper_text, atomically.

    Builds a ``contentless_delete=1`` table under a temporary name, then drops
    the old table and renames the new one, all in one transaction. Readers on
    other connections (WAL) keep searching the old index until the commit.
    Idempotent; also converts a legacy table.
    """
    start = time.time()
    previous = has_contentless_delete(conn)
    rows = conn.execute(
        "SELECT COUNT(*) FROM papers p JOIN paper_text pt ON pt.paper_id = p.id"
    ).fetchone()[0]
    previous_rows = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    report = {
        "dry_run": dry_run,
        "rows": rows,
        "previous_rows": previous_rows,
        "previous_contentless_delete": previous,
        "contentless_delete": previous if dry_run else True,
    }
    if dry_run:
        report["elapsed_seconds"] = round(time.time() - start, 2)
        return report

    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(f"DROP TABLE IF EXISTS {TEMP_TABLE}")
        conn.execute(table_ddl(TEMP_TABLE))
        conn.execute(_POPULATE_SQL.format(name=TEMP_TABLE))
        conn.execute(f"DROP TABLE {TABLE}")
        conn.execute(f"ALTER TABLE {TEMP_TABLE} RENAME TO {TABLE}")
        if _before_commit is not None:
            _before_commit()
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    report["elapsed_seconds"] = round(time.time() - start, 2)
    return report
