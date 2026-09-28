"""Hand metadata edits: ``gantry meta set | clear | normalize``.

Before these existed, agents set titles/authors/citekeys with raw SQL (16
writes in two months). Each writer picked its own conventions, which is how
the index came to hold ``;``-separated author strings beside JSON lists,
``YYYY-MM-DD HH:MM:SS`` timestamps beside ISO ones, and abstracts orphaned
when a wrong match was cleared without them. Every write here goes through
one normaliser, so the stored shapes stay uniform:

- ``authors`` is always a JSON-encoded list of names;
- timestamps are ``utils.now_iso()`` (ISO 8601, UTC offset);
- ``metadata_source`` is ``manual:<by>`` for hand-set metadata, which
  ``enrich`` never overwrites (see ``queue.manual_condition``).
"""

import json
import re
import sqlite3

from .queue import MANUAL_SOURCE_PREFIX, MISS_SOURCE_PREFIX
from .utils import now_iso, resolve_ids

# Fields `meta set` accepts, in output order.
SET_FIELDS = ("title", "authors", "year", "doi", "abstract", "citekey")

# Fields `meta clear` nulls: everything a provider match writes.
CLEAR_FIELDS = ("title", "authors", "year", "doi", "abstract", "semantic_scholar_id")

# Columns a hand-SQL writer filled with SQLite's datetime('now') format.
TIMESTAMP_COLUMNS = ("metadata_enriched_at", "metadata_verified_at", "updated_at")

_SQLITE_DATETIME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}(?:\.\d+)?)$"
)
_DOI_PREFIX_RE = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", re.I)

LEGACY_MISS_SOURCE = MISS_SOURCE_PREFIX + "legacy"


def normalize_authors(value) -> list[str] | None:
    """Coerce an authors value to a list of names.

    Accepts a list, a JSON-encoded list, or a ``;``-separated string. A plain
    string with no ``;`` is one author: commas are ambiguous ("Last, First")
    so they are never split on. ``None`` stays ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                value = parsed
        if isinstance(value, str):
            value = text.split(";") if ";" in text else ([text] if text else [])
    return [str(name).strip() for name in value if str(name).strip()]


def normalize_timestamp(value: str | None) -> str | None:
    """Rewrite SQLite ``datetime('now')`` output (UTC) as ISO 8601.

    Anything else is returned unchanged, so this is safe to re-run.
    """
    if value is None:
        return None
    m = _SQLITE_DATETIME_RE.match(value)
    if not m:
        return value
    return f"{m.group(1)}T{m.group(2)}+00:00"


def _normalize_doi(doi: str) -> str:
    return _DOI_PREFIX_RE.sub("", doi.strip())


def _decode_authors(stored):
    """Stored authors value, decoded for display in a change report."""
    return normalize_authors(stored) if stored is not None else None


def set_metadata(
    conn: sqlite3.Connection,
    paper_id: int,
    *,
    title: str | None = None,
    authors=None,
    year: int | None = None,
    doi: str | None = None,
    abstract: str | None = None,
    citekey: str | None = None,
    by: str = "cli",
    dry_run: bool = False,
) -> dict:
    """Set metadata fields on one paper by hand.

    Only the fields given are written. The paper is marked
    ``metadata_source='manual:<by>'`` and its suspect flag and verify score
    are cleared, since a human asserted these values. Raises ``LookupError``
    if the paper does not exist, ``ValueError`` if no field was given.
    """
    given = {
        "title": title, "authors": authors, "year": year,
        "doi": doi, "abstract": abstract, "citekey": citekey,
    }
    given = {k: v for k, v in given.items() if v is not None}
    if not given:
        raise ValueError(
            "Nothing to set: pass at least one of --" + ", --".join(SET_FIELDS)
        )

    row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    if row is None:
        raise LookupError(paper_id)

    source = MANUAL_SOURCE_PREFIX + by
    stored: dict = {}
    changes: dict = {}
    for field in SET_FIELDS:
        if field not in given:
            continue
        new = given[field]
        old = row[field]
        if field == "authors":
            names = normalize_authors(new)
            stored[field] = json.dumps(names)
            changes[field] = {"old": _decode_authors(old), "new": names}
            continue
        if field == "doi":
            new = _normalize_doi(new)
        if field == "year":
            new = int(new)
        stored[field] = new
        changes[field] = {"old": old, "new": new}

    result = {
        "id": paper_id,
        "dry_run": dry_run,
        "metadata_source": source,
        "changes": changes,
    }
    if dry_run:
        return result

    now = now_iso()
    assignments = dict(stored)
    if "citekey" in stored:
        assignments["citekey_source"] = source
    assignments.update({
        "metadata_source": source,
        "metadata_enriched_at": now,
        "metadata_suspect": 0,
        "metadata_verify_score": None,
        "metadata_verified_at": None,
        "updated_at": now,
    })
    sets = ", ".join(f"{col} = ?" for col in assignments)
    conn.execute(
        f"UPDATE papers SET {sets} WHERE id = ?",
        [*assignments.values(), paper_id],
    )
    conn.commit()
    return result


def clear_metadata(
    conn: sqlite3.Connection,
    paper_ids: list[int],
    *,
    citekey: bool = False,
    dry_run: bool = False,
) -> dict:
    """Null provider/hand metadata on papers and reset enrich/verify state.

    Clears title, authors, year, doi, abstract and semantic_scholar_id --
    abstract included, which the 09-10 hand clearing missed -- and resets
    ``metadata_source``/``metadata_enriched_at``/suspect/verify columns so the
    next default ``enrich`` re-attempts the paper. ``citekey`` is left alone
    unless asked, because it usually comes from the user's own .bib.
    """
    found, not_found = resolve_ids(conn, paper_ids)
    fields = list(CLEAR_FIELDS) + (["citekey"] if citekey else [])
    result = {
        "dry_run": dry_run,
        "cleared": found,
        "not_found": not_found,
        "fields": fields,
    }
    if dry_run or not found:
        return result

    assignments = {f: None for f in CLEAR_FIELDS}
    if citekey:
        assignments["citekey"] = None
        assignments["citekey_source"] = None
    assignments.update({
        "metadata_source": None,
        "metadata_enriched_at": None,
        "metadata_suspect": 0,
        "metadata_verify_score": None,
        "metadata_verified_at": None,
    })
    sets = ", ".join(f"{col} = ?" for col in assignments)
    placeholders = ",".join("?" * len(found))
    conn.execute(
        f"UPDATE papers SET {sets}, updated_at = ? WHERE id IN ({placeholders})",
        [*assignments.values(), now_iso(), *found],
    )
    conn.commit()
    return result


def normalize_metadata(
    conn: sqlite3.Connection,
    *,
    dry_run: bool = False,
    all_orphans: bool = False,
) -> dict:
    """Repair metadata written outside gantry's code paths. Idempotent.

    1. Non-JSON ``authors`` values become JSON lists.
    2. ``YYYY-MM-DD HH:MM:SS`` timestamps in ``TIMESTAMP_COLUMNS`` become
       ISO 8601 (SQLite's ``datetime('now')`` is UTC, so ``+00:00``).
    3. Orphaned abstracts -- abstract present, title NULL, and flagged
       ``metadata_suspect`` (a wrong match cleared by hand without its
       abstract) -- are nulled. Other abstract-without-title rows are only
       reported, unless ``all_orphans``.
    4. Legacy provider misses (stamped, no source, no title) are tagged
       ``metadata_source='none:legacy'``.
    """
    report: dict = {"dry_run": dry_run}
    content_changed: set[int] = set()

    # 1. authors
    author_fixes: list[tuple[str, int]] = []
    for row in conn.execute(
        "SELECT id, authors FROM papers WHERE authors IS NOT NULL ORDER BY id"
    ):
        raw = row["authors"]
        try:
            ok = isinstance(json.loads(raw), list)
        except (json.JSONDecodeError, TypeError):
            ok = False
        if not ok:
            author_fixes.append((json.dumps(normalize_authors(raw)), row["id"]))
    report["authors_normalized"] = len(author_fixes)
    report["authors_ids"] = [pid for _, pid in author_fixes]

    # 2. timestamps
    ts_fixes: dict[str, list[tuple[str, int]]] = {}
    for col in TIMESTAMP_COLUMNS:
        ts_fixes[col] = [
            (normalize_timestamp(r[col]), r["id"])
            for r in conn.execute(
                f"SELECT id, {col} FROM papers WHERE {col} LIKE '____-__-__ __:__:__%'"
            )
            if normalize_timestamp(r[col]) != r[col]
        ]
    report["timestamps_normalized"] = {col: len(v) for col, v in ts_fixes.items()}

    # 3. orphaned abstracts
    orphan_where = "abstract IS NOT NULL AND abstract != '' AND title IS NULL"
    suspect_orphans = [r[0] for r in conn.execute(
        f"SELECT id FROM papers WHERE {orphan_where} AND metadata_suspect = 1 ORDER BY id"
    )]
    other_orphans = [r[0] for r in conn.execute(
        f"SELECT id FROM papers WHERE {orphan_where} AND metadata_suspect = 0 ORDER BY id"
    )]
    to_clear = sorted(suspect_orphans + other_orphans) if all_orphans else suspect_orphans
    report["orphan_abstracts_cleared"] = len(to_clear)
    report["orphan_abstract_ids"] = to_clear
    report["other_orphan_abstract_ids"] = [] if all_orphans else other_orphans

    # 4. legacy misses
    legacy = [r[0] for r in conn.execute(
        "SELECT id FROM papers WHERE metadata_source IS NULL "
        "AND metadata_enriched_at IS NOT NULL AND (title IS NULL OR title = '') "
        "ORDER BY id"
    )]
    report["legacy_misses_tagged"] = len(legacy)

    if dry_run:
        return report

    # Timestamps first, so the updated_at bump below (already ISO) wins.
    for col, fixes in ts_fixes.items():
        conn.executemany(f"UPDATE papers SET {col} = ? WHERE id = ?", fixes)
    conn.executemany("UPDATE papers SET authors = ? WHERE id = ?", author_fixes)
    content_changed.update(pid for _, pid in author_fixes)
    conn.executemany(
        "UPDATE papers SET abstract = NULL WHERE id = ?", [(i,) for i in to_clear]
    )
    content_changed.update(to_clear)
    conn.executemany(
        "UPDATE papers SET metadata_source = ? WHERE id = ?",
        [(LEGACY_MISS_SOURCE, i) for i in legacy],
    )
    content_changed.update(legacy)
    now = now_iso()
    conn.executemany(
        "UPDATE papers SET updated_at = ? WHERE id = ?",
        [(now, i) for i in sorted(content_changed)],
    )
    conn.commit()
    return report
