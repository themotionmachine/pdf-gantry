"""A spaced-repetition reading inbox and a reading log, in a sidecar next to index.db.

This module is the single write path for ``reading.db``. It is built the way
collections.py is: it never writes index.db and never loads sqlite-vec, so
``index_conn`` may be a plain ``mode=ro`` connection.

- **Items** are keyed on file_hash + path and matched back onto ``papers`` at
  read time by ``identity.match``, because ``papers.id`` is not stable. An item
  is a whole paper (``page IS NULL``) or a passage ``{page, quote, note}``,
  with page 1-based like chunk ``page_start``. A paper holds any number of
  passage items but at most one live whole-paper item, where live means
  neither tombstoned nor retired.
- **Every write** to items is a batch with ``by`` and ``why``; removal is a
  tombstone; ``undo`` reverses one batch. A review keeps the item's prior
  scheduling state in its ``reviews`` row so its undo can put it back.
- **Opens** are an append-only reading log, written per event and not batched.
- **Reads never write.** Times are UTC ISO strings, and everything that reads
  the clock takes ``now=`` (a datetime, naive meaning UTC, or an ISO string).
"""

import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .identity import Index, Match, Paper, load_index, load_index_for, match
from .utils import resolve_title

SCHEMA_VERSION = 1

VERDICTS = ("key", "good", "doubt", "later", "done")   # !!  !  ?  →  x
VERDICT_MARKS = {"key": "!!", "good": "!", "doubt": "?", "later": "→", "done": "x"}
SOURCES = ("manual", "agent", "vault")
STATUSES = ("active", "due", "scheduled", "done", "all")

# The scheduler. Intervals are in days.
FIRST_INTERVAL_DAYS = 1.0   # what the first review of good, later or doubt gives
KEY_MULTIPLIER = 3.0
KEY_MIN_DAYS = 7.0          # a key paper never comes back sooner than a week
GOOD_MULTIPLIER = 2.0
LATER_MULTIPLIER = 1.5      # "not now": still grows, no penalty
DOUBT_FACTOR = 0.5          # doubt drops back to half the interval, clamped to 1-2 days
DOUBT_MIN_DAYS = 1.0
DOUBT_MAX_DAYS = 2.0

_SCHEMA = (
    """CREATE TABLE batches (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL UNIQUE,
        op TEXT NOT NULL,
        item_id INTEGER,
        verdict TEXT,
        by TEXT NOT NULL,
        why TEXT,
        at TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 0,
        detail TEXT,
        undoes TEXT,
        undone_by TEXT
    )""",
    """CREATE TABLE items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_hash TEXT NOT NULL,
        path TEXT NOT NULL,
        filename TEXT,
        title TEXT,
        citekey TEXT,
        page INTEGER CHECK (page IS NULL OR page >= 1),
        quote TEXT,
        note TEXT,
        source TEXT NOT NULL CHECK (source IN ('manual', 'agent', 'vault')),
        added_by TEXT NOT NULL,
        why TEXT,
        added_at TEXT NOT NULL,
        added_batch_id TEXT NOT NULL REFERENCES batches(batch_id),
        due_at TEXT,
        interval_days REAL,
        reviews INTEGER NOT NULL DEFAULT 0,
        last_verdict TEXT,
        last_reviewed_at TEXT,
        retired_at TEXT,
        removed_at TEXT,
        removed_by TEXT,
        removed_batch_id TEXT REFERENCES batches(batch_id)
    )""",
    "CREATE UNIQUE INDEX items_live_paper ON items(file_hash, path)"
    " WHERE page IS NULL AND removed_at IS NULL AND retired_at IS NULL",
    "CREATE INDEX items_hash ON items(file_hash)",
    "CREATE INDEX items_path ON items(path)",
    "CREATE INDEX items_due ON items(due_at) WHERE removed_at IS NULL AND retired_at IS NULL",
    "CREATE INDEX items_added_batch ON items(added_batch_id)",
    "CREATE INDEX items_removed_batch ON items(removed_batch_id)",
    """CREATE TABLE reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id INTEGER NOT NULL REFERENCES items(id),
        batch_id TEXT NOT NULL UNIQUE REFERENCES batches(batch_id),
        verdict TEXT NOT NULL,
        by TEXT NOT NULL,
        at TEXT NOT NULL,
        before_due_at TEXT,
        before_interval_days REAL,
        before_reviews INTEGER NOT NULL,
        before_last_verdict TEXT,
        before_last_reviewed_at TEXT,
        before_retired_at TEXT,
        due_at TEXT,
        interval_days REAL,
        undone_batch_id TEXT REFERENCES batches(batch_id)
    )""",
    "CREATE INDEX reviews_item ON reviews(item_id)",
    """CREATE TABLE opens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_hash TEXT NOT NULL,
        path TEXT NOT NULL,
        page INTEGER,
        seconds REAL,
        by TEXT NOT NULL,
        at TEXT NOT NULL
    )""",
    "CREATE INDEX opens_hash_at ON opens(file_hash, at)",
    "CREATE INDEX opens_path ON opens(path)",
)


class ReadingError(Exception):
    """A refused operation. ``code`` is stable for callers; the text is for people."""

    code = "error"


class InvalidArgumentError(ReadingError):
    code = "invalid_argument"


class UnknownItemError(ReadingError):
    code = "unknown_item"


class UnknownPaperError(ReadingError):
    code = "not_found"


class ItemRetiredError(ReadingError):
    code = "item_retired"


class UnknownBatchError(ReadingError):
    code = "unknown_batch"


class AlreadyUndoneError(ReadingError):
    code = "already_undone"


class UndoConflictError(ReadingError):
    code = "undo_conflict"


# --- connections ---


def reading_path(index_dir) -> Path:
    return Path(index_dir) / "reading.db"


def _configure(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _check_version(version: int) -> None:
    if version > SCHEMA_VERSION:
        raise ReadingError(
            f"reading.db is schema v{version}; this gantry understands "
            f"v{SCHEMA_VERSION}. Upgrade gantry."
        )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    _check_version(version)
    if version == SCHEMA_VERSION:
        return
    with _tx(conn):
        # Another process may have migrated while this one waited for the lock.
        if conn.execute("PRAGMA user_version").fetchone()[0] < 1:
            for statement in _SCHEMA:
                conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _connect_path(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        os.fchmod(fd, 0o600)
        os.close(fd)
    conn = _configure(sqlite3.connect(path, isolation_level=None))
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        _ensure_schema(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def connect(cfg) -> sqlite3.Connection:
    """Open reading.db for writing, creating it (mode 0600) if needed."""
    return _connect_path(reading_path(cfg.index_dir))


def connect_readonly(cfg) -> sqlite3.Connection | None:
    """Open reading.db mode=ro, or None if there is nothing to read yet.

    The read path uses this rather than ``connect``, so a read never creates
    the file.
    """
    path = reading_path(cfg.index_dir)
    if not path.exists():
        return None
    conn = _configure(sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None,
    ))
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        _check_version(version)
    except BaseException:
        conn.close()
        raise
    if version == 0:  # created but not yet migrated: no tables to read
        conn.close()
        return None
    return conn


@contextmanager
def _tx(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


# --- time ---


def _as_datetime(now) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if isinstance(now, str):
        try:
            now = datetime.fromisoformat(now.replace("Z", "+00:00"))
        except ValueError as e:
            raise InvalidArgumentError(f"Not an ISO time: {now!r}") from e
    if not isinstance(now, datetime):
        raise InvalidArgumentError(f"now must be a datetime or ISO string, not {now!r}")
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds")


def _stamp(now) -> str:
    """``now`` as the one string format every stored time uses, so they sort."""
    return _iso(_as_datetime(now))


# --- scheduler ---


def _check_verdict(verdict) -> None:
    if verdict not in VERDICTS:
        raise InvalidArgumentError(
            f"Unknown verdict {verdict!r}; use one of {', '.join(VERDICTS)}."
        )


def next_interval(prev_interval_days, verdict, reviews) -> float | None:
    """Days until the item is due again after ``verdict``; None retires it.

    ``reviews`` counts the reviews before this one. A first review (no prior
    interval) starts from FIRST_INTERVAL_DAYS; key never goes below
    KEY_MIN_DAYS; doubt resets to DOUBT_MIN_DAYS..DOUBT_MAX_DAYS; later grows
    the interval more slowly than good but does not shrink it.
    """
    _check_verdict(verdict)
    if verdict == "done":
        return None
    first = not reviews or not prev_interval_days
    prev = FIRST_INTERVAL_DAYS if first else float(prev_interval_days)
    if verdict == "key":
        return max(prev * KEY_MULTIPLIER, KEY_MIN_DAYS)
    if first:
        return FIRST_INTERVAL_DAYS
    if verdict == "good":
        return prev * GOOD_MULTIPLIER
    if verdict == "later":
        return prev * LATER_MULTIPLIER
    return min(max(prev * DOUBT_FACTOR, DOUBT_MIN_DAYS), DOUBT_MAX_DAYS)


# --- helpers ---


def _require_by(by) -> None:
    if not by or not str(by).strip():
        raise ReadingError("'by' is required: say who is making this change.")


def _clean_passage(passage) -> tuple[int, str | None, str | None] | None:
    if passage is None:
        return None
    if not isinstance(passage, dict):
        raise InvalidArgumentError("passage must be {page, quote, note}.")
    page = passage.get("page")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise InvalidArgumentError("A passage needs a 1-based integer page.")
    quote, note = passage.get("quote"), passage.get("note")
    for name, value in (("quote", quote), ("note", note)):
        if value is not None and not isinstance(value, str):
            raise InvalidArgumentError(f"A passage {name} must be text.")
    return page, quote or None, note or None


def _passage(row) -> dict | None:
    if row["page"] is None:
        return None
    return {"page": row["page"], "quote": row["quote"], "note": row["note"]}


def _open_batch(conn, op, by, at, *, why=None, item_id=None, verdict=None,
                detail=None, undoes=None) -> str:
    batch_id = secrets.token_hex(6)
    conn.execute(
        "INSERT INTO batches (batch_id, op, item_id, verdict, by, why, at, detail, undoes)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (batch_id, op, item_id, verdict, by, why, at,
         json.dumps(detail) if detail else None, undoes),
    )
    return batch_id


def _set_count(conn, batch_id: str, count: int) -> None:
    conn.execute("UPDATE batches SET count = ? WHERE batch_id = ?", (count, batch_id))


def _status(row, now_iso: str) -> str:
    if row["removed_at"] is not None:
        return "removed"
    if row["retired_at"] is not None:
        return "done"
    return "due" if row["due_at"] is not None and row["due_at"] <= now_iso else "scheduled"


def _item_dict(row, now_iso: str, found: Match | None = None) -> dict:
    paper = found.paper if found is not None else None
    if found is None:
        resolution = None
    elif found.ambiguous:
        resolution = "ambiguous"
    else:
        resolution = "missing" if paper is None else "resolved"
    if paper is not None:
        title = resolve_title(paper.title, paper.layout_title, paper.filename)[0]
    else:
        title = resolve_title(row["title"], None, row["filename"] or row["path"])[0]
    return {
        "item_id": row["id"],
        "paper_id": paper.id if paper is not None else None,
        "resolution": resolution,
        "candidates": list(found.candidates) if found is not None else [],
        "file_hash": paper.file_hash if paper is not None else row["file_hash"],
        "path": paper.path if paper is not None else row["path"],
        "filename": paper.filename if paper is not None else row["filename"],
        "title": title,
        "citekey": paper.citekey if paper is not None else row["citekey"],
        "source": row["source"],
        "by": row["added_by"],
        "why": row["why"],
        "added_at": row["added_at"],
        "batch_id": row["added_batch_id"],
        "passage": _passage(row),
        "status": _status(row, now_iso),
        "due_at": row["due_at"],
        "interval_days": row["interval_days"],
        "reviews": row["reviews"],
        "last_verdict": row["last_verdict"],
        "last_reviewed_at": row["last_reviewed_at"],
    }


def _has_drifted(row, paper: Paper) -> bool:
    return (row["file_hash"], row["path"]) != (paper.file_hash, paper.path)


def _collides(conn, row, paper: Paper) -> bool:
    """Would moving whole-paper ``row`` onto ``paper``'s key clash with a live one?"""
    if row["page"] is not None or row["retired_at"] is not None:
        return False
    return conn.execute(
        "SELECT 1 FROM items WHERE file_hash = ? AND path = ? AND page IS NULL"
        " AND removed_at IS NULL AND retired_at IS NULL AND id != ?",
        (paper.file_hash, paper.path, row["id"]),
    ).fetchone() is not None


def _persist_keys(conn, row, paper: Paper) -> None:
    conn.execute(
        "UPDATE items SET file_hash = ?, path = ?, filename = ?, title = ?, citekey = ?"
        " WHERE id = ?",
        (paper.file_hash, paper.path, paper.filename, paper.title or paper.layout_title,
         paper.citekey, row["id"]),
    )


_LIVE = "removed_at IS NULL"
_ACTIVE = "removed_at IS NULL AND retired_at IS NULL"


# --- writes ---


def enqueue(conn, index_conn, paper_ids, by, why=None, source="manual", passage=None,
            papers_dir=None, now=None) -> dict:
    """Put papers (or one passage of each) in the inbox, due at ``now``.

    Idempotent: a paper whose live whole-paper item (or an identical live
    passage) already resolves to it is reported in ``already_present``, and
    that item's key is brought up to date. ``items`` lists the item id for
    every paper that has one now.
    """
    _require_by(by)
    if source not in SOURCES:
        raise InvalidArgumentError(f"Unknown source {source!r}; use one of {', '.join(SOURCES)}.")
    cleaned = _clean_passage(passage)
    at = _stamp(now)
    index = load_index(index_conn, papers_dir)
    added: list[int] = []
    present: list[int] = []
    not_found: list[int] = []
    out_items: list[dict] = []
    with _tx(conn):
        if cleaned is None:
            rows = conn.execute(
                f"SELECT * FROM items WHERE {_ACTIVE} AND page IS NULL ORDER BY id"
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM items WHERE {_ACTIVE} AND page = ?"
                " AND quote IS ? AND note IS ? ORDER BY id",
                cleaned,
            ).fetchall()
        keys = {(r["file_hash"], r["path"]): r for r in rows}
        held: dict[int, sqlite3.Row] = {}
        for r in rows:
            found = match(index, r["file_hash"], r["path"])
            if found.paper is not None:
                held.setdefault(found.paper.id, r)
        batch_id = _open_batch(conn, "enqueue", by, at, why=why)
        page, quote, note = cleaned or (None, None, None)
        for pid in dict.fromkeys(paper_ids):
            paper = index.by_id.get(pid)
            if paper is None:
                not_found.append(pid)
                continue
            existing = held.get(pid) or keys.get((paper.file_hash, paper.path))
            if existing is not None:
                present.append(pid)
                if (_has_drifted(existing, paper) and not _collides(conn, existing, paper)
                        and (paper.file_hash, paper.path) not in keys):
                    _persist_keys(conn, existing, paper)
                out_items.append({"paper_id": pid, "item_id": existing["id"],
                                  "status": "already_present"})
                continue
            item_id = conn.execute(
                "INSERT INTO items (file_hash, path, filename, title, citekey, page, quote,"
                " note, source, added_by, why, added_at, added_batch_id, due_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (paper.file_hash, paper.path, paper.filename,
                 paper.title or paper.layout_title, paper.citekey, page, quote, note,
                 source, by, why, at, batch_id, at),
            ).lastrowid
            keys[(paper.file_hash, paper.path)] = conn.execute(
                "SELECT * FROM items WHERE id = ?", (item_id,)
            ).fetchone()
            added.append(pid)
            out_items.append({"paper_id": pid, "item_id": item_id, "status": "added"})
        _set_count(conn, batch_id, len(added))
    return {"batch_id": batch_id, "added": added, "already_present": present,
            "not_found": not_found, "items": out_items}


def _get_item(conn, item_id) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None or row["removed_at"] is not None:
        raise UnknownItemError(f"Unknown inbox item: {item_id}")
    return row


def review(conn, item_id, verdict, by, now=None, *, index_conn=None, papers_dir=None) -> dict:
    """Record a verdict and reschedule. Returns the item with its new due_at
    and interval_days (``paper_id`` is resolved only when given ``index_conn``)."""
    _require_by(by)
    _check_verdict(verdict)
    at = _stamp(now)
    with _tx(conn):
        row = _get_item(conn, item_id)
        if row["retired_at"] is not None:
            raise ItemRetiredError(
                f"Inbox item {item_id} is done; undo that review or enqueue the paper again."
            )
        interval = next_interval(row["interval_days"], verdict, row["reviews"])
        due_at = None if interval is None else _iso(_as_datetime(at) + timedelta(days=interval))
        retired_at = at if interval is None else None
        batch_id = _open_batch(conn, "review", by, at, item_id=row["id"], verdict=verdict)
        conn.execute(
            "INSERT INTO reviews (item_id, batch_id, verdict, by, at, before_due_at,"
            " before_interval_days, before_reviews, before_last_verdict,"
            " before_last_reviewed_at, before_retired_at, due_at, interval_days)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row["id"], batch_id, verdict, by, at, row["due_at"], row["interval_days"],
             row["reviews"], row["last_verdict"], row["last_reviewed_at"],
             row["retired_at"], due_at, interval),
        )
        conn.execute(
            "UPDATE items SET due_at = ?, interval_days = ?, reviews = reviews + 1,"
            " last_verdict = ?, last_reviewed_at = ?, retired_at = ? WHERE id = ?",
            (due_at, interval, verdict, at, retired_at, row["id"]),
        )
        _set_count(conn, batch_id, 1)
        row = conn.execute("SELECT * FROM items WHERE id = ?", (row["id"],)).fetchone()
    found = None
    if index_conn is not None:
        idx = load_index_for(index_conn, [(row["file_hash"], row["path"])], papers_dir)
        found = match(idx, row["file_hash"], row["path"])
    return {**_item_dict(row, at, found), "batch_id": batch_id}


def record_open(conn, index_conn, paper_id, page, seconds, by, now=None) -> dict:
    """Append one open to the reading log. Not a batch: the log is never undone."""
    _require_by(by)
    if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
        raise InvalidArgumentError("page must be a 1-based integer.")
    if seconds is not None:
        try:
            seconds = float(seconds)
        except (TypeError, ValueError) as e:
            raise InvalidArgumentError("seconds must be a number.") from e
        if seconds < 0:
            raise InvalidArgumentError("seconds cannot be negative.")
    row = index_conn.execute(
        "SELECT file_hash, path FROM papers WHERE id = ?", (paper_id,)
    ).fetchone()
    if row is None:
        raise UnknownPaperError(f"No paper with ID {paper_id}.")
    file_hash, path = row[0], row[1]
    at = _stamp(now)
    conn.execute(
        "INSERT INTO opens (file_hash, path, page, seconds, by, at) VALUES (?, ?, ?, ?, ?, ?)",
        (file_hash, path, page, seconds, by, at),
    )
    return {"paper_id": paper_id, "file_hash": file_hash, "path": path, "page": page,
            "seconds": seconds, "by": by, "at": at}


# --- reads ---


def _resolve_rows(rows, index: Index, now_iso: str) -> list[dict]:
    return [_item_dict(r, now_iso, match(index, r["file_hash"], r["path"])) for r in rows]


def due(conn, index_conn, now=None, limit=None, papers_dir=None) -> list[dict]:
    """Items due at ``now``, oldest due first. Never writes."""
    now_iso = _stamp(now)
    sql = (f"SELECT * FROM items WHERE {_ACTIVE} AND due_at <= ?"
           " ORDER BY due_at, id")
    params: list = [now_iso]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []
    return _resolve_rows(rows, load_index(index_conn, papers_dir), now_iso)


def items(conn, index_conn, status="active", papers_dir=None, now=None) -> list[dict]:
    """Items by status: active (due or scheduled), due, scheduled, done or all
    (every item not tombstoned). Oldest first. Never writes."""
    if status not in STATUSES:
        raise InvalidArgumentError(
            f"Unknown status {status!r}; use one of {', '.join(STATUSES)}."
        )
    now_iso = _stamp(now)
    where = {"active": _ACTIVE, "due": f"{_ACTIVE} AND due_at <= ?",
             "scheduled": f"{_ACTIVE} AND (due_at IS NULL OR due_at > ?)",
             "done": f"{_LIVE} AND retired_at IS NOT NULL", "all": _LIVE}[status]
    params = [now_iso] if "?" in where else []
    rows = conn.execute(f"SELECT * FROM items WHERE {where} ORDER BY id", params).fetchall()
    if not rows:
        return []
    return _resolve_rows(rows, load_index(index_conn, papers_dir), now_iso)


def _in(values) -> str:
    return ",".join("?" * len(values))


def state_for(conn, index_conn, paper_ids, papers_dir=None, now=None) -> dict[int, dict]:
    """Inbox and reading-log state for each of ``paper_ids`` that has any.

    Built for the search hot path: a fixed handful of queries, whatever the
    number of ids, and only the ``papers`` rows those keys can match. Each
    value is ``{item_id, status, due_at, interval_days, last_verdict,
    last_opened, opens, passages}``. ``status`` is due, scheduled, done (only
    a retired item) or None (opened but never enqueued). The whole-paper item
    wins over passages; ``passages`` counts the active passage items.
    """
    if conn is None or not paper_ids:
        return {}
    now_iso = _stamp(now)
    wanted = list(dict.fromkeys(int(i) for i in paper_ids))
    seeds = index_conn.execute(
        f"SELECT file_hash, path FROM papers WHERE id IN ({_in(wanted)})", wanted
    ).fetchall()
    if not seeds:
        return {}
    hashes = sorted({r[0] for r in seeds})
    paths = sorted({r[1] for r in seeds})
    near = f"(file_hash IN ({_in(hashes)}) OR path IN ({_in(paths)}))"
    rows = conn.execute(
        f"SELECT id, file_hash, path, page, due_at, interval_days, last_verdict,"
        f" retired_at, last_reviewed_at FROM items WHERE {_LIVE} AND {near}",
        [*hashes, *paths],
    ).fetchall()
    opens = conn.execute(
        f"SELECT file_hash, path, COUNT(*) AS n, MAX(at) AS last FROM opens WHERE {near}"
        " GROUP BY file_hash, path",
        [*hashes, *paths],
    ).fetchall()
    keys = {(r["file_hash"], r["path"]) for r in rows} | {
        (o["file_hash"], o["path"]) for o in opens}
    index = load_index_for(index_conn, keys, papers_dir)
    resolved: dict[tuple[str, str], int | None] = {}
    for key in keys:
        found = match(index, *key)
        resolved[key] = found.paper.id if found.paper is not None else None

    wanted_set = set(wanted)
    out: dict[int, dict] = {}

    def entry(pid: int) -> dict:
        return out.setdefault(pid, {
            "item_id": None, "status": None, "due_at": None, "interval_days": None,
            "last_verdict": None, "last_opened": None, "opens": 0, "passages": 0,
        })

    def better(r, than) -> bool:
        # Active beats retired; then whole paper beats passage, then soonest due.
        # Between retired items the most recently retired wins.
        if (r["retired_at"] is None) != (than["retired_at"] is None):
            return r["retired_at"] is None
        if r["retired_at"] is not None:
            return (r["retired_at"], r["id"]) > (than["retired_at"], than["id"])
        return ((r["page"] is not None, r["due_at"] or "", r["id"])
                < (than["page"] is not None, than["due_at"] or "", than["id"]))

    best: dict[int, sqlite3.Row] = {}
    for r in rows:
        pid = resolved[(r["file_hash"], r["path"])]
        if pid not in wanted_set:
            continue
        e = entry(pid)
        if r["page"] is not None and r["retired_at"] is None:
            e["passages"] += 1
        if pid not in best or better(r, best[pid]):
            best[pid] = r
    for pid, r in best.items():
        out[pid].update(
            item_id=r["id"],
            status="done" if r["retired_at"] is not None
            else ("due" if r["due_at"] is not None and r["due_at"] <= now_iso else "scheduled"),
            due_at=r["due_at"], interval_days=r["interval_days"],
            last_verdict=r["last_verdict"],
        )
    for o in opens:
        pid = resolved[(o["file_hash"], o["path"])]
        if pid not in wanted_set:
            continue
        e = entry(pid)
        e["opens"] += o["n"]
        if e["last_opened"] is None or o["last"] > e["last_opened"]:
            e["last_opened"] = o["last"]
    return out


def history(conn, limit=50) -> list[dict]:
    """One row per batch, newest first."""
    rows = conn.execute("SELECT * FROM batches ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
    return [{
        "batch_id": r["batch_id"],
        "op": r["op"],
        "by": r["by"],
        "why": r["why"],
        "at": r["at"],
        "count": r["count"],
        "item_id": r["item_id"],
        "verdict": r["verdict"],
        "undone": r["undone_by"] is not None,
        "undone_by": r["undone_by"],
        "undoes": r["undoes"],
    } for r in rows]


# --- undo ---


def _undo_enqueue(conn, batch, undo_batch, by, at) -> tuple[int, list[dict]]:
    reversed_count, skipped = 0, []
    for r in conn.execute("SELECT * FROM items WHERE added_batch_id = ? ORDER BY id",
                          (batch["batch_id"],)).fetchall():
        if r["removed_at"] is not None:
            skipped.append({"item_id": r["id"], "file_hash": r["file_hash"],
                            "path": r["path"], "reason": "already_removed"})
            continue
        conn.execute(
            "UPDATE items SET removed_at = ?, removed_by = ?, removed_batch_id = ? WHERE id = ?",
            (at, by, undo_batch, r["id"]),
        )
        reversed_count += 1
    return reversed_count, skipped


def _undo_review(conn, batch, undo_batch, by, at) -> tuple[int, list[dict]]:
    rev = conn.execute("SELECT * FROM reviews WHERE batch_id = ?",
                       (batch["batch_id"],)).fetchone()
    item = conn.execute("SELECT * FROM items WHERE id = ?", (rev["item_id"],)).fetchone()
    if item["removed_at"] is not None:
        raise UndoConflictError(
            f"Inbox item {item['id']} was removed (batch {item['removed_batch_id']}); "
            "undo that first."
        )
    latest = conn.execute(
        "SELECT batch_id FROM reviews WHERE item_id = ? AND undone_batch_id IS NULL"
        " ORDER BY id DESC LIMIT 1",
        (item["id"],),
    ).fetchone()
    if latest["batch_id"] != rev["batch_id"]:
        raise UndoConflictError(
            f"Inbox item {item['id']} was reviewed again since (batch {latest['batch_id']}); "
            "undo that first."
        )
    if rev["before_retired_at"] is None and item["retired_at"] is not None \
            and item["page"] is None and conn.execute(
                f"SELECT 1 FROM items WHERE {_ACTIVE} AND page IS NULL AND file_hash = ?"
                " AND path = ? AND id != ?",
                (item["file_hash"], item["path"], item["id"]),
            ).fetchone():
        raise UndoConflictError(
            f"The paper behind inbox item {item['id']} was enqueued again since; "
            "undo that enqueue first."
        )
    conn.execute(
        "UPDATE items SET due_at = ?, interval_days = ?, reviews = ?, last_verdict = ?,"
        " last_reviewed_at = ?, retired_at = ? WHERE id = ?",
        (rev["before_due_at"], rev["before_interval_days"], rev["before_reviews"],
         rev["before_last_verdict"], rev["before_last_reviewed_at"],
         rev["before_retired_at"], item["id"]),
    )
    conn.execute("UPDATE reviews SET undone_batch_id = ? WHERE id = ?", (undo_batch, rev["id"]))
    return 1, []


_UNDO = {"enqueue": _undo_enqueue, "review": _undo_review}


def undo(conn, batch_id, by) -> dict:
    """Reverse what one batch actually did, recording the undo as its own batch.

    Rows an enqueue undo can no longer reverse land in ``skipped``. A review
    undo puts back the item's scheduling state from before that review (a
    ``done`` comes back un-retired); it is refused, changing nothing, if the
    item has been reviewed again, removed, or replaced by a fresh enqueue.
    """
    _require_by(by)
    at = _stamp(None)
    with _tx(conn):
        batch = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        if batch is None:
            raise UnknownBatchError(f"Unknown batch: {batch_id}")
        if batch["undone_by"] is not None:
            raise AlreadyUndoneError(
                f"Batch {batch_id} was already undone by {batch['undone_by']}."
            )
        reverse = _UNDO.get(batch["op"])
        if reverse is None:
            raise ReadingError(f"A {batch['op']} batch cannot be undone.")
        undo_batch = _open_batch(conn, "undo", by, at, item_id=batch["item_id"],
                                 undoes=batch_id)
        reversed_count, skipped = reverse(conn, batch, undo_batch, by, at)
        _set_count(conn, undo_batch, reversed_count)
        conn.execute("UPDATE batches SET undone_by = ? WHERE batch_id = ?",
                     (undo_batch, batch_id))
    return {"batch_id": undo_batch, "undone": batch_id, "op": batch["op"],
            "reversed": reversed_count, "skipped": skipped}


# --- refresh ---


def refresh(conn, index_conn, papers_dir) -> dict:
    """Persist drifted item keys, so an item survives the old row being pruned.

    Covers retired items too, so a later undo finds current keys. Missing and
    ambiguous items are counted and left exactly as they are. The opens log is
    append-only and is matched at read time instead.
    """
    index = load_index(index_conn, papers_dir)
    rows = conn.execute(f"SELECT * FROM items WHERE {_LIVE}").fetchall()
    missing = ambiguous = 0
    drifted: list[tuple[sqlite3.Row, Paper]] = []
    for r in rows:
        found = match(index, r["file_hash"], r["path"])
        if found.ambiguous:
            ambiguous += 1
        elif found.paper is None:
            missing += 1
        elif _has_drifted(r, found.paper):
            drifted.append((r, found.paper))

    updated = collided = 0
    batch_id = None
    if drifted:
        at = _stamp(None)
        with _tx(conn):
            for r, paper in drifted:
                if _collides(conn, r, paper):
                    collided += 1
                    continue
                if batch_id is None:
                    batch_id = _open_batch(conn, "refresh", "gantry", at)
                _persist_keys(conn, r, paper)
                updated += 1
            if batch_id is not None:
                _set_count(conn, batch_id, updated)
    return {"checked": len(rows), "updated": updated, "collided": collided,
            "missing": missing, "ambiguous": ambiguous, "batch_id": batch_id}


def refresh_if_present(index_dir, index_conn, papers_dir) -> dict | None:
    """``refresh`` for callers outside this feature: None if there is no sidecar,
    which it must not create."""
    path = reading_path(index_dir)
    if not path.exists():
        return None
    conn = _connect_path(path)
    try:
        return refresh(conn, index_conn, papers_dir)
    finally:
        conn.close()
