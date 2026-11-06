"""Hand-curated paper collections, kept in a sidecar next to index.db.

This module is the single write path for ``collections.db``. It never writes
index.db and never loads sqlite-vec, so ``index_conn`` may be a plain
``mode=ro`` connection: rows from it are read by position because such a
connection has no row factory.

``papers.id`` is not stable (a rename gives a new row and leaves the old one
until a manual ``gantry prune``), so a membership is keyed on file_hash + path
and matched back onto ``papers`` at read time by ``_match``, the one function
resolve, collections_for, refresh, add and remove all share.
"""

import json
import os
import secrets
import sqlite3
import unicodedata
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .utils import resolve_title

SCHEMA_VERSION = 1

_SCHEMA = (
    """CREATE TABLE collections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        name_key TEXT NOT NULL,
        note TEXT,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        removed_at TEXT,
        removed_by TEXT,
        removed_batch_id TEXT
    )""",
    "CREATE UNIQUE INDEX collections_live_name ON collections(name_key)"
    " WHERE removed_at IS NULL",
    """CREATE TABLE batches (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL UNIQUE,
        op TEXT NOT NULL,
        collection_id INTEGER REFERENCES collections(id),
        collection_name TEXT,
        by TEXT NOT NULL,
        why TEXT,
        at TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 0,
        detail TEXT,
        undoes TEXT,
        undone_by TEXT
    )""",
    "CREATE INDEX batches_collection ON batches(collection_id)",
    """CREATE TABLE memberships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_id INTEGER NOT NULL REFERENCES collections(id),
        file_hash TEXT NOT NULL,
        path TEXT NOT NULL,
        filename TEXT,
        title TEXT,
        citekey TEXT,
        added_by TEXT NOT NULL,
        why TEXT,
        added_at TEXT NOT NULL,
        added_batch_id TEXT NOT NULL REFERENCES batches(batch_id),
        removed_at TEXT,
        removed_by TEXT,
        removed_batch_id TEXT REFERENCES batches(batch_id)
    )""",
    "CREATE UNIQUE INDEX memberships_live ON memberships(collection_id, file_hash, path)"
    " WHERE removed_at IS NULL",
    "CREATE INDEX memberships_added_batch ON memberships(added_batch_id)",
    "CREATE INDEX memberships_removed_batch ON memberships(removed_batch_id)",
)


class CollectionError(Exception):
    """A refused operation. ``code`` is stable for callers; the text is for people."""

    code = "error"


class UnknownCollectionError(CollectionError):
    code = "unknown_collection"


class NameConflictError(CollectionError):
    code = "name_conflict"


class UnknownBatchError(CollectionError):
    code = "unknown_batch"


class AlreadyUndoneError(CollectionError):
    code = "already_undone"


class UndoConflictError(CollectionError):
    code = "undo_conflict"


# --- connections ---


def collections_path(index_dir) -> Path:
    return Path(index_dir) / "collections.db"


def _configure(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _check_version(version: int) -> None:
    if version > SCHEMA_VERSION:
        raise CollectionError(
            f"collections.db is schema v{version}; this gantry understands "
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
    """Open collections.db for writing, creating it (mode 0600) if needed."""
    return _connect_path(collections_path(cfg.index_dir))


def connect_readonly(cfg) -> sqlite3.Connection | None:
    """Open collections.db mode=ro, or None if there is nothing to read yet.

    The read path uses this rather than ``connect``, so a read never creates
    the file.
    """
    path = collections_path(cfg.index_dir)
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


# --- matching ---


@dataclass(frozen=True)
class _Paper:
    id: int
    path: str
    filename: str
    file_hash: str
    title: str | None
    layout_title: str | None
    citekey: str | None


@dataclass(frozen=True)
class _Match:
    paper: _Paper | None = None
    candidates: tuple[int, ...] = ()

    @property
    def ambiguous(self) -> bool:
        return bool(self.candidates)


class _Index:
    """One ``papers`` query, keyed for matching, with a cached liveness check."""

    def __init__(self, papers: list[_Paper], papers_dir=None):
        self.papers = papers
        self.by_id = {p.id: p for p in papers}
        self.by_path = {p.path: p for p in papers}
        self.by_hash: dict[str, list[_Paper]] = defaultdict(list)
        for p in papers:
            self.by_hash[p.file_hash].append(p)
        # An unreachable papers_dir says nothing about any one file (Quern
        # calls over SSH): skip the check rather than call everything gone.
        self.papers_dir = Path(papers_dir) if papers_dir and Path(papers_dir).is_dir() else None
        self._live: dict[str, bool] = {}

    def is_live(self, paper: _Paper) -> bool:
        if paper.path not in self._live:
            self._live[paper.path] = (self.papers_dir / paper.path).exists()
        return self._live[paper.path]

    def without_liveness(self) -> "_Index":
        return _Index(self.papers)


def _load_index(index_conn: sqlite3.Connection, papers_dir=None) -> _Index:
    rows = index_conn.execute(
        "SELECT id, path, filename, file_hash, title, title_from_layout, citekey"
        " FROM papers ORDER BY id"
    ).fetchall()
    return _Index([_Paper(*tuple(r)) for r in rows], papers_dir)


def _decide(tier: list[_Paper]) -> _Match:
    if len(tier) == 1:
        return _Match(paper=tier[0])
    return _Match(candidates=tuple(p.id for p in tier))


def _match(index: _Index, file_hash: str, path: str) -> _Match:
    """Find the paper a stored key points at now.

    Tiers: hash and path, then hash only, then path only. With a papers_dir, a
    live candidate in any tier beats a dead one in a better tier, so a stale
    row left by a rename yields to the live row with the same hash. If nothing
    is live the best dead row still resolves: its text and chunks serve reads,
    and an evicted file is not a lost paper.
    """
    same_hash = index.by_hash.get(file_hash, [])
    at_path = index.by_path.get(path)
    tiers = (
        [p for p in same_hash if p.path == path],
        [p for p in same_hash if p.path != path],
        [at_path] if at_path and at_path.file_hash != file_hash else [],
    )
    if index.papers_dir:
        for tier in tiers:
            live = [p for p in tier if index.is_live(p)]
            if live:
                return _decide(live)
    for tier in tiers:
        if tier:
            return _decide(tier)
    return _Match()


# --- helpers ---


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _clean_name(name: str) -> str:
    cleaned = unicodedata.normalize("NFC", name or "").strip()
    if not cleaned:
        raise CollectionError("Collection name cannot be empty.")
    return cleaned


def _name_key(name: str) -> str:
    # In Python, not COLLATE NOCASE, which folds ASCII only.
    return _clean_name(name).casefold()


def _require_by(by) -> None:
    if not by or not str(by).strip():
        raise CollectionError("'by' is required: say who is making this change.")


def _name_holder(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM collections WHERE name_key = ? AND removed_at IS NULL",
        (_name_key(name),),
    ).fetchone()


def _find(conn: sqlite3.Connection, name: str) -> sqlite3.Row:
    row = _name_holder(conn, name)
    if row is None:
        raise UnknownCollectionError(f"Unknown collection: {name}")
    return row


def _live_memberships(conn: sqlite3.Connection, collection_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM memberships WHERE collection_id = ? AND removed_at IS NULL ORDER BY id",
        (collection_id,),
    ).fetchall()


def _open_batch(conn, op, by, collection_id=None, collection_name=None, *, why=None,
                detail=None, undoes=None) -> str:
    batch_id = secrets.token_hex(6)
    conn.execute(
        "INSERT INTO batches (batch_id, op, collection_id, collection_name, by, why, at,"
        " detail, undoes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (batch_id, op, collection_id, collection_name, by, why, _now(),
         json.dumps(detail) if detail else None, undoes),
    )
    return batch_id


def _set_count(conn, batch_id: str, count: int) -> None:
    conn.execute("UPDATE batches SET count = ? WHERE batch_id = ?", (count, batch_id))


def _tombstone(conn, membership_id: int, by: str, batch_id: str) -> None:
    conn.execute(
        "UPDATE memberships SET removed_at = ?, removed_by = ?, removed_batch_id = ?"
        " WHERE id = ?",
        (_now(), by, batch_id, membership_id),
    )


def _collides(conn, membership: sqlite3.Row, paper: _Paper) -> bool:
    """Would moving ``membership`` onto ``paper``'s key clash with a live sibling?"""
    return conn.execute(
        "SELECT 1 FROM memberships WHERE collection_id = ? AND file_hash = ? AND path = ?"
        " AND removed_at IS NULL AND id != ?",
        (membership["collection_id"], paper.file_hash, paper.path, membership["id"]),
    ).fetchone() is not None


def _has_drifted(membership: sqlite3.Row, paper: _Paper) -> bool:
    return (membership["file_hash"], membership["path"]) != (paper.file_hash, paper.path)


def _persist_keys(conn, membership: sqlite3.Row, paper: _Paper) -> None:
    conn.execute(
        "UPDATE memberships SET file_hash = ?, path = ?, filename = ?, title = ?, citekey = ?"
        " WHERE id = ?",
        (paper.file_hash, paper.path, paper.filename, paper.title or paper.layout_title,
         paper.citekey, membership["id"]),
    )


def _collection_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "note": row["note"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
    }


def _provenance(m: sqlite3.Row) -> dict:
    return {
        "added_by": m["added_by"],
        "added_at": m["added_at"],
        "why": m["why"],
        "batch_id": m["added_batch_id"],
    }


def _stored(m: sqlite3.Row) -> dict:
    """What a membership remembers, for a member that no longer resolves."""
    return {
        "file_hash": m["file_hash"],
        "path": m["path"],
        "filename": m["filename"],
        "title": m["title"],
        "citekey": m["citekey"],
        **_provenance(m),
    }


def _resolved(memberships: list[sqlite3.Row], index: _Index) -> dict:
    ids: list[int] = []
    members: list[dict] = []
    missing: list[dict] = []
    ambiguous: list[dict] = []
    seen: set[int] = set()
    for m in memberships:
        found = _match(index, m["file_hash"], m["path"])
        if found.ambiguous:
            ambiguous.append({**_stored(m), "candidates": list(found.candidates)})
        elif found.paper is None:
            missing.append(_stored(m))
        elif found.paper.id not in seen:  # two keys can drift onto one row
            p = found.paper
            seen.add(p.id)
            ids.append(p.id)
            members.append({
                "id": p.id,
                "path": p.path,
                "filename": p.filename,
                "file_hash": p.file_hash,
                "title": p.title,
                "display_title": resolve_title(p.title, p.layout_title, p.filename)[0],
                "citekey": p.citekey,
                **_provenance(m),
            })
    return {"ids": ids, "members": members, "missing": missing, "ambiguous": ambiguous}


# --- reads ---


def list_collections(conn, index_conn=None, papers_dir=None) -> list[dict]:
    """Live collections by name. ``count`` is stored memberships; with
    ``index_conn`` each also carries resolved, missing and ambiguous counts."""
    index = _load_index(index_conn, papers_dir) if index_conn is not None else None
    out = []
    rows = conn.execute(
        "SELECT * FROM collections WHERE removed_at IS NULL ORDER BY name_key"
    ).fetchall()
    for row in rows:
        memberships = _live_memberships(conn, row["id"])
        entry = {**_collection_dict(row), "count": len(memberships)}
        if index is not None:
            r = _resolved(memberships, index)
            entry.update(resolved=len(r["ids"]), missing=len(r["missing"]),
                         ambiguous=len(r["ambiguous"]))
        out.append(entry)
    return out


def resolve(conn, index_conn, name, papers_dir=None) -> dict:
    """Current ids, members, missing and ambiguous for a collection. Never writes."""
    row = _find(conn, name)
    result = _resolved(_live_memberships(conn, row["id"]),
                       _load_index(index_conn, papers_dir))
    return {"collection": _collection_dict(row), **result}


def collections_for(conn, index_conn, paper_id, papers_dir=None) -> list[dict]:
    """Live collections this paper belongs to. Read-only. An ambiguous
    membership counts for no paper until it is settled."""
    index = _load_index(index_conn, papers_dir)
    if paper_id not in index.by_id:
        return []
    rows = conn.execute(
        "SELECT m.*, c.name AS c_name, c.note AS c_note FROM memberships m"
        " JOIN collections c ON c.id = m.collection_id"
        " WHERE m.removed_at IS NULL AND c.removed_at IS NULL ORDER BY c.name_key, m.id"
    ).fetchall()
    out: dict[int, dict] = {}
    for m in rows:
        found = _match(index, m["file_hash"], m["path"])
        if found.paper is not None and found.paper.id == paper_id:
            out.setdefault(m["collection_id"], {
                "id": m["collection_id"],
                "name": m["c_name"],
                "note": m["c_note"],
                **_provenance(m),
            })
    return list(out.values())


def history(conn, name=None, limit=50) -> list[dict]:
    """One row per batch, newest first. A name covers its deleted namesakes too."""
    where, params = "", []
    if name is not None:
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM collections WHERE name_key = ?", (_name_key(name),)
        )]
        if not ids:
            raise UnknownCollectionError(f"Unknown collection: {name}")
        where = f"WHERE collection_id IN ({','.join('?' * len(ids))})"
        params = ids
    rows = conn.execute(
        f"SELECT * FROM batches {where} ORDER BY seq DESC LIMIT ?", [*params, limit]
    ).fetchall()
    return [{
        "batch_id": r["batch_id"],
        "op": r["op"],
        "by": r["by"],
        "why": r["why"],
        "at": r["at"],
        "count": r["count"],
        "undone": r["undone_by"] is not None,
        "collection": r["collection_name"],
        "undone_by": r["undone_by"],
        "undoes": r["undoes"],
    } for r in rows]


# --- writes ---


def create(conn, name, note=None, by=None) -> dict:
    _require_by(by)
    cleaned = _clean_name(name)
    with _tx(conn):
        if _name_holder(conn, cleaned) is not None:
            raise NameConflictError(f"A collection named {cleaned!r} already exists.")
        cid = conn.execute(
            "INSERT INTO collections (name, name_key, note, created_by, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (cleaned, _name_key(cleaned), note, by, _now()),
        ).lastrowid
        batch_id = _open_batch(conn, "create", by, cid, cleaned)
        _set_count(conn, batch_id, 1)
        row = conn.execute("SELECT * FROM collections WHERE id = ?", (cid,)).fetchone()
    return {**_collection_dict(row), "batch_id": batch_id}


def add(conn, index_conn, name, paper_ids, by, why=None, papers_dir=None) -> dict:
    """Add papers. Idempotent: a paper a membership already resolves to is
    reported in already_present, and that membership's key is brought up to date."""
    _require_by(by)
    index = _load_index(index_conn, papers_dir)
    added: list[int] = []
    present: list[int] = []
    not_found: list[int] = []
    with _tx(conn):
        row = _find(conn, name)
        memberships = _live_memberships(conn, row["id"])
        keys = {(m["file_hash"], m["path"]) for m in memberships}
        held: dict[int, sqlite3.Row] = {}
        for m in memberships:
            found = _match(index, m["file_hash"], m["path"])
            if found.paper is not None:
                held.setdefault(found.paper.id, m)
        batch_id = _open_batch(conn, "add", by, row["id"], row["name"], why=why)
        for pid in dict.fromkeys(paper_ids):
            paper = index.by_id.get(pid)
            if paper is None:
                not_found.append(pid)
            elif pid in held:
                present.append(pid)
                m = held[pid]
                if _has_drifted(m, paper) and not _collides(conn, m, paper):
                    _persist_keys(conn, m, paper)
            elif (paper.file_hash, paper.path) in keys:
                present.append(pid)
            else:
                conn.execute(
                    "INSERT INTO memberships (collection_id, file_hash, path, filename, title,"
                    " citekey, added_by, why, added_at, added_batch_id)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (row["id"], paper.file_hash, paper.path, paper.filename,
                     paper.title or paper.layout_title, paper.citekey, by, why, _now(),
                     batch_id),
                )
                keys.add((paper.file_hash, paper.path))
                added.append(pid)
        _set_count(conn, batch_id, len(added))
    return {"batch_id": batch_id, "added": added, "already_present": present,
            "not_found": not_found}


def remove(conn, name, paper_ids, by, index_conn, papers_dir=None) -> dict:
    """Tombstone the memberships these ids resolve to.

    An id may be the row a membership resolves to now, or (with papers_dir) the
    stale row it resolved to before its file moved. An id that is only one
    candidate of an ambiguous membership is reported, not guessed at.
    """
    _require_by(by)
    index = _load_index(index_conn, papers_dir)
    plain = index.without_liveness()
    removed: list[int] = []
    not_found: list[int] = []
    ambiguous: list[int] = []
    with _tx(conn):
        row = _find(conn, name)
        current: dict[int, list[sqlite3.Row]] = defaultdict(list)
        stale: dict[int, list[sqlite3.Row]] = defaultdict(list)
        contested: set[int] = set()
        for m in _live_memberships(conn, row["id"]):
            found = _match(index, m["file_hash"], m["path"])
            contested.update(found.candidates)
            if found.paper is not None:
                current[found.paper.id].append(m)
            exact = _match(plain, m["file_hash"], m["path"])
            if exact.paper is not None and exact.paper != found.paper:
                stale[exact.paper.id].append(m)
        batch_id = _open_batch(conn, "remove", by, row["id"], row["name"])
        count = 0
        for pid in dict.fromkeys(paper_ids):
            hits = current.get(pid) or stale.get(pid)
            if hits:
                for m in hits:
                    _tombstone(conn, m["id"], by, batch_id)
                count += len(hits)
                removed.append(pid)
            elif pid in contested:
                ambiguous.append(pid)
            else:
                not_found.append(pid)
        _set_count(conn, batch_id, count)
    return {"batch_id": batch_id, "removed": removed, "not_found": not_found,
            "ambiguous": ambiguous}


def rename(conn, name, new_name, by) -> dict:
    _require_by(by)
    cleaned = _clean_name(new_name)
    with _tx(conn):
        row = _find(conn, name)
        holder = _name_holder(conn, cleaned)
        if holder is not None and holder["id"] != row["id"]:
            raise NameConflictError(f"A collection named {cleaned!r} already exists.")
        batch_id = _open_batch(conn, "rename", by, row["id"], cleaned,
                               detail={"old": row["name"], "new": cleaned})
        _set_name(conn, row["id"], cleaned)
        _set_count(conn, batch_id, 1)
    return {"batch_id": batch_id, "id": row["id"], "old_name": row["name"], "name": cleaned}


def _set_name(conn, collection_id: int, name: str) -> None:
    conn.execute("UPDATE collections SET name = ?, name_key = ? WHERE id = ?",
                 (name, _name_key(name), collection_id))


def delete(conn, name, by) -> dict:
    """Tombstone a collection. Its memberships stay, so undo brings them back."""
    _require_by(by)
    with _tx(conn):
        row = _find(conn, name)
        count = len(_live_memberships(conn, row["id"]))
        batch_id = _open_batch(conn, "delete", by, row["id"], row["name"])
        conn.execute(
            "UPDATE collections SET removed_at = ?, removed_by = ?, removed_batch_id = ?"
            " WHERE id = ?",
            (_now(), by, batch_id, row["id"]),
        )
        _set_count(conn, batch_id, count)
    return {"batch_id": batch_id, "id": row["id"], "name": row["name"], "count": count}


def _require_live(collection: sqlite3.Row) -> None:
    if collection["removed_at"] is not None:
        raise UndoConflictError(
            f"Collection {collection['name']!r} is deleted; undo its delete first."
        )


def _skip(m: sqlite3.Row, reason: str) -> dict:
    return {"file_hash": m["file_hash"], "path": m["path"], "filename": m["filename"],
            "reason": reason}


def _undo_add(conn, batch, collection, undo_batch, by) -> tuple[int, list[dict]]:
    _require_live(collection)
    reversed_count, skipped = 0, []
    for m in conn.execute("SELECT * FROM memberships WHERE added_batch_id = ? ORDER BY id",
                          (batch["batch_id"],)).fetchall():
        if m["removed_at"] is not None:
            skipped.append(_skip(m, "already_removed"))
        else:
            _tombstone(conn, m["id"], by, undo_batch)
            reversed_count += 1
    return reversed_count, skipped


def _undo_remove(conn, batch, collection, undo_batch, by) -> tuple[int, list[dict]]:
    _require_live(collection)
    reversed_count, skipped = 0, []
    for m in conn.execute("SELECT * FROM memberships WHERE removed_batch_id = ? ORDER BY id",
                          (batch["batch_id"],)).fetchall():
        held = conn.execute(
            "SELECT 1 FROM memberships WHERE collection_id = ? AND file_hash = ? AND path = ?"
            " AND removed_at IS NULL",
            (m["collection_id"], m["file_hash"], m["path"]),
        ).fetchone()
        if held:
            skipped.append(_skip(m, "re_added"))
        else:
            conn.execute(
                "UPDATE memberships SET removed_at = NULL, removed_by = NULL,"
                " removed_batch_id = NULL WHERE id = ?",
                (m["id"],),
            )
            reversed_count += 1
    return reversed_count, skipped


def _undo_rename(conn, batch, collection, undo_batch, by) -> tuple[int, list[dict]]:
    _require_live(collection)
    names = json.loads(batch["detail"])
    if collection["name"] != names["new"]:
        raise UndoConflictError(
            f"Collection has been renamed to {collection['name']!r} since this batch."
        )
    holder = _name_holder(conn, names["old"])
    if holder is not None and holder["id"] != collection["id"]:
        raise UndoConflictError(f"The name {names['old']!r} is now taken by a live collection.")
    _set_name(conn, collection["id"], names["old"])
    return 1, []


def _undo_delete(conn, batch, collection, undo_batch, by) -> tuple[int, list[dict]]:
    if _name_holder(conn, collection["name"]) is not None:
        raise UndoConflictError(
            f"The name {collection['name']!r} is now taken by a live collection."
        )
    conn.execute(
        "UPDATE collections SET removed_at = NULL, removed_by = NULL, removed_batch_id = NULL"
        " WHERE id = ?",
        (collection["id"],),
    )
    return 1, []


def _undo_create(conn, batch, collection, undo_batch, by) -> tuple[int, list[dict]]:
    _require_live(collection)
    held = len(_live_memberships(conn, collection["id"]))
    if held:
        raise UndoConflictError(
            f"Collection {collection['name']!r} has {held} member(s) this batch did not "
            "add. Undo those batches, or delete the collection."
        )
    conn.execute(
        "UPDATE collections SET removed_at = ?, removed_by = ?, removed_batch_id = ?"
        " WHERE id = ?",
        (_now(), by, undo_batch, collection["id"]),
    )
    return 1, []


_UNDO = {
    "add": _undo_add,
    "remove": _undo_remove,
    "rename": _undo_rename,
    "delete": _undo_delete,
    "create": _undo_create,
}


def undo(conn, batch_id, by) -> dict:
    """Reverse what one batch actually did, recording the undo as its own batch.

    What can no longer be reversed row by row lands in ``skipped``; a conflict
    at the collection level refuses the whole undo and changes nothing.
    """
    _require_by(by)
    with _tx(conn):
        batch = conn.execute(
            "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if batch is None:
            raise UnknownBatchError(f"Unknown batch: {batch_id}")
        if batch["undone_by"] is not None:
            raise AlreadyUndoneError(
                f"Batch {batch_id} was already undone by {batch['undone_by']}."
            )
        reverse = _UNDO.get(batch["op"])
        if reverse is None:
            raise CollectionError(f"A {batch['op']} batch cannot be undone.")
        collection = conn.execute(
            "SELECT * FROM collections WHERE id = ?", (batch["collection_id"],)
        ).fetchone()
        undo_batch = _open_batch(conn, "undo", by, collection["id"], collection["name"],
                                 undoes=batch_id)
        reversed_count, skipped = reverse(conn, batch, collection, undo_batch, by)
        _set_count(conn, undo_batch, reversed_count)
        conn.execute("UPDATE batches SET undone_by = ? WHERE batch_id = ?",
                     (undo_batch, batch_id))
    return {"batch_id": undo_batch, "undone": batch_id, "op": batch["op"],
            "reversed": reversed_count, "skipped": skipped}


def refresh(conn, index_conn, papers_dir) -> dict:
    """Persist drifted keys, so a membership survives the old row being pruned.

    Covers deleted collections too, so undoing a delete finds current keys.
    Missing and ambiguous memberships are counted and left exactly as they are.
    """
    index = _load_index(index_conn, papers_dir)
    rows = conn.execute("SELECT * FROM memberships WHERE removed_at IS NULL").fetchall()
    missing = ambiguous = 0
    drifted: list[tuple[sqlite3.Row, _Paper]] = []
    for m in rows:
        found = _match(index, m["file_hash"], m["path"])
        if found.ambiguous:
            ambiguous += 1
        elif found.paper is None:
            missing += 1
        elif _has_drifted(m, found.paper):
            drifted.append((m, found.paper))

    updated = collided = 0
    batch_id = None
    if drifted:
        with _tx(conn):
            for m, paper in drifted:
                if _collides(conn, m, paper):
                    collided += 1
                    continue
                if batch_id is None:
                    batch_id = _open_batch(conn, "refresh", "gantry")
                _persist_keys(conn, m, paper)
                updated += 1
            if batch_id is not None:
                _set_count(conn, batch_id, updated)
    return {"checked": len(rows), "updated": updated, "collided": collided,
            "missing": missing, "ambiguous": ambiguous, "batch_id": batch_id}


def refresh_if_present(index_dir, index_conn, papers_dir) -> dict | None:
    """``refresh`` for callers outside this feature: None if there is no sidecar,
    which it must not create."""
    path = collections_path(index_dir)
    if not path.exists():
        return None
    conn = _connect_path(path)
    try:
        return refresh(conn, index_conn, papers_dir)
    finally:
        conn.close()
