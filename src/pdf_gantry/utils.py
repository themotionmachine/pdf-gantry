"""Shared utilities: hashing, formatting, helpers."""

import hashlib
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m"


def format_count(n: int) -> str:
    """Format integer with comma separators."""
    return f"{n:,}"


def format_pct(part: int, total: int) -> str:
    """Format as percentage string."""
    if total == 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def parse_ids(s: str) -> list[int]:
    """Parse a comma-separated string of integer IDs.

    Strips whitespace from each segment and skips blank segments (handles
    trailing or leading commas gracefully).  Raises ``ValueError`` if any
    non-blank segment cannot be converted to an integer.

    Examples::

        parse_ids("1,2,3")   -> [1, 2, 3]
        parse_ids("1,2,3,")  -> [1, 2, 3]   # trailing comma tolerated
        parse_ids(" 1 , 2 ") -> [1, 2]       # whitespace stripped
        parse_ids("1,foo")   # raises ValueError
    """
    parts = [x.strip() for x in s.split(",")]
    return [int(p) for p in parts if p]


def missing_ids(requested: list[int], found: Iterable[int]) -> list[int]:
    """IDs in ``requested`` that are absent from ``found``.

    Preserves the order of first appearance in ``requested`` and drops
    duplicates. ``found`` may be any iterable of IDs actually resolved by a
    lookup (a set, list, or dict of DB rows).

    This exists to close a composition gap: when a caller pipes an ID set
    from one command (e.g. ``search --ids-only``) into another (e.g.
    ``info --ids``), some IDs may no longer resolve — a paper was pruned,
    a digit was mistyped, the ID set is from a stale snapshot. Without this,
    the callee just returns fewer results than requested and says nothing;
    the caller has no way to know a drop happened without diffing the sets
    itself. ``missing_ids`` makes that diff a shared, tested primitive
    instead of an inline afterthought (or an omission) in each command.

    Examples::

        missing_ids([1, 2, 3], {1, 3})  -> [2]
        missing_ids([5, 2, 9], {2})     -> [5, 9]   # request order preserved
        missing_ids([4, 4], set())      -> [4]      # duplicates collapsed
    """
    found_set = set(found)
    seen: set[int] = set()
    out = []
    for i in requested:
        if i not in found_set and i not in seen:
            out.append(i)
            seen.add(i)
    return out


def resolve_ids(
    conn: sqlite3.Connection, requested: list[int]
) -> tuple[list[int], list[int]]:
    """Split ``requested`` paper IDs into those present in ``papers`` and those not.

    Runs one existence query up front rather than letting a downstream
    ``WHERE id IN (...)`` silently filter out unresolved IDs — the same
    caller-composes-commands failure mode ``missing_ids`` closes for
    fetch-target commands (``info``/``ocr``/``retry``), applied here to
    scoping-filter commands (``search``/``semantic --restrict-to-ids``,
    ``verify --ids``) where a bad ID would otherwise just shrink the
    candidate set with no signal.

    Both ``found`` and ``not_found`` preserve order of first appearance in
    ``requested``, with duplicates collapsed.

    Examples::

        resolve_ids(conn, [1, 2, 999])  -> ([1, 2], [999])
    """
    if not requested:
        return [], []
    placeholders = ",".join("?" * len(requested))
    rows = conn.execute(
        f"SELECT id FROM papers WHERE id IN ({placeholders})", requested
    ).fetchall()
    existing = {row["id"] for row in rows}
    not_found = missing_ids(requested, existing)
    found = [i for i in dict.fromkeys(requested) if i in existing]
    return found, not_found
