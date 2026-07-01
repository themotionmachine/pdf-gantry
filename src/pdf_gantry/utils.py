"""Shared utilities: hashing, formatting, helpers."""

import hashlib
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
