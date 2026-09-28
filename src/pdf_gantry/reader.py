"""Per-paper chunk addressing for ``gantry read --index``.

``chunk_id`` is a global row ID; ``chunk_index`` is a chunk's position
within its paper. Agents think in the latter ("chunks 28-41 of paper 53"),
so ``read ID --index A-B`` resolves a contiguous per-paper range in one
query instead of pulling every chunk via ``info --chunks`` and filtering.
Pure-DB: never opens a PDF.
"""

import re
import sqlite3

_RANGE_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def parse_index_range(value: str) -> tuple[int, int]:
    """Parse ``"N"`` or ``"A-B"`` (inclusive, A <= B) into ``(start, end)``."""
    m = _RANGE_RE.match(value or "")
    if not m:
        raise ValueError(f"invalid index range {value!r}: expected N or A-B")
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) is not None else start
    if end < start:
        raise ValueError(f"invalid index range {value!r}: end is before start")
    return start, end


def page_columns(conn: sqlite3.Connection) -> list[str]:
    """Chunk columns carrying page information (``page_start`` and any later
    additions such as ``page_end``), so output picks them up when present."""
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()]
    return [c for c in cols if c.startswith("page_")]


def chunks_by_index(
    conn: sqlite3.Connection, paper_id: int, start: int, end: int
) -> list[dict]:
    """Chunks of ``paper_id`` with ``start <= chunk_index <= end``, in order.

    Each dict carries ``chunk_id``, ``chunk_index``, ``section_header``,
    ``char_offset``, ``char_end`` (offset + text length), every ``page_*``
    column present on ``chunks``, and ``text``.
    """
    pages = page_columns(conn)
    page_sql = "".join(f", {c}" for c in pages)
    rows = conn.execute(
        f"""SELECT chunk_id, chunk_index, section_header, char_offset{page_sql}, text
            FROM chunks
            WHERE doc_id = ? AND chunk_index BETWEEN ? AND ?
            ORDER BY chunk_index""",
        (paper_id, start, end),
    ).fetchall()
    out = []
    for r in rows:
        d = {
            "chunk_id": r["chunk_id"],
            "chunk_index": r["chunk_index"],
            "section_header": r["section_header"],
            "char_offset": r["char_offset"],
            "char_end": r["char_offset"] + len(r["text"]),
        }
        for c in pages:
            d[c] = r[c]
        d["text"] = r["text"]
        out.append(d)
    return out


def chunk_count(conn: sqlite3.Connection, paper_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (paper_id,)
    ).fetchone()[0]
