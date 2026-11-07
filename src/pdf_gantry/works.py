"""Works: copies of the same paper folded into one unit.

A library collects the same paper more than once: a publisher PDF and a
preprint, a renamed re-download, a byte-identical duplicate. ``works_for``
groups every row of ``papers`` into works by union-find over three kinds of
evidence that two rows are the same paper:

- the same ``file_hash`` (identical bytes);
- an agreeing normalized DOI, when the display titles also agree and the page
  counts differ by 2 or less;
- the same normalized ``title`` plus ``year``, for titles of 5 or more words
  that are not generic, when the page counts differ by 2 or less.

A wrong merge hides a real paper, which is worse than showing a duplicate, so
every edge except the file hash is gated:

- a ``metadata_suspect`` row takes no DOI or title edge, because its DOI and
  title came from a match that doesn't resemble the PDF;
- when both rows have a title read off page 1 (``title_from_layout``), those
  must agree too. A DOI lookup stamps the same title on both rows, so the
  metadata titles agreeing proves little; the printed titles are independent.
  This is what keeps two chapters stamped with their book's DOI apart;
- page counts must both be known. A chapter is not its book.

``title_from_layout`` never creates a title edge: it is an unverified guess.

The key of a work is ``work:<smallest member id>``. Results are never re-id'd:
``collapse`` keeps the matched copy's own id (so page numbers stay right) and
adds ``work_key``, ``canonical_id`` and ``copies``.

The library-wide grouping is cached per database file. Reads never write: this
module only runs SELECTs and PRAGMA queries, and works on a plain ``mode=ro``
connection with no row factory.
"""

import dataclasses
import os
import re
import sqlite3
import threading
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from .link import normalize_doi
from .metadata import title_similarity
from .models import SearchResult
from .queue import BIB_SOURCE_PREFIX
from .utils import resolve_title

TITLE_SIMILARITY = 0.75
MAX_PAGE_DIFF = 2
MIN_TITLE_WORDS = 5
MIN_CONTAINED_WORDS = 3  # a shorter title must be this long to count as contained

# Titles that many different documents share. Matched against the normalized
# title (lowercase, punctuation dropped) as a whole or as its first words.
_GENERIC_EXACT = frozenset("""
introduction editorial preface foreword contents abstract abstracts index
references bibliography acknowledgments acknowledgements conclusion conclusions
erratum corrigendum errata commentary comment reply response discussion
obituary afterword epilogue prologue appendix glossary notes
""".split()) | {
    "book review", "book reviews", "table of contents", "front matter",
    "back matter", "list of contributors", "notes on contributors",
    "about the authors", "guest editorial", "in memoriam", "letter to the editor",
    "editorial board", "title page", "copyright page",
}
_GENERIC_PREFIXES = (
    "introduction to the special issue", "introduction to this special issue",
    "introduction to the symposium", "introduction to the volume",
    "editorial", "guest editorial", "book review", "review of", "reply to",
    "response to", "a reply to", "a response to", "comment on", "commentary on",
    "erratum", "corrigendum", "correction to", "retraction", "preface",
    "foreword", "table of contents", "contents", "front matter", "back matter",
    "letter to the editor", "notes on contributors", "in memoriam", "obituary",
    "special issue",
)

_NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
_SPACE_RE = re.compile(r"\s+")

_COLUMNS = (
    "id, filename, file_hash, doi, title, title_from_layout, year, page_count, "
    "metadata_suspect, citekey, metadata_source, has_text, indexed_at"
)


@dataclass(frozen=True)
class Work:
    key: str                     # "work:<smallest member id>"
    canonical_id: int
    member_ids: tuple[int, ...]  # every copy in the library, ascending


@dataclass(frozen=True)
class _Row:
    id: int
    filename: str
    file_hash: str | None
    doi: str | None
    title: str | None
    layout: str | None
    year: int | None
    pages: int | None
    suspect: bool
    citekey: str | None
    source: str | None
    has_text: bool
    indexed_at: str | None

    @property
    def display_title(self) -> str:
        return resolve_title(self.title, self.layout, self.filename or "")[0]


# --- normalization ----------------------------------------------------------------

def normalize_title(title: str | None) -> str:
    """Lowercase ASCII words: accents folded, punctuation dropped, spaces collapsed."""
    if not title:
        return ""
    t = unicodedata.normalize("NFKD", title)
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    return _SPACE_RE.sub(" ", _NON_WORD_RE.sub(" ", t)).strip()


def is_generic_title(norm: str) -> bool:
    """Whether a normalized title is one many unrelated documents share."""
    if norm in _GENERIC_EXACT:
        return True
    return any(norm == p or norm.startswith(p + " ") for p in _GENERIC_PREFIXES)


def _titles_agree(a: str | None, b: str | None) -> bool:
    """Similarity >= 0.75, or one title contained (on word boundaries) in the other."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return False
    if na == nb or title_similarity(a, b) >= TITLE_SIMILARITY:
        return True
    short, long_ = sorted((na, nb), key=len)
    if len(short.split()) < MIN_CONTAINED_WORDS or is_generic_title(short):
        return False
    return f" {short} " in f" {long_} "


def _pages_close(a: _Row, b: _Row) -> bool:
    return a.pages is not None and b.pages is not None and abs(a.pages - b.pages) <= MAX_PAGE_DIFF


def _printed_titles_agree(a: _Row, b: _Row) -> bool:
    """Layout titles, when both rows have one, must not contradict the match."""
    if not (a.layout and a.layout.strip() and b.layout and b.layout.strip()):
        return True
    return _titles_agree(a.layout, b.layout)


def _doi_edge(a: _Row, b: _Row) -> bool:
    return (not a.suspect and not b.suspect and _pages_close(a, b)
            and _titles_agree(a.display_title, b.display_title)
            and _printed_titles_agree(a, b))


def _title_edge(a: _Row, b: _Row) -> bool:
    return (not a.suspect and not b.suspect and _pages_close(a, b)
            and _printed_titles_agree(a, b))


# --- canonical copy -------------------------------------------------------------------

def _source_rank(source: str | None) -> int:
    """manual > bib > openalex > anything else (other providers, misses, none)."""
    s = (source or "").lower()
    if s.startswith(BIB_SOURCE_PREFIX):
        return 1
    if "manual" in s:  # 'manual:<who>' and legacy free-form names (queue.manual_condition)
        return 0
    if s.startswith("openalex"):
        return 2
    return 3


def _canonical(rows: list[_Row]) -> int:
    # Newest indexed_at first: sort ascending on everything else, then pick
    # with a stable two-pass sort so ISO strings need no negation.
    by_newest = sorted(rows, key=lambda r: (r.indexed_at or "", -r.id), reverse=True)
    best = min(by_newest, key=lambda r: (
        0 if (r.citekey or "").strip() else 1,
        _source_rank(r.source),
        1 if r.suspect else 0,
        0 if r.has_text else 1,
    ))
    return best.id


# --- grouping -----------------------------------------------------------------------

def _load_rows(conn: sqlite3.Connection) -> list[_Row]:
    cur = conn.cursor()
    cur.row_factory = None  # read by position whatever the connection's factory
    return [
        _Row(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], bool(r[8]), r[9], r[10],
             bool(r[11]), r[12])
        for r in cur.execute(f"SELECT {_COLUMNS} FROM papers ORDER BY id")
    ]


def _group(rows: list[_Row]) -> dict[int, Work]:
    parent = {r.id: r.id for r in rows}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_hash: dict[str, list[_Row]] = defaultdict(list)
    by_doi: dict[str, list[_Row]] = defaultdict(list)
    by_title: dict[tuple[str, int], list[_Row]] = defaultdict(list)
    for r in rows:
        if r.file_hash:
            by_hash[r.file_hash].append(r)
        if r.doi and r.doi.strip():
            by_doi[normalize_doi(r.doi.strip())].append(r)
        if r.title and r.year is not None:
            norm = normalize_title(r.title)
            if len(norm.split()) >= MIN_TITLE_WORDS and not is_generic_title(norm):
                by_title[(norm, r.year)].append(r)

    for members in by_hash.values():
        for r in members[1:]:
            union(members[0].id, r.id)
    for buckets, edge in ((by_doi, _doi_edge), (by_title, _title_edge)):
        for members in buckets.values():
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    if edge(a, b):
                        union(a.id, b.id)

    groups: dict[int, list[_Row]] = defaultdict(list)
    for r in rows:
        groups[find(r.id)].append(r)
    out: dict[int, Work] = {}
    for members in groups.values():
        ids = tuple(sorted(r.id for r in members))
        work = Work(key=f"work:{ids[0]}", canonical_id=_canonical(members), member_ids=ids)
        for pid in ids:
            out[pid] = work
    return out


# --- cache ------------------------------------------------------------------------------
#
# PRAGMA data_version only means something within one connection: it changes
# when another connection commits, and not for the connection's own writes. So
# a hit on the same connection needs data_version and total_changes unchanged.
# Callers like Ramelli open a fresh connection per request, and a fresh
# connection's data_version is not comparable with an old one's, so across
# connections the cache is keyed on the database file itself: the size and
# mtime of the file and its -wal, which every commit changes.

@dataclass
class _Entry:
    conn_id: int  # id() of the connection that last validated the entry
    data_version: int
    total_changes: int
    file_stamp: tuple | None
    works: dict[int, Work]


_cache: dict[str, _Entry] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _main_file(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.row_factory = None
    for row in cur.execute("PRAGMA database_list"):
        if row[1] == "main":
            return row[2] or ""
    return ""


def _file_stamp(path: str) -> tuple | None:
    stamp = []
    for p in (path, path + "-wal"):
        try:
            st = os.stat(p)
        except OSError:
            stamp.append(None)
        else:
            stamp.append((st.st_size, st.st_mtime_ns, st.st_ino))
    return tuple(stamp)


def _data_version(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.row_factory = None
    return cur.execute("PRAGMA data_version").fetchone()[0]


def _library_works(conn: sqlite3.Connection) -> dict[int, Work]:
    path = _main_file(conn)
    key = path or f"memory:{id(conn)}"
    data_version = _data_version(conn)
    total_changes = conn.total_changes
    stamp = _file_stamp(path) if path else None
    with _lock:
        entry = _cache.get(key)
    if entry is not None:
        # id() can be reused by a later connection; for a file the stamp
        # check below still guards that case.
        same_conn = entry.conn_id == id(conn)
        if same_conn:
            fresh = (entry.data_version == data_version
                     and entry.total_changes == total_changes
                     and entry.file_stamp == stamp)
        else:
            fresh = stamp is not None and entry.file_stamp == stamp
        if fresh:
            if not same_conn:
                entry.conn_id = id(conn)
                entry.data_version, entry.total_changes = data_version, total_changes
            return entry.works
    grouped = _group(_load_rows(conn))
    with _lock:
        _cache[key] = _Entry(id(conn), data_version, total_changes, stamp, grouped)
    return grouped


# --- public API ---------------------------------------------------------------------------

def works_for(conn: sqlite3.Connection, ids) -> dict[int, Work]:
    """Each id that is in the library, mapped to its Work (library-wide members)."""
    library = _library_works(conn)
    return {pid: library[pid] for pid in ids if pid in library}


def collapse(conn: sqlite3.Connection, results: list[SearchResult],
             limit: int) -> list[SearchResult]:
    """One result per work, in rank order, truncated to ``limit``.

    Keeps the best-ranked matched copy of each work with its own id, and sets
    ``work_key``, ``canonical_id`` and ``copies`` (the work's other member ids,
    library-wide). Inputs are not modified. Callers over-fetch.
    """
    found = works_for(conn, [r.id for r in results])
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        work = found.get(r.id)
        if work is None:  # not in the library: pass through, unfolded
            out.append(dataclasses.replace(r, copies=[]))
        else:
            if work.key in seen:
                continue
            seen.add(work.key)
            out.append(dataclasses.replace(
                r, work_key=work.key, canonical_id=work.canonical_id,
                copies=[m for m in work.member_ids if m != r.id],
            ))
        if len(out) >= limit:
            break
    return out
