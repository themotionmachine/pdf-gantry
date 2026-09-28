"""Query execution and result shaping shared by ``search`` and ``semantic``.

The CLI commands stay thin: they parse flags, open the DB, and hand each
query to ``run_search``/``run_semantic`` here. Keeping execution in one
place is what lets ``--queries-file`` run N queries in one process (one
model load) with exactly the same semantics as N single-query calls.

Search functions are looked up on the ``search`` module at call time
(``_search.hybrid_search`` rather than a direct import) so tests and callers can
substitute them.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from . import search as _search
from .models import SearchResult
from .utils import parse_authors

# Result keys an agent can ask for with --fields. Order is the JSON key order.
BASE_RESULT_FIELDS = (
    "id", "filename", "path", "title", "year", "authors", "citekey",
    "score", "snippet", "has_markdown", "has_embeddings",
)
COMPONENT_FIELDS = ("score_fts", "score_vector", "rank_fts", "rank_vector")
SEARCH_RESULT_FIELDS = BASE_RESULT_FIELDS + COMPONENT_FIELDS
SEMANTIC_RESULT_FIELDS = BASE_RESULT_FIELDS
# ``gantry info`` paper keys (chunks/top_chunk appear with --chunks/--query).
INFO_FIELDS = (
    "id", "filename", "path", "title", "authors", "year", "doi", "abstract",
    "snippet", "page_count", "has_text", "has_markdown", "has_embeddings",
    "has_chunk_embeddings", "is_scanned", "error_count", "citekey",
    "citekey_source", "chunks", "top_chunk",
)

# What ``total`` and ``returned`` mean, shared by --help and the README.
TOTAL_HELP = (
    "JSON counts: 'returned' is the number of results in this response. "
    "'total' is the number of matching papers when that is knowable: for "
    "--fts (unrestricted) it is the global FTS5 match count, independent of "
    "--limit; for hybrid, vector_fallback and semantic, which rank rather "
    "than match, total equals returned."
)


class FtsSyntaxError(Exception):
    """A raw ``--fts-syntax`` query that FTS5 rejected (FTS-only mode)."""


@dataclass
class QueryOutcome:
    query: str
    mode: str
    results: list[SearchResult]
    total: int
    warnings: list[str] = field(default_factory=list)
    fts_error: str | None = None

    @property
    def returned(self) -> int:
        return len(self.results)


EmbedFn = Callable[[str], bytes]


def run_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    embed: EmbedFn | None,
    fts_only: bool = False,
    fts_syntax: bool = False,
    limit: int = 20,
    restrict_ids: list[int] | None = None,
) -> QueryOutcome:
    """Run one ``gantry search`` query.

    Modes:
      ``hybrid``          FTS5 + vector RRF fusion both ran
      ``vector_fallback`` hybrid requested, FTS5 rejected the query, vector only
      ``fts``             hybrid requested, embeddings unavailable (degraded)
      ``fts_only``        caller passed --fts

    ``embed`` maps a query to a vector and may raise ``ImportError`` when
    the model is unavailable. Raises ``FtsSyntaxError`` when FTS5 rejects a
    raw-syntax query and there is no vector side to fall back to.
    """
    warnings: list[str] = []
    query_vec = None
    mode = "fts_only" if fts_only else "hybrid"
    if not fts_only:
        try:
            if embed is None:
                raise ImportError("no embedding function")
            query_vec = embed(query)
        except ImportError:
            warnings.append("Embeddings unavailable; falling back to FTS.")
            mode = "fts"

    if query_vec is not None:
        status: dict = {}
        results = _search.hybrid_search(
            conn, query, query_vec, limit=limit, restrict_ids=restrict_ids,
            syntax=fts_syntax, status=status,
        )
        mode = status.get("mode", "hybrid")
        fts_error = status.get("fts_error")
        if fts_error:
            warnings.append(f"FTS5 rejected the query ({fts_error}); results are vector-only.")
        return QueryOutcome(query, mode, results, len(results), warnings, fts_error)

    try:
        results = _search.fts_search(
            conn, query, limit=limit, restrict_ids=restrict_ids, syntax=fts_syntax,
        )
        if restrict_ids is not None:
            total = len(results)
        else:
            total = _search.search_count(conn, query, syntax=fts_syntax)
    except sqlite3.OperationalError as e:
        if fts_syntax:
            raise FtsSyntaxError(f"Invalid FTS5 syntax: {e}") from e
        raise
    return QueryOutcome(query, mode, results, total, warnings)


def run_semantic(
    conn: sqlite3.Connection,
    query_vec: bytes,
    query: str,
    *,
    limit: int = 20,
    doc_only: bool = False,
    restrict_ids: list[int] | None = None,
    has_chunks: bool | None = None,
) -> QueryOutcome:
    """Run one ``gantry semantic`` query against an already-embedded vector.

    Modes: ``cascade`` (doc filter + chunk retrieval) or ``doc`` (doc-level
    embeddings only: --doc-only, --restrict-to-ids, or no chunk embeddings).
    """
    if has_chunks is None:
        has_chunks = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE has_chunk_embeddings = 1"
        ).fetchone()[0] > 0
    if restrict_ids is not None:
        results = _search.semantic_search(conn, query_vec, limit=limit, restrict_ids=restrict_ids)
        mode = "doc"
    elif has_chunks and not doc_only:
        results = _search.cascade_search(conn, query_vec, limit=limit)
        mode = "cascade"
    else:
        results = _search.semantic_search(conn, query_vec, limit=limit)
        mode = "doc"
    return QueryOutcome(query, mode, results, len(results))


# --- result shaping --------------------------------------------------------

def paper_meta(conn: sqlite3.Connection, ids: list[int]) -> dict[int, dict]:
    """Fetch title/year/authors(list)/citekey for result papers in one query."""
    ids = list(dict.fromkeys(ids))
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, title, year, authors, citekey FROM papers WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {
        row["id"]: {
            "title": row["title"],
            "year": row["year"],
            "authors": parse_authors(row["authors"]),
            "citekey": row["citekey"],
        }
        for row in rows
    }


def result_dict(r: SearchResult, meta: dict[int, dict], *, components: bool = False) -> dict:
    m = meta.get(r.id, {})
    d = {
        "id": r.id,
        "filename": r.filename,
        "path": r.path,
        "title": m.get("title", r.title),
        "year": m.get("year"),
        "authors": m.get("authors", []),
        "citekey": m.get("citekey"),
        "score": r.score,
        "snippet": r.snippet,
        "has_markdown": r.has_markdown,
        "has_embeddings": r.has_embeddings,
    }
    if components:
        d["score_fts"] = r.score_fts
        d["score_vector"] = r.score_vector
        d["rank_fts"] = r.rank_fts
        d["rank_vector"] = r.rank_vector
    return d


def select_fields(d: dict, fields: list[str] | None) -> dict:
    if fields is None:
        return d
    wanted = set(fields)
    return {k: v for k, v in d.items() if k in wanted}


def split_fields(field_list: str | None, valid) -> tuple[list[str] | None, list[str]]:
    """Parse a ``--fields`` value into (requested, unknown).

    ``requested`` is ``None`` when the option was omitted. ``unknown`` lists
    names not in ``valid`` (in request order) so the caller can warn instead
    of silently dropping them.
    """
    if field_list is None:
        return None, []
    requested = [f.strip() for f in field_list.split(",") if f.strip()]
    valid_set = set(valid)
    unknown = [f for f in requested if f not in valid_set]
    return requested, unknown


def _clean_cell(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def display_title(r: SearchResult, meta: dict[int, dict]) -> str:
    title = meta.get(r.id, {}).get("title", r.title)
    return _clean_cell(title) or r.filename


def oneline(r: SearchResult, meta: dict[int, dict], index: int | None = None) -> str:
    """``id<TAB>score<TAB>year<TAB>citekey<TAB>title`` (title falls back to filename).

    With ``index``, the line is prefixed by ``index<TAB>`` (multi-query).
    """
    m = meta.get(r.id, {})
    cells = [
        str(r.id),
        f"{r.score:.4g}",
        _clean_cell(m.get("year")),
        _clean_cell(m.get("citekey")),
        display_title(r, meta),
    ]
    if index is not None:
        cells.insert(0, str(index))
    return "\t".join(cells)
