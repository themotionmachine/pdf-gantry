"""Semantic Scholar API integration for metadata enrichment."""

import json
import re
import sqlite3
import time
from dataclasses import dataclass

from .utils import now_iso

DOI_PATTERN = re.compile(r'10\.\d{4,}/[^\s]+')

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"
FIELDS = "title,authors,year,abstract,citationCount,influentialCitationCount,externalIds"


# Fields `enrich_documents` populates from a provider. `metadata_enriched_at`
# being set does NOT mean these are filled in -- a paper matched by title
# (no DOI exists) or one whose provider has no abstract on file still gets
# marked "enriched", and every filter in queue.py (`needs metadata`) only
# looks at `doi IS NULL OR metadata_enriched_at IS NULL`. Once enrichment has
# run once, a permanently-incomplete paper is invisible to every existing
# command. field_completeness()/gap_ids() exist to make that gap visible.
GAP_FIELDS = ("title", "authors", "year", "abstract", "doi")

_FIELD_EMPTY_SQL = {
    "title": "(title IS NULL OR title = '')",
    # authors is a JSON-encoded list; '[]' is non-NULL but zero authors.
    "authors": "(authors IS NULL OR authors = '' OR authors = '[]')",
    "year": "(year IS NULL)",
    "abstract": "(abstract IS NULL OR abstract = '')",
    "doi": "(doi IS NULL OR doi = '')",
}


def _check_fields(fields: list[str]) -> None:
    unknown = [f for f in fields if f not in _FIELD_EMPTY_SQL]
    if unknown:
        raise ValueError(
            f"Unknown metadata field(s): {', '.join(unknown)}. "
            f"Valid fields: {', '.join(GAP_FIELDS)}"
        )


def field_completeness(
    conn: sqlite3.Connection, fields: list[str] | None = None
) -> dict:
    """Per-field metadata completeness across the whole corpus.

    For each field, splits the gap into:
      - ``never_attempted``: enrichment has never run for this paper
        (``metadata_enriched_at IS NULL``) -- the ordinary, expected gap.
      - ``attempted_incomplete``: enrichment ran and the field is *still*
        empty. Re-running ``enrich`` with the same provider won't fix
        these -- the provider simply doesn't have the data (or the paper
        was matched by title with no DOI to find). These are the gaps
        that look identical to "done" everywhere else in gantry.
    """
    fields = list(fields) if fields is not None else list(GAP_FIELDS)
    _check_fields(fields)

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    field_stats = {}
    for field in fields:
        empty = _FIELD_EMPTY_SQL[field]
        row = conn.execute(
            f"""SELECT
                SUM(CASE WHEN {empty} THEN 1 ELSE 0 END),
                SUM(CASE WHEN {empty} AND metadata_enriched_at IS NULL
                    THEN 1 ELSE 0 END),
                SUM(CASE WHEN {empty} AND metadata_enriched_at IS NOT NULL
                    THEN 1 ELSE 0 END)
            FROM papers"""
        ).fetchone()
        missing = row[0] or 0
        never_attempted = row[1] or 0
        attempted_incomplete = row[2] or 0
        field_stats[field] = {
            "missing": missing,
            "never_attempted": never_attempted,
            "attempted_incomplete": attempted_incomplete,
            "complete": total - missing,
        }
    return {"total": total, "fields": field_stats}


def gap_ids(
    conn: sqlite3.Connection, field: str, attempted_only: bool = False
) -> list[int]:
    """Paper IDs missing ``field``, sorted ascending.

    With ``attempted_only=True``, scopes to papers where enrichment already
    ran and still left the field empty -- the silent-failure bucket that
    ``queue --needs metadata`` never re-surfaces. Chain straight into
    ``gantry info --ids`` or ``gantry enrich --ids`` (via a different
    ``--provider``) without round-tripping full JSON through an agent.
    """
    _check_fields([field])
    where = _FIELD_EMPTY_SQL[field]
    if attempted_only:
        where += " AND metadata_enriched_at IS NOT NULL"
    rows = conn.execute(f"SELECT id FROM papers WHERE {where} ORDER BY id").fetchall()
    return [r[0] for r in rows]


@dataclass
class EnrichStats:
    """Statistics from metadata enrichment."""
    total: int = 0
    doi_found: int = 0
    matched_by_title: int = 0
    no_match: int = 0
    api_errors: int = 0
    elapsed_seconds: float = 0.0


def extract_doi(text: str) -> str | None:
    """Extract a DOI from text content."""
    match = DOI_PATTERN.search(text)
    if match:
        doi = match.group(0)
        # Clean trailing punctuation
        doi = doi.rstrip(".,;:)]}")
        return doi
    return None


def _fetch_by_doi(doi: str, rate_limit: float = 0.1) -> dict | None:
    """Fetch paper metadata from Semantic Scholar by DOI."""
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx not installed. Run: pip install httpx")

    url = f"{SEMANTIC_SCHOLAR_API}/DOI:{doi}?fields={FIELDS}"
    time.sleep(rate_limit)  # Rate limiting

    try:
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
        else:
            return None
    except Exception:
        return None


def _fetch_by_title(title: str, rate_limit: float = 0.1) -> dict | None:
    """Search Semantic Scholar by title."""
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx not installed. Run: pip install httpx")

    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1&fields={FIELDS}"
    time.sleep(rate_limit)

    try:
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                return data["data"][0]
        return None
    except Exception:
        return None


def _extract_title_from_text(raw_text: str) -> str | None:
    """Attempt to extract a title from the first few lines of PDF text."""
    lines = raw_text.strip().split("\n")[:10]
    # Find the first non-empty, non-trivial line
    for line in lines:
        line = line.strip()
        if len(line) > 10 and not line.startswith("http"):
            return line[:200]
    return None


def _normalize_semantic_scholar(raw: dict | None) -> dict | None:
    """Map a Semantic Scholar paper into the shared normalized metadata shape."""
    if not raw:
        return None
    return {
        "title": raw.get("title"),
        "authors": [a.get("name", "") for a in raw.get("authors", [])],
        "year": raw.get("year"),
        "abstract": raw.get("abstract"),
        "doi": (raw.get("externalIds") or {}).get("DOI"),
        "source_id": raw.get("paperId"),
    }


def _semantic_scholar_provider(rate_limit: float) -> dict:
    return {
        "by_doi": lambda doi: _normalize_semantic_scholar(
            _fetch_by_doi(doi, rate_limit)
        ),
        "by_filename": None,
        "by_title": lambda title: _normalize_semantic_scholar(
            _fetch_by_title(title, rate_limit)
        ),
        "doi_source": "semantic_scholar",
        "filename_source": "semantic_scholar_filename",
        "title_source": "semantic_scholar_title",
        "is_semantic_scholar": True,
    }


def _openalex_provider(rate_limit: float, mailto: str | None) -> dict:
    from . import openalex

    return {
        "by_doi": lambda doi: openalex.fetch_by_doi(
            doi, mailto=mailto, rate_limit=rate_limit
        ),
        "by_filename": lambda filename: openalex.fetch_by_filename(
            filename, mailto=mailto, rate_limit=rate_limit
        ),
        "by_title": lambda title: openalex.fetch_by_title(
            title, mailto=mailto, rate_limit=rate_limit
        ),
        "doi_source": "openalex",
        "filename_source": "openalex_filename",
        "title_source": "openalex_title",
        "is_semantic_scholar": False,
    }


def _get_provider(provider: str, rate_limit: float, mailto: str | None) -> dict:
    if provider in ("semantic-scholar", "semantic_scholar", "s2"):
        return _semantic_scholar_provider(rate_limit)
    return _openalex_provider(rate_limit, mailto)


def enrich_documents(
    conn: sqlite3.Connection,
    paper_ids: list[int] | None = None,
    rate_limit: float = 0.1,
    limit: int | None = None,
    progress_callback=None,
    provider: str = "openalex",
    mailto: str | None = None,
) -> EnrichStats:
    """Fetch metadata for documents from the chosen provider (OpenAlex default)."""
    stats = EnrichStats()
    start = time.time()
    prov = _get_provider(provider, rate_limit, mailto)

    if paper_ids is not None:
        placeholders = ",".join("?" * len(paper_ids))
        rows = conn.execute(
            f"""SELECT p.id, p.filename, p.doi, pt.raw_text
            FROM papers p
            LEFT JOIN paper_text pt ON pt.paper_id = p.id
            WHERE p.id IN ({placeholders})""",
            paper_ids,
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT p.id, p.filename, p.doi, pt.raw_text
            FROM papers p
            LEFT JOIN paper_text pt ON pt.paper_id = p.id
            WHERE p.metadata_enriched_at IS NULL"""
        ).fetchall()

    if limit:
        rows = rows[:limit]

    stats.total = len(rows)
    completed = 0

    for row in rows:
        paper_id = row["id"]
        doi = row["doi"]
        filename = row["filename"]
        raw_text = row["raw_text"] or ""

        metadata = None
        source = None

        # Try DOI first
        if not doi and raw_text:
            doi = extract_doi(raw_text)
            if doi:
                conn.execute("UPDATE papers SET doi = ? WHERE id = ?", (doi, paper_id))

        if doi:
            stats.doi_found += 1
            try:
                metadata = prov["by_doi"](doi)
                if metadata:
                    source = prov["doi_source"]
            except Exception:
                stats.api_errors += 1

        # Fallback: filename-derived author+year lookup (provider-dependent)
        if not metadata and prov["by_filename"] and filename:
            try:
                metadata = prov["by_filename"](filename)
                if metadata:
                    source = prov["filename_source"]
            except Exception:
                stats.api_errors += 1

        # Fallback: title search
        if not metadata and raw_text:
            title_guess = _extract_title_from_text(raw_text)
            if title_guess:
                try:
                    metadata = prov["by_title"](title_guess)
                    if metadata:
                        source = prov["title_source"]
                        stats.matched_by_title += 1
                except Exception:
                    stats.api_errors += 1

        if metadata:
            now = now_iso()
            authors = json.dumps(metadata.get("authors") or [])
            ss_id = (
                metadata.get("source_id")
                if prov["is_semantic_scholar"] else None
            )
            conn.execute(
                """UPDATE papers SET
                    title = COALESCE(?, title),
                    authors = ?,
                    year = ?,
                    abstract = ?,
                    semantic_scholar_id = COALESCE(?, semantic_scholar_id),
                    doi = COALESCE(?, doi),
                    metadata_source = ?,
                    metadata_enriched_at = ?,
                    updated_at = ?
                WHERE id = ?""",
                (
                    metadata.get("title"),
                    authors,
                    metadata.get("year"),
                    metadata.get("abstract"),
                    ss_id,
                    metadata.get("doi"),
                    source,
                    now, now, paper_id,
                ),
            )
        else:
            if not (doi and stats.api_errors > 0):
                stats.no_match += 1
            # Still mark as checked
            now = now_iso()
            conn.execute(
                "UPDATE papers SET metadata_enriched_at = ?, updated_at = ? WHERE id = ?",
                (now, now, paper_id),
            )

        completed += 1
        if completed % 10 == 0:
            conn.commit()
        if progress_callback:
            progress_callback(completed, stats.total)

    conn.commit()
    stats.elapsed_seconds = round(time.time() - start, 1)
    return stats
