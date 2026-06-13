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
