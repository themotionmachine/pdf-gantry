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


def enrich_documents(
    conn: sqlite3.Connection,
    paper_ids: list[int] | None = None,
    rate_limit: float = 0.1,
    limit: int | None = None,
    progress_callback=None,
) -> EnrichStats:
    """Fetch metadata from Semantic Scholar for documents."""
    stats = EnrichStats()
    start = time.time()

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
                metadata = _fetch_by_doi(doi, rate_limit)
                if metadata:
                    source = "semantic_scholar"
            except Exception:
                stats.api_errors += 1

        # Fallback: title search
        if not metadata and raw_text:
            title_guess = _extract_title_from_text(raw_text)
            if title_guess:
                try:
                    metadata = _fetch_by_title(title_guess, rate_limit)
                    if metadata:
                        source = "semantic_scholar_title"
                        stats.matched_by_title += 1
                except Exception:
                    stats.api_errors += 1

        if metadata:
            now = now_iso()
            authors = json.dumps([a.get("name", "") for a in metadata.get("authors", [])])
            conn.execute(
                """UPDATE papers SET
                    title = COALESCE(?, title),
                    authors = ?,
                    year = ?,
                    abstract = ?,
                    semantic_scholar_id = ?,
                    metadata_source = ?,
                    metadata_enriched_at = ?,
                    updated_at = ?
                WHERE id = ?""",
                (
                    metadata.get("title"),
                    authors,
                    metadata.get("year"),
                    metadata.get("abstract"),
                    metadata.get("paperId"),
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
