"""OpenAlex metadata provider.

OpenAlex needs no API key and offers a "polite pool" (100 req/s) when requests
carry a `mailto` tag. Lookups return the same normalized shape as the Semantic
Scholar provider in metadata.py so enrich_documents can stay provider-agnostic.
"""

import re
import time

OPENALEX_WORKS_API = "https://api.openalex.org/works"

_FILENAME_UNDERSCORE_RE = re.compile(r"([a-z]+)_(19|20)(\d\d)_")
_FILENAME_HYPHEN_RE = re.compile(r"([a-z]+)(-[a-z]+)?(-et-al)?-(19|20)(\d\d)-")
_DOI_URL_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Rebuild abstract text from OpenAlex's word -> [positions] inverted index."""
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(word for _, word in positions) or None


def parse_openalex_work(work: dict | None) -> dict | None:
    """Map a raw OpenAlex work into the normalized metadata shape."""
    if not work:
        return None

    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])
    ]

    doi = work.get("doi")
    if doi:
        doi = _DOI_URL_RE.sub("", doi).strip("/").lower() or None

    return {
        "title": work.get("title") or work.get("display_name"),
        "authors": [a for a in authors if a],
        "year": work.get("publication_year"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "doi": doi,
        "source_id": work.get("id"),
    }


def parse_filename(filename: str) -> tuple[str | None, int | None]:
    """Derive (first-author surname, year) from common filename conventions."""
    base = filename.lower().rsplit(".", 1)[0]

    m = _FILENAME_UNDERSCORE_RE.match(base)
    if m:
        return m.group(1), int(m.group(2) + m.group(3))

    m = _FILENAME_HYPHEN_RE.match(base)
    if m:
        return m.group(1), int(m.group(4) + m.group(5))

    return None, None


def _params(mailto: str | None, **extra) -> dict:
    params = dict(extra)
    if mailto:
        params["mailto"] = mailto
    return params


def fetch_by_doi(
    doi: str, mailto: str | None = None, rate_limit: float = 0.0
) -> dict | None:
    """Look up a work by exact DOI; returns normalized metadata or None."""
    import httpx

    if rate_limit:
        time.sleep(rate_limit)
    try:
        resp = httpx.get(
            f"{OPENALEX_WORKS_API}/doi:{doi}",
            params=_params(mailto),
            timeout=15,
        )
        if resp.status_code == 200:
            return parse_openalex_work(resp.json())
        return None
    except Exception:
        return None


def fetch_by_title(
    title: str, mailto: str | None = None, rate_limit: float = 0.0
) -> dict | None:
    """Search works by title; returns the top normalized result or None."""
    import httpx

    if rate_limit:
        time.sleep(rate_limit)
    try:
        resp = httpx.get(
            OPENALEX_WORKS_API,
            params=_params(mailto, search=title, **{"per-page": 1}),
            timeout=15,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                return parse_openalex_work(results[0])
        return None
    except Exception:
        return None


def fetch_by_filename(
    filename: str, mailto: str | None = None, rate_limit: float = 0.0
) -> dict | None:
    """Match a paper via its filename-derived author surname + year."""
    surname, year = parse_filename(filename)
    if not (surname and year):
        return None

    import httpx

    if rate_limit:
        time.sleep(rate_limit)
    try:
        resp = httpx.get(
            OPENALEX_WORKS_API,
            params=_params(
                mailto,
                filter=f"publication_year:{year},raw_author_name.search:{surname}",
                **{"per-page": 1},
            ),
            timeout=15,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                return parse_openalex_work(results[0])
        return None
    except Exception:
        return None
