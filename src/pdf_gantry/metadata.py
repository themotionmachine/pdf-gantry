"""Semantic Scholar API integration for metadata enrichment."""

import difflib
import json
import re
import sqlite3
import time
from dataclasses import dataclass

from .queue import (
    MISS_SOURCE_PREFIX,
    enriched_condition,
    manual_condition,
    miss_condition,
    never_enriched_condition,
)
from .utils import now_iso

DOI_PATTERN = re.compile(r'10\.\d{4,}/[^\s]+')

# A paper's OWN doi is printed in its front matter; every other doi in the file
# belongs to something it cites. Searching the whole document and taking the
# first hit (the pre-2026-09 behaviour) therefore mis-identifies any paper whose
# doi is not printed on page 1 — it silently adopts a reference-list doi and,
# because the doi tier is treated as exact, writes that work's title/authors/year
# over the paper with full confidence. Bound the search instead.
DOI_HEAD_CHARS = 6000  # ~first two pages of extracted text

# Never read a doi out of the reference section, however early it starts.
_REFERENCES_RE = re.compile(
    r'\n\s*(references|bibliography|works cited|reference list|notes and references)\s*\n',
    re.I,
)

# Front-matter dois are usually adjacent to one of these; reference-list dois
# rarely are (they sit after a page range or a closing quote).
_SELF_DOI_CONTEXT = re.compile(
    r'(doi\.org|dx\.doi\.org|\bdoi\b\s*[:.]|to cite|how to cite|cite this|permalink)',
    re.I,
)

_TITLE_NORMALIZE_RE = re.compile(r'[^a-z0-9\s]')
_WHITESPACE_RE = re.compile(r'\s+')

# Below this similarity, a title-search-sourced metadata match is flagged
# `metadata_suspect` — the stored title doesn't resemble what's actually on
# the PDF's first page closely enough to trust the DOI/authors/year that
# rode in with it. Calibrated loosely (SequenceMatcher ratio, not a strict
# edit distance): distinguishing titles score well above this; an unrelated
# title lands well below it.
SUSPECT_THRESHOLD = 0.55

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
    miss = miss_condition()
    enrich_misses = conn.execute(
        f"SELECT COUNT(*) FROM papers WHERE {miss}"
    ).fetchone()[0]
    field_stats = {}
    for field in fields:
        empty = _FIELD_EMPTY_SQL[field]
        row = conn.execute(
            f"""SELECT
                SUM(CASE WHEN {empty} THEN 1 ELSE 0 END),
                SUM(CASE WHEN {empty} AND metadata_enriched_at IS NULL
                    THEN 1 ELSE 0 END),
                SUM(CASE WHEN {empty} AND metadata_enriched_at IS NOT NULL
                    THEN 1 ELSE 0 END),
                SUM(CASE WHEN {empty} AND {miss} THEN 1 ELSE 0 END)
            FROM papers"""
        ).fetchone()
        missing = row[0] or 0
        never_attempted = row[1] or 0
        attempted_incomplete = row[2] or 0
        field_stats[field] = {
            "missing": missing,
            "never_attempted": never_attempted,
            "attempted_incomplete": attempted_incomplete,
            # Subset of attempted_incomplete where the provider matched
            # nothing at all (retry with `enrich --retry-misses`).
            "provider_miss": row[3] or 0,
            "complete": total - missing,
        }
    return {"total": total, "enrich_misses": enrich_misses, "fields": field_stats}


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
    suspect: int = 0
    skipped_manual: int = 0
    matched: int = 0
    elapsed_seconds: float = 0.0


def extract_doi(text: str, head_chars: int = DOI_HEAD_CHARS) -> str | None:
    """Extract the paper's OWN doi from its front matter.

    Scoped deliberately: only the first ``head_chars`` of extracted text, and
    never past a reference-section heading. Among candidates in that window,
    prefer one that appears in a self-citation context (``doi.org``, ``DOI:``,
    ``How to cite``) and then the one that repeats most often — a paper's own
    doi is typically printed more than once in its front matter, a cited one
    is not. Returns None rather than guessing when the window holds no doi;
    the caller falls through to the title-search tier, which is verified.
    """
    if not text:
        return None

    cut = len(text)
    ref = _REFERENCES_RE.search(text)
    if ref:
        cut = ref.start()
    head = text[:min(cut, head_chars)]

    candidates: list[tuple[str, int]] = []  # (doi, start offset)
    for match in DOI_PATTERN.finditer(head):
        doi = match.group(0).rstrip(".,;:)]}")
        if doi:
            candidates.append((doi, match.start()))
    if not candidates:
        return None

    counts: dict[str, int] = {}
    contextual: dict[str, bool] = {}
    for doi, start in candidates:
        counts[doi] = counts.get(doi, 0) + 1
        if not contextual.get(doi):
            window = head[max(0, start - 60):start]
            contextual[doi] = bool(_SELF_DOI_CONTEXT.search(window))

    def rank(item: tuple[str, int]) -> tuple[int, int, int]:
        doi, start = item
        # contextual first, then most-repeated, then earliest
        return (0 if contextual.get(doi) else 1, -counts[doi], start)

    return min(candidates, key=rank)[0]


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


# --- verification by containment -------------------------------------------
#
# Verifying a metadata match does NOT require locating the title on the page —
# only deciding whether the provider's answer is present in this document. That
# reframing sidesteps the whole masthead/cover-sheet problem: an extractor has
# to pick the right line, containment does not care where the words are.
#
# Measured on 142 papers whose citekey links them to a human-curated .bib
# entry (ground truth independent of any heuristic here): at threshold 0.6,
# 92.3% of true title/document pairs pass and 2.1% of deliberately mismatched
# pairs do. The line-picking extractor below recovers a usable title for only
# ~half of the same corpus, which is why it must not be the verifier.

CONTAINMENT_HEAD_CHARS = 6000
CONTAINMENT_THRESHOLD = 0.6

# Words too common in academic titles to evidence a match.
_TITLE_STOPWORDS = frozenset("""
the a an of and or for in on to with by from as at is are be that this how what
why when who whose which its their our not no new using use toward towards
between within into over under after before more most less least case study
studies analysis approach evidence role impact effects effect
""".split())

_CONTENT_WORD_RE = re.compile(r"[a-z][a-z'\-]{3,}")


def title_containment(
    title: str | None,
    raw_text: str | None,
    head_chars: int = CONTAINMENT_HEAD_CHARS,
) -> float:
    """Fraction of a title's distinctive words present in a document's opening.

    Position-free by design — see the note above. Returns 0.0 when either side
    is missing, so an absent title never reads as a confident match.
    """
    if not title or not raw_text:
        return 0.0
    tokens = [w for w in _CONTENT_WORD_RE.findall(title.lower())
              if w not in _TITLE_STOPWORDS]
    if not tokens:  # a title made entirely of stopwords ("The Case For It")
        tokens = _CONTENT_WORD_RE.findall(title.lower())
    if not tokens:
        return 0.0
    present = set(_CONTENT_WORD_RE.findall(raw_text[:head_chars].lower()))
    return sum(1 for w in tokens if w in present) / len(tokens)


# --- title extraction (for the title-SEARCH query tier only) ---------------

_LINE_NORMALIZE_RE = re.compile(r"\d+")
_FRONT_MATTER_END_RE = re.compile(
    r"^\s*(abstract|a b s t r a c t|keywords|key words|introduction|summary)\b", re.I
)
_LINE_JUNK_RE = re.compile(
    r"^\s*(doi|issn|isbn|vol\.?|volume|no\.?|pp?\.|https?:|www\.|©|copyright|"
    r"downloaded|received|accepted|published|available online|contents lists|"
    r"journal homepage|article|research article|original article|open access|"
    r"this content|all use subject|view |submit your|citing articles|full terms|"
    r"to link to this|electronic copy|preprint|working paper no)\b",
    re.I,
)
_AFFILIATION_RE = re.compile(
    r"\b(university|universiteit|universit[eé]|department|dept\.|institute|"
    r"faculty|school of|centre for|center for|college of|academy of|laborator)\b",
    re.I,
)
_NAME_PARTICLES = frozenset(
    ("and", "de", "van", "der", "von", "del", "la", "le", "y")
)


def normalize_line(line: str) -> str:
    """Fold a line to a comparable key: digits collapsed, whitespace squeezed."""
    return _LINE_NORMALIZE_RE.sub("#", " ".join(line.split()).lower())


def _looks_like_authors(text: str) -> bool:
    """Heuristic byline detector.

    Capitalisation alone is not enough — a Title Case title is capitalised too
    ("What Is Actually Being Annotated?" was misread as a byline until this was
    tightened). Require positive evidence of names: a separator between people,
    or an initial like "Q.".
    """
    tokens = text.replace(",", " ").split()
    if not 2 <= len(tokens) <= 14:
        return False
    capitalized = sum(
        1 for t in tokens if t[:1].isupper() or (len(t) <= 3 and t.endswith("."))
    )
    if capitalized / len(tokens) < 0.8:
        return False
    if any(t in _NAME_PARTICLES for t in tokens if t[:1].islower()) is False and any(
        t[:1].islower() for t in tokens
    ):
        return False
    has_separator = ("," in text) or (" and " in text) or (" & " in text)
    has_initial = any(
        len(t.rstrip(",")) <= 3 and t.rstrip(",").endswith(".") and t[:1].isupper()
        for t in text.split()
    )
    return has_separator or has_initial


def _alpha_ratio(text: str) -> float:
    return sum(c.isalpha() or c.isspace() for c in text) / max(1, len(text))


def _extract_title_from_text(
    raw_text: str,
    boilerplate: frozenset[str] | set[str] = frozenset(),
    max_lines: int = 30,
) -> str | None:
    """Best-effort title from a PDF's front matter, for use as a search query.

    Boilerplate is supplied by the caller and defined by recurrence rather than
    by rules: a title appears in one document, a masthead or cover sheet
    appears in every paper from that publisher (see ``corpus_boilerplate``).
    Author and affiliation lines are dropped heuristically, and the title is
    taken to be the FIRST substantial block of surviving front matter — not the
    longest, which walks into the abstract.
    """
    if not raw_text:
        return None
    lines = raw_text.split("\n")[:max_lines]

    end = len(lines)
    for i, line in enumerate(lines):
        if i and _FRONT_MATTER_END_RE.match(line):
            end = i
            break

    candidates: list[tuple[int, str]] = []
    for i, line in enumerate(lines[:end]):
        text = " ".join(line.split())
        if not 12 <= len(text) <= 250:
            continue
        if normalize_line(text) in boilerplate:
            continue
        if _LINE_JUNK_RE.match(text) or "@" in text:
            continue
        if _alpha_ratio(text) < 0.65:
            continue
        if _AFFILIATION_RE.search(text) and (text[0].isdigit() or "," in text):
            continue
        if _looks_like_authors(text):
            continue
        candidates.append((i, text))
    if not candidates:
        # Nothing survived: fall back to the old behaviour rather than give up,
        # since a weak query still beats no query for the title-search tier.
        for line in lines[:10]:
            stripped = line.strip()
            if len(stripped) > 10 and not stripped.startswith("http"):
                return stripped[:200]
        return None

    groups: list[list[tuple[int, str]]] = [[candidates[0]]]
    for previous, item in zip(candidates, candidates[1:]):
        if item[0] == previous[0] + 1 and len(groups[-1]) < 4:
            groups[-1].append(item)
        else:
            groups.append([item])

    for group in groups:
        joined = " ".join(text for _, text in group)
        if len(joined) >= 20:
            return joined[:250]
    return " ".join(text for _, text in groups[0])[:250]


def corpus_boilerplate(
    conn: sqlite3.Connection, min_documents: int = 3, scan_lines: int = 25
) -> frozenset[str]:
    """Front-matter lines that recur across distinct documents, i.e. boilerplate.

    No rules and no publisher list: a line appearing in many different papers
    cannot be any one paper's title. Improves as the library grows.
    """
    counts: dict[str, int] = {}
    for (text,) in conn.execute(
        "SELECT raw_text FROM paper_text WHERE raw_text IS NOT NULL"
    ):
        seen = set()
        for line in (text or "").split("\n")[:scan_lines]:
            key = normalize_line(line)
            if 4 <= len(key) <= 120:
                seen.add(key)
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    return frozenset(k for k, n in counts.items() if n >= min_documents)


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
        "name": "semantic_scholar",
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
        "name": "openalex",
        "doi_source": "openalex",
        "filename_source": "openalex_filename",
        "title_source": "openalex_title",
        "is_semantic_scholar": False,
    }


def _get_provider(provider: str, rate_limit: float, mailto: str | None) -> dict:
    if provider in ("semantic-scholar", "semantic_scholar", "s2"):
        return _semantic_scholar_provider(rate_limit)
    return _openalex_provider(rate_limit, mailto)


def default_enrich_ids(conn: sqlite3.Connection, retry_misses: bool = False) -> list[int]:
    """IDs default ``enrich`` picks up: never attempted (and not hand-set),
    plus earlier provider misses when ``retry_misses``."""
    where = never_enriched_condition()
    if retry_misses:
        where = f"({where} OR {miss_condition()})"
    return [r[0] for r in conn.execute(f"SELECT id FROM papers WHERE {where} ORDER BY id")]


def enrich_documents(
    conn: sqlite3.Connection,
    paper_ids: list[int] | None = None,
    rate_limit: float = 0.1,
    limit: int | None = None,
    progress_callback=None,
    provider: str = "openalex",
    mailto: str | None = None,
    retry_misses: bool = False,
) -> EnrichStats:
    """Fetch metadata for documents from the chosen provider (OpenAlex default).

    Default selection (``paper_ids=None``) is papers never attempted; with
    ``retry_misses`` it also re-attempts earlier provider misses. Hand-set
    metadata (``meta set`` / legacy hand SQL) is never overwritten, even when
    named in ``paper_ids`` -- clear it with ``meta clear`` first.

    A miss writes ``metadata_source='none:<provider>'``. A lookup that raised
    (network or API error) with no match leaves the paper unstamped so the
    next run retries it.
    """
    stats = EnrichStats()
    start = time.time()
    prov = _get_provider(provider, rate_limit, mailto)

    select = """SELECT p.id, p.filename, p.doi, pt.raw_text,
            CASE WHEN p.metadata_source IS NOT NULL AND {manual}
                 THEN 1 ELSE 0 END AS is_manual
        FROM papers p
        LEFT JOIN paper_text pt ON pt.paper_id = p.id""".format(
        manual=manual_condition().replace("metadata_source", "p.metadata_source")
    )
    if paper_ids is not None:
        placeholders = ",".join("?" * len(paper_ids))
        rows = conn.execute(
            f"{select} WHERE p.id IN ({placeholders}) ORDER BY p.id",
            paper_ids,
        ).fetchall()
    else:
        ids = default_enrich_ids(conn, retry_misses=retry_misses)
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"{select} WHERE p.id IN ({placeholders}) ORDER BY p.id", ids
        ).fetchall() if ids else []

    manual_rows = [r for r in rows if r["is_manual"]]
    stats.skipped_manual = len(manual_rows)
    rows = [r for r in rows if not r["is_manual"]]

    if limit:
        rows = rows[:limit]

    stats.total = len(rows)
    completed = 0
    boilerplate = corpus_boilerplate(conn)

    for row in rows:
        paper_id = row["id"]
        doi = row["doi"]
        filename = row["filename"]
        raw_text = row["raw_text"] or ""

        metadata = None
        source = None
        errors_before = stats.api_errors

        # Try DOI first. A scraped doi is a hypothesis, not a fact: it is only
        # written back below, once a provider lookup on it actually resolved.
        scraped_doi = False
        if not doi and raw_text:
            doi = extract_doi(raw_text)
            scraped_doi = bool(doi)

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
            title_guess = _extract_title_from_text(raw_text, boilerplate)
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
            # Verify EVERY tier, not just title search. A doi lifted from the
            # text is only as trustworthy as the claim that it belongs to this
            # paper, so score the provider's title against the one printed on
            # the PDF and record the verdict rather than asserting confidence.
            # Containment is the primary test (position-free, ~92% recall at
            # 2% false accepts); line-picking similarity is a weaker second
            # opinion that can rescue a title the containment head window cut.
            verify_score = max(
                title_containment(metadata.get("title"), raw_text),
                title_similarity(
                    metadata.get("title"),
                    _extract_title_from_text(raw_text, boilerplate) if raw_text else None,
                ),
            )
            suspect = 1 if verify_score < CONTAINMENT_THRESHOLD else 0
            if suspect and scraped_doi and source == prov["doi_source"]:
                # The doi was a guess AND its answer doesn't match the page:
                # two independent reasons to disbelieve it. Drop the doi so a
                # later pass re-attempts rather than inheriting the bad key.
                metadata = dict(metadata)
                metadata["doi"] = None
                doi = None
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
                    metadata_suspect = ?,
                    metadata_verify_score = ?,
                    metadata_verified_at = ?,
                    updated_at = ?
                WHERE id = ?""",
                (
                    metadata.get("title"),
                    authors,
                    metadata.get("year"),
                    metadata.get("abstract"),
                    ss_id,
                    metadata.get("doi") or (doi if not scraped_doi else None),
                    source,
                    now, suspect, round(verify_score, 3), now, now, paper_id,
                ),
            )
            stats.matched += 1
            if suspect:
                stats.suspect += 1
        elif stats.api_errors > errors_before:
            # A lookup raised: the provider never answered, so this is not
            # evidence of a miss. Leave it unstamped for the next run.
            pass
        else:
            stats.no_match += 1
            # Record the miss explicitly, so "tried and found nothing" is
            # distinguishable from "never tried" and from a real match.
            now = now_iso()
            conn.execute(
                "UPDATE papers SET metadata_source = ?, metadata_enriched_at = ?, "
                "updated_at = ? WHERE id = ?",
                (MISS_SOURCE_PREFIX + prov["name"], now, now, paper_id),
            )

        completed += 1
        if completed % 10 == 0:
            conn.commit()
        if progress_callback:
            progress_callback(completed, stats.total)

    conn.commit()
    stats.elapsed_seconds = round(time.time() - start, 1)
    return stats


# ---------------------------------------------------------------------------
# gantry verify — auditing title-search-sourced metadata matches
#
# enrich_documents() above has three fallback tiers: DOI (exact), filename
# (author-surname + year heuristic), and title search (fuzzy text query,
# top-result-wins). NONE of the three is unambiguous — the doi tier reads the
# doi out of the paper's own text and can pick up a cited work's. Title search
# likewise: querying OpenAlex or Semantic Scholar with a generic or truncated
# title guess (see _extract_title_from_text) can return a plausible-looking
# but *wrong* paper, and nothing before this module ever checked. The wrong
# title/authors/year/DOI then gets written back with full confidence and
# `metadata_enriched_at` set, so the paper never gets re-attempted either.
#
# verify_documents() re-derives the title actually printed on the PDF and
# compares it against what got stored, for every paper whose metadata came
# from a title search. Low-similarity matches are flagged `metadata_suspect`
# in the papers table (queryable via `queue --is metadata-suspect`) so the
# mismatch becomes a fact an agent can find and act on, not a silent one.
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    """The verdict for one title-search-sourced metadata match."""
    paper_id: int
    filename: str
    stored_title: str | None
    extracted_title_guess: str | None
    similarity: float
    suspect: bool


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for comparison."""
    t = title.lower()
    t = _TITLE_NORMALIZE_RE.sub(" ", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


def title_similarity(a: str | None, b: str | None) -> float:
    """Similarity ratio in [0.0, 1.0] between two titles, format-insensitive.

    Returns 0.0 if either title is missing or empty — there's nothing to
    compare, and an absent title should never read as a confident match.
    """
    if not a or not b:
        return 0.0
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def verify_documents(
    conn: sqlite3.Connection,
    paper_ids: list[int] | None = None,
    threshold: float = CONTAINMENT_THRESHOLD,
    limit: int | None = None,
) -> list[VerifyResult]:
    """Audit title-search-sourced metadata against each PDF's own extracted title.

    Scoped to every enriched paper. A doi tier is NOT exact by construction:
    ``extract_doi`` reads the doi out of the PDF's own text, so it is exact
    only if that doi belongs to this paper — and before 2026-09-10 it was
    routinely lifted from the reference list, silently overwriting good
    metadata with a cited work's.

    Persists the verdict to ``metadata_suspect`` / ``metadata_verify_score`` /
    ``metadata_verified_at`` on each checked paper, and returns results
    ordered by ascending similarity so the worst mismatches surface first.
    """
    query = """SELECT p.id, p.filename, p.title, pt.raw_text
        FROM papers p
        LEFT JOIN paper_text pt ON pt.paper_id = p.id
        WHERE {enriched}""".format(
        enriched=enriched_condition().replace("metadata_source", "p.metadata_source")
    )
    params: list = []

    if paper_ids is not None:
        placeholders = ",".join("?" * len(paper_ids))
        query += f" AND p.id IN ({placeholders})"
        params.extend(paper_ids)

    rows = conn.execute(query, params).fetchall()
    if limit:
        rows = rows[:limit]

    results = []
    now = now_iso()
    boilerplate = corpus_boilerplate(conn)
    for row in rows:
        raw_text = row["raw_text"] or ""
        extracted_guess = (
            _extract_title_from_text(raw_text, boilerplate) if raw_text else None
        )
        # Containment first: it asks whether the stored title is present in the
        # document at all, which needs no correct line-pick. Similarity against
        # the extracted line is the fallback opinion.
        similarity = max(
            title_containment(row["title"], raw_text),
            title_similarity(row["title"], extracted_guess),
        )
        suspect = similarity < threshold

        results.append(VerifyResult(
            paper_id=row["id"],
            filename=row["filename"],
            stored_title=row["title"],
            extracted_title_guess=extracted_guess,
            similarity=round(similarity, 4),
            suspect=suspect,
        ))

        conn.execute(
            "UPDATE papers SET metadata_suspect = ?, metadata_verify_score = ?, "
            "metadata_verified_at = ? WHERE id = ?",
            (int(suspect), similarity, now, row["id"]),
        )

    conn.commit()
    results.sort(key=lambda r: r.similarity)
    return results
