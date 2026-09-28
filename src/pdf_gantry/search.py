"""Full-text search (FTS5), semantic search, hybrid search, and chunk-level search."""

import json
import re
import sqlite3
from collections import defaultdict

from .models import ChunkResult, SearchResult
from .utils import parse_authors

_FTS_OPERATORS = {"AND", "OR", "NOT"}
# A balanced double-quoted phrase, or a run of non-space characters.
_FTS_TOKEN_RE = re.compile(r'"[^"]*"\*?|[^\s"]+')
_WORD_CHAR_RE = re.compile(r"\w")


def _legacy_hyphen_sanitize(query: str) -> str:
    """Replace hyphens with spaces (FTS5 reads ``a-b`` as a column filter)."""
    return query.replace("-", " ")


def _quote_fts_term(text: str) -> str:
    """Quote one term as an FTS5 string, escaping embedded double quotes."""
    return '"' + text.replace('"', '""') + '"'


def _sanitize_fts_query(query: str, syntax: bool = False) -> str:
    """
    Turn user text into a safe FTS5 MATCH expression.

    Default (literal) mode treats the text as search terms, never as syntax:

    - each bare word is wrapped in double quotes, so ``:``, ``(``, ``+``,
      ``^``, ``'``, ``-`` and ``NEAR`` lose their FTS5 meaning;
    - balanced ``"quoted phrases"`` are kept as phrases;
    - a trailing ``*`` on a word keeps prefix matching (``adapt*``);
    - ``AND``/``OR``/``NOT`` (upper case) stay operators only when they sit
      between two terms; a dangling or leading operator becomes a literal.

    Adjacent terms are implicitly ANDed, exactly as bare FTS5 terms were, so
    plain queries rank as before. Returns ``""`` when nothing searchable is
    left (callers treat that as no results).

    ``syntax=True`` passes the query through as raw FTS5 syntax, with only the
    historical hyphen-to-space rewrite applied. It can raise
    ``sqlite3.OperationalError`` on malformed input.
    """
    if syntax:
        return _legacy_hyphen_sanitize(query)

    tokens: list[tuple[str, str]] = []  # (kind, text): kind in {"term", "op"}
    for raw in _FTS_TOKEN_RE.findall(query):
        if raw.startswith('"') and raw.rstrip("*").endswith('"') and len(raw.rstrip("*")) >= 2:
            inner = raw.rstrip("*")[1:-1]
            if _WORD_CHAR_RE.search(inner):
                star = "*" if raw.endswith("*") else ""
                tokens.append(("term", _quote_fts_term(inner) + star))
            continue
        if raw in _FTS_OPERATORS:
            tokens.append(("op", raw))
            continue
        prefix = raw.endswith("*")
        word = raw.rstrip("*")
        if not _WORD_CHAR_RE.search(word):
            continue
        tokens.append(("term", _quote_fts_term(word) + ("*" if prefix else "")))

    out: list[str] = []
    for i, (kind, text) in enumerate(tokens):
        if kind == "op":
            prev_is_term = bool(out) and tokens[i - 1][0] == "term"
            next_is_term = i + 1 < len(tokens) and tokens[i + 1][0] == "term"
            if prev_is_term and next_is_term:
                out.append(text)
            else:
                out.append(_quote_fts_term(text))
        else:
            out.append(text)
    return " ".join(out)


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    restrict_ids: list[int] | None = None,
    syntax: bool = False,
    snippets: bool = True,
) -> list[SearchResult]:
    """
    Run FTS5 search and return ranked results with snippets.

    The snippet is the chunk that best matches the query terms (see
    ``attach_snippets``); ``snippets=False`` skips that work.

    ``query`` is literal text by default (see ``_sanitize_fts_query``);
    ``syntax=True`` passes raw FTS5 syntax and may raise
    ``sqlite3.OperationalError``.

    When ``restrict_ids`` is given, the search is scoped to only those paper
    IDs (a cross-paper "which of THESE discuss X?" query). An empty list yields
    no results without touching the DB.
    """
    if restrict_ids is not None and not restrict_ids:
        return []

    safe_query = _sanitize_fts_query(query, syntax=syntax)
    if not safe_query.strip():
        return []
    scope_sql = ""
    params: list = [safe_query]
    if restrict_ids is not None:
        placeholders = ",".join("?" * len(restrict_ids))
        scope_sql = f" AND p.id IN ({placeholders})"
        params.extend(restrict_ids)
    params.append(limit)
    # Contentless FTS5 can't use snippet() — we get snippets from paper_text instead
    rows = conn.execute(
        f"""SELECT
            p.id, p.filename, p.path, p.title,
            p.has_markdown, p.has_embeddings,
            rank
        FROM papers_fts
        JOIN papers p ON p.id = papers_fts.rowid
        WHERE papers_fts MATCH ?{scope_sql}
        ORDER BY rank
        LIMIT ?""",
        params,
    ).fetchall()

    results = []
    for row in rows:
        results.append(SearchResult(
            id=row["id"],
            filename=row["filename"],
            path=row["path"],
            title=row["title"],
            score=abs(row["rank"]),  # FTS5 rank is negative, lower = better
            has_markdown=bool(row["has_markdown"]),
            has_embeddings=bool(row["has_embeddings"]),
        ))

    # Normalize scores: highest = 1.0
    if results:
        max_score = max(r.score for r in results)
        if max_score > 0:
            for r in results:
                r.score = round(r.score / max_score, 2)

    if snippets:
        attach_snippets(conn, results, query=query)
    return results


def search_count(conn: sqlite3.Connection, query: str, syntax: bool = False) -> int:
    """Return the number of FTS5 matches for a query (global, ignores any limit)."""
    safe_query = _sanitize_fts_query(query, syntax=syntax)
    if not safe_query.strip():
        return 0
    row = conn.execute(
        "SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH ?",
        (safe_query,),
    ).fetchone()
    return row[0]


def semantic_search(
    conn: sqlite3.Connection,
    query_vector: bytes,
    limit: int = 20,
    restrict_ids: list[int] | None = None,
    snippets: bool = True,
) -> list[SearchResult]:
    """
    Run vector similarity search using sqlite-vec.

    Snippets come from each paper's best-matching chunk when chunk
    embeddings exist (``snippets=False`` skips that work).

    When ``restrict_ids`` is given, the comparison is scoped to those papers'
    embeddings via ``vec_distance_cosine`` (not global KNN), mirroring
    ``best_chunk_per_doc``. An empty list yields no results.
    """
    if restrict_ids is not None and not restrict_ids:
        return []

    if restrict_ids is not None:
        placeholders = ",".join("?" * len(restrict_ids))
        rows = conn.execute(
            f"""SELECT
                p.id, p.filename, p.path, p.title,
                p.has_markdown, p.has_embeddings,
                vec_distance_cosine(e.embedding, ?) AS distance
            FROM paper_embeddings e
            INNER JOIN papers p ON p.id = e.paper_id
            WHERE p.id IN ({placeholders})
            ORDER BY distance
            LIMIT ?""",
            [query_vector, *restrict_ids, limit],
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT
                p.id, p.filename, p.path, p.title,
                p.has_markdown, p.has_embeddings,
                e.distance
            FROM paper_embeddings e
            INNER JOIN papers p ON p.id = e.paper_id
            WHERE e.embedding MATCH ?
                AND k = ?
            ORDER BY e.distance""",
            (query_vector, limit),
        ).fetchall()

    results = []
    for row in rows:
        # Convert distance to similarity score (1 - distance for cosine)
        score = round(1.0 - row["distance"], 4) if row["distance"] is not None else 0.0
        results.append(SearchResult(
            id=row["id"],
            filename=row["filename"],
            path=row["path"],
            title=row["title"],
            score=score,
            has_markdown=bool(row["has_markdown"]),
            has_embeddings=bool(row["has_embeddings"]),
        ))

    if snippets:
        attach_snippets(conn, results, query_vector=query_vector)
    return results


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    query_vector: bytes,
    limit: int = 20,
    rrf_k: int = 60,
    restrict_ids: list[int] | None = None,
    syntax: bool = False,
    status: dict | None = None,
) -> list[SearchResult]:
    """
    Combine FTS5 and vector search using Reciprocal Rank Fusion (RRF).
    Component scores (FTS rank, vector cosine, ordinal positions) are
    preserved on each result for downstream composition.

    ``restrict_ids`` scopes both components to the given paper set; an empty
    list yields no results.

    If the FTS side raises ``sqlite3.OperationalError`` (only reachable with
    ``syntax=True``), the search continues vector-only. Pass a ``status``
    dict to learn which path ran: ``status["mode"]`` is ``"hybrid"`` or
    ``"vector_fallback"`` (with ``status["fts_error"]`` set).
    """
    if status is None:
        status = {}
    status["mode"] = "hybrid"
    if restrict_ids is not None and not restrict_ids:
        return []

    fetch_limit = limit * 2

    # Get FTS5 results; a malformed raw-syntax query degrades to vector-only.
    try:
        fts_results = fts_search(
            conn, query, limit=fetch_limit, restrict_ids=restrict_ids, syntax=syntax,
            snippets=False,
        )
    except sqlite3.OperationalError as e:
        fts_results = []
        status["mode"] = "vector_fallback"
        status["fts_error"] = str(e)

    # Get semantic results
    sem_results = semantic_search(
        conn, query_vector, limit=fetch_limit, restrict_ids=restrict_ids, snippets=False,
    )

    # Build RRF scores and track component data
    rrf_scores: dict[int, float] = {}
    result_map: dict[int, SearchResult] = {}
    fts_data: dict[int, tuple[float, int]] = {}   # doc_id -> (score, rank)
    vec_data: dict[int, tuple[float, int]] = {}    # doc_id -> (score, rank)

    for rank, r in enumerate(fts_results):
        rrf_scores[r.id] = rrf_scores.get(r.id, 0) + 1.0 / (rrf_k + rank + 1)
        result_map[r.id] = r
        fts_data[r.id] = (r.score, rank + 1)

    for rank, r in enumerate(sem_results):
        rrf_scores[r.id] = rrf_scores.get(r.id, 0) + 1.0 / (rrf_k + rank + 1)
        if r.id not in result_map:
            result_map[r.id] = r
        vec_data[r.id] = (r.score, rank + 1)

    # Sort by combined RRF score
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:limit]

    results = []
    for doc_id in sorted_ids:
        r = result_map[doc_id]
        r.score = round(rrf_scores[doc_id], 4)
        # Populate component scores
        if doc_id in fts_data:
            r.score_fts, r.rank_fts = fts_data[doc_id]
        if doc_id in vec_data:
            r.score_vector, r.rank_vector = vec_data[doc_id]
        results.append(r)

    # Snippets only for the fused top-``limit``: best vector chunk, then FTS chunk.
    attach_snippets(conn, results, query=query, query_vector=query_vector)
    return results


def chunk_search(
    conn: sqlite3.Connection,
    query_vector: bytes,
    limit: int = 100,
) -> list[ChunkResult]:
    """Run KNN search on chunk-level embeddings."""
    rows = conn.execute(
        """SELECT
            cv.chunk_id, cv.distance,
            c.doc_id, c.chunk_index, c.section_header, c.page_start, c.text,
            p.filename, p.path, p.title
        FROM chunk_vec cv
        INNER JOIN chunks c ON c.chunk_id = cv.chunk_id
        INNER JOIN papers p ON p.id = c.doc_id
        WHERE cv.embedding MATCH ?
            AND k = ?
        ORDER BY cv.distance""",
        (query_vector, limit),
    ).fetchall()

    results = []
    for row in rows:
        score = round(1.0 - row["distance"], 4) if row["distance"] is not None else 0.0
        results.append(ChunkResult(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            chunk_index=row["chunk_index"],
            section_header=row["section_header"],
            page_start=row["page_start"],
            chunk_text=row["text"],
            score=score,
            filename=row["filename"],
            path=row["path"],
            title=row["title"],
        ))

    return results


def best_chunk_per_doc(
    conn: sqlite3.Connection,
    query_vector: bytes,
    doc_ids: list[int],
) -> dict[int, ChunkResult]:
    """
    Return the single best-matching chunk for each document in doc_ids.

    Scopes a cosine-distance comparison to only the requested documents'
    chunks (via vec_distance_cosine, not global KNN), so an agent can fetch
    the most relevant excerpt from each of N papers in one call rather than
    looping per paper. Documents without chunk embeddings are omitted.
    """
    if not doc_ids:
        return {}

    placeholders = ",".join("?" * len(doc_ids))
    rows = conn.execute(
        f"""SELECT
            c.doc_id, c.chunk_id, c.chunk_index, c.section_header,
            c.page_start, c.text,
            p.filename, p.path, p.title,
            vec_distance_cosine(cv.embedding, ?) AS distance
        FROM chunks c
        JOIN chunk_vec cv ON cv.chunk_id = c.chunk_id
        JOIN papers p ON p.id = c.doc_id
        WHERE c.doc_id IN ({placeholders})
        ORDER BY c.doc_id, distance""",
        [query_vector, *doc_ids],
    ).fetchall()

    best: dict[int, ChunkResult] = {}
    for row in rows:
        doc_id = row["doc_id"]
        if doc_id in best:
            continue  # rows ordered by distance within each doc; first is closest
        score = round(1.0 - row["distance"], 4) if row["distance"] is not None else 0.0
        best[doc_id] = ChunkResult(
            chunk_id=row["chunk_id"],
            doc_id=doc_id,
            chunk_index=row["chunk_index"],
            section_header=row["section_header"],
            page_start=row["page_start"],
            chunk_text=row["text"],
            score=score,
            filename=row["filename"],
            path=row["path"],
            title=row["title"],
        )
    return best


def _top_k_pool(scores: list[float], k: int = 3) -> float:
    """Average of top-k scores for document aggregation."""
    top_k = sorted(scores, reverse=True)[:k]
    return sum(top_k) / len(top_k) if top_k else 0.0


def cascade_search(
    conn: sqlite3.Connection,
    query_vector: bytes,
    limit: int = 10,
    doc_candidates: int = 50,
    chunk_candidates: int = 100,
    boost: float = 0.05,
    pool_k: int = 3,
) -> list[SearchResult]:
    """
    Two-stage cascade: doc-level filtering + chunk-level retrieval with top-k pooling.

    1. Top-N docs from paper_embeddings (coarse filter)
    2. Top-M chunks from chunk_vec (fine-grained retrieval)
    3. Boost chunk scores for docs that appeared in step 1
    4. Group by document, top-k mean pooling
    5. Return documents ranked by aggregated score with best chunk as excerpt
    """
    # Stage 1: doc-level candidates (snippets are only needed on fallback)
    doc_results = semantic_search(conn, query_vector, limit=doc_candidates, snippets=False)
    doc_ids_in_top = {r.id for r in doc_results}

    # Stage 2: chunk-level search
    chunk_results = chunk_search(conn, query_vector, limit=chunk_candidates)

    if not chunk_results:
        # Fall back to doc-level results if no chunks
        return attach_snippets(conn, doc_results[:limit], query_vector=query_vector)

    # Group chunks by document, apply boost
    doc_chunks: dict[int, list[ChunkResult]] = defaultdict(list)
    for cr in chunk_results:
        if cr.doc_id in doc_ids_in_top:
            cr.score = min(cr.score + boost, 1.0)
        doc_chunks[cr.doc_id].append(cr)

    # Top-k pooling per document
    doc_scores: list[tuple[int, float, ChunkResult]] = []
    for doc_id, chunks in doc_chunks.items():
        scores = [c.score for c in chunks]
        agg_score = _top_k_pool(scores, k=pool_k)
        best_chunk = max(chunks, key=lambda c: c.score)
        doc_scores.append((doc_id, agg_score, best_chunk))

    doc_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score, best_chunk in doc_scores[:limit]:
        results.append(SearchResult(
            id=doc_id,
            filename=best_chunk.filename,
            path=best_chunk.path,
            title=best_chunk.title,
            score=round(score, 4),
            snippet=_trim_snippet(best_chunk.chunk_text, []),
            has_markdown=True,
            has_embeddings=True,
        ))

    return results


_FIND_FIELDS = ("title", "authors", "citekey", "year", "doi", "filename")
_FIND_STOPWORDS = {"&", "and", "et", "al", "etal", "+"}
_EDGE_PUNCT = " \t.,;:()[]{}\"'`?!"
_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                 "http://dx.doi.org/", "doi.org/", "doi:")


def _fold(text: str) -> str:
    """Case- and accent-insensitive form for matching ("Ondřej" -> "ondrej")."""
    import unicodedata
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def _find_tokens(query: str) -> list[str]:
    """Split a find query into folded tokens, dropping author-list glue
    ("&", "and", "et al.") and edge punctuation, and unwrapping DOI URLs."""
    tokens = []
    for raw in query.split():
        tok = raw
        low = tok.lower()
        for prefix in _DOI_PREFIXES:
            if low.startswith(prefix):
                tok = tok[len(prefix):]
                break
        tok = tok.strip(_EDGE_PUNCT)
        if tok.startswith("@"):
            tok = tok[1:]
        tok = _fold(tok)
        if tok:
            tokens.append(tok)
    content = [t for t in tokens if t not in _FIND_STOPWORDS]
    return content or tokens


def _authors_text(authors) -> str:
    """Authors column as searchable text: JSON lists are decoded (so \\u
    escapes become real characters); anything else is used verbatim."""
    if not authors:
        return ""
    try:
        parsed = json.loads(authors)
    except (ValueError, TypeError):
        return str(authors)
    if isinstance(parsed, list):
        return "; ".join(str(a) for a in parsed)
    return str(parsed)


def find_papers(
    conn: sqlite3.Connection,
    fragment: str,
    limit: int = 20,
) -> list[dict]:
    """Known-item lookup over filename, title, authors, citekey, DOI and year.

    Case- and accent-insensitive. The query is split into tokens and every
    token must match at least one field (a 4-digit token also matches
    ``year`` exactly), so "Zhang 2022", "Flew & Martin" and "Mind games"
    all work. Connective tokens ("&", "and", "et al.") are ignored.

    Ranking: exact/phrase title matches first, then all tokens in the title,
    then all tokens in bibliographic fields (authors/citekey/year), then
    matches that needed the filename or DOI. Each result carries
    ``matched_fields``, the fields at least one token matched, in the order
    title, authors, citekey, year, doi, filename.

    Pure-DB and done in Python over the papers table (a few thousand short
    rows), which gives Unicode-aware folding that SQLite's LIKE lacks.
    """
    tokens = _find_tokens(fragment)
    if not tokens:
        return []
    phrase = " ".join(_fold(fragment).split())

    rows = conn.execute(
        """SELECT id, filename, path, title, authors, year, doi, page_count,
                  has_text, has_markdown, has_embeddings, has_chunk_embeddings,
                  is_scanned, citekey
        FROM papers"""
    ).fetchall()

    scored = []
    for r in rows:
        fields = {
            "title": _fold(r["title"] or ""),
            "authors": _fold(_authors_text(r["authors"])),
            "citekey": _fold(r["citekey"] or ""),
            "doi": _fold(r["doi"] or ""),
            "filename": _fold(r["filename"] or ""),
        }
        year = str(r["year"]) if r["year"] is not None else ""
        matched: set[str] = set()
        per_token: list[set[str]] = []
        for tok in tokens:
            hit = {name for name, val in fields.items() if tok in val}
            if year and tok == year:
                hit.add("year")
            if not hit:
                break
            per_token.append(hit)
            matched |= hit
        else:
            title = fields["title"]
            if title and (title == phrase or phrase in title):
                tier = 0
            elif all("title" in h for h in per_token):
                tier = 1
            elif all(h & {"title", "authors", "citekey", "year"} for h in per_token):
                tier = 2
            else:
                tier = 3
            d = dict(r)
            d["authors"] = parse_authors(r["authors"])
            d["matched_fields"] = [f for f in _FIND_FIELDS if f in matched]
            scored.append((tier, r["filename"] or "", r["id"], d))

    scored.sort(key=lambda x: x[:3])
    return [d for *_, d in scored[:limit]]


def get_chunk_context(
    conn: sqlite3.Connection,
    chunk_id: int,
    max_chars: int = 2000,
) -> dict | None:
    """
    Get a chunk with surrounding context from neighboring chunks.

    Returns a dict with the target chunk text, expanded context from
    neighbors, and document metadata. Returns None if chunk_id not found.
    """
    # Get the target chunk with document info
    row = conn.execute(
        """SELECT c.chunk_id, c.doc_id, c.chunk_index, c.section_header,
                  c.page_start, c.text,
                  p.filename, p.path, p.title
        FROM chunks c
        JOIN papers p ON p.id = c.doc_id
        WHERE c.chunk_id = ?""",
        (chunk_id,),
    ).fetchone()

    if not row:
        return None

    doc_id = row["doc_id"]
    target_index = row["chunk_index"]
    target_text = row["text"]

    # Count total chunks for this document
    total_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0]

    # Get all chunks for this document, ordered
    all_chunks = conn.execute(
        "SELECT chunk_index, text FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()

    # Build context by expanding outward from the target chunk
    context_parts = [target_text]
    chars_used = len(target_text)

    # Expand outward: alternate before and after
    before_idx = target_index - 1
    after_idx = target_index + 1
    chunk_map = {c["chunk_index"]: c["text"] for c in all_chunks}

    while chars_used < max_chars:
        added = False

        if before_idx >= 0 and before_idx in chunk_map:
            text = chunk_map[before_idx]
            if chars_used + len(text) <= max_chars + 200:  # Allow slight overshoot
                context_parts.insert(0, text)
                chars_used += len(text)
                before_idx -= 1
                added = True

        if after_idx in chunk_map:
            text = chunk_map[after_idx]
            if chars_used + len(text) <= max_chars + 200:
                context_parts.append(text)
                chars_used += len(text)
                after_idx += 1
                added = True

        if not added:
            break

    return {
        "chunk_id": row["chunk_id"],
        "doc_id": doc_id,
        "chunk_index": target_index,
        "section_header": row["section_header"],
        "page_start": row["page_start"],
        "chunk_text": target_text,
        "context": "\n\n".join(context_parts),
        "filename": row["filename"],
        "path": row["path"],
        "title": row["title"],
        "total_chunks": total_chunks,
    }


# --- snippets ---------------------------------------------------------------
#
# A result's snippet should show *why* it matched. The first 200 characters
# of raw_text are usually a masthead or ISSN line, so instead:
#   1. best vector chunk (semantic/hybrid, when chunk embeddings exist),
#   2. else the chunk containing the most query terms (FTS),
#   3. else a window of raw_text around the first query term,
#   4. else the start of raw_text.
# All of this reads the DB only; PDFs are never opened on the read path.

SNIPPET_WIDTH = 300
_TERM_RE = re.compile(r"\w+")


def _query_terms(query: str | None) -> list[str]:
    """Lower-cased search words from a query, minus FTS operators."""
    if not query:
        return []
    terms = []
    for word in _TERM_RE.findall(query):
        if word in _FTS_OPERATORS or len(word) < 2:
            continue
        w = word.lower()
        if w not in terms:
            terms.append(w)
    return terms


def _match_stem(term: str) -> str:
    """Crude stem so substring matching tolerates FTS5's porter stemming."""
    return term if len(term) <= 5 else term[:-2]


def _trim_snippet(text: str, terms: list[str], width: int = SNIPPET_WIDTH) -> str:
    """Collapse whitespace and cut ``width`` chars around the first term hit.

    Cuts snap to word boundaries; "..." marks each truncated side.
    """
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    lower = flat.lower()
    positions = [lower.find(_match_stem(t)) for t in terms]
    positions = [p for p in positions if p >= 0]
    start = max(0, min(positions) - width // 5) if positions else 0
    if start > 0:
        space = flat.find(" ", start)
        start = space + 1 if 0 <= space < start + 30 else start
    end = start + width
    if end < len(flat):
        space = flat.rfind(" ", start, end)
        end = space if space > start + width // 2 else end
    body = flat[start:end].strip()
    return ("..." if start > 0 else "") + body + ("..." if end < len(flat) else "")


def _fts_best_chunks(
    conn: sqlite3.Connection, doc_ids: list[int], terms: list[str],
) -> dict[int, str]:
    """For each doc, the chunk containing the most distinct query terms."""
    if not doc_ids or not terms:
        return {}
    hit_sql = " + ".join("(instr(lower(text), ?) > 0)" for _ in terms)
    placeholders = ",".join("?" * len(doc_ids))
    rows = conn.execute(
        f"""SELECT doc_id, text FROM (
                SELECT doc_id, text, hits, ROW_NUMBER() OVER (
                    PARTITION BY doc_id ORDER BY hits DESC, chunk_index
                ) AS rn
                FROM (SELECT doc_id, chunk_index, text, ({hit_sql}) AS hits
                      FROM chunks WHERE doc_id IN ({placeholders}))
                WHERE hits > 0
            ) WHERE rn = 1""",
        [*(_match_stem(t) for t in terms), *doc_ids],
    ).fetchall()
    return {row["doc_id"]: row["text"] for row in rows}


def _raw_text_snippets(
    conn: sqlite3.Connection, doc_ids: list[int], terms: list[str],
    width: int = SNIPPET_WIDTH,
) -> dict[int, str]:
    """A window of raw_text around the first query term (or its start)."""
    if not doc_ids:
        return {}
    stems = [_match_stem(t) for t in terms]
    pos_sql = "".join(f", instr(lower(raw_text), ?) AS p{i}" for i in range(len(stems)))
    placeholders = ",".join("?" * len(doc_ids))
    rows = conn.execute(
        f"SELECT paper_id{pos_sql} FROM paper_text WHERE paper_id IN ({placeholders})",
        [*stems, *doc_ids],
    ).fetchall()
    out: dict[int, str] = {}
    for row in rows:
        hits = [row[f"p{i}"] for i in range(len(stems)) if row[f"p{i}"]]
        start = max(1, min(hits) - width) if hits else 1
        text = conn.execute(
            "SELECT SUBSTR(raw_text, ?, ?) FROM paper_text WHERE paper_id = ?",
            (start, width * 3, row["paper_id"]),
        ).fetchone()[0]
        if text:
            trimmed = _trim_snippet(text, terms, width)
            if start > 1 and not trimmed.startswith("..."):
                trimmed = "..." + trimmed
            out[row["paper_id"]] = trimmed
    return out


def attach_snippets(
    conn: sqlite3.Connection,
    results: list[SearchResult],
    query: str | None = None,
    query_vector: bytes | None = None,
    width: int = SNIPPET_WIDTH,
) -> list[SearchResult]:
    """Set each result's ``snippet`` to its most relevant stored text."""
    if not results:
        return results
    terms = _query_terms(query)
    pending = {r.id for r in results}
    chosen: dict[int, str] = {}

    if query_vector is not None:
        for doc_id, chunk in best_chunk_per_doc(conn, query_vector, list(pending)).items():
            chosen[doc_id] = _trim_snippet(chunk.chunk_text, terms, width)
        pending -= chosen.keys()
    if pending and terms:
        for doc_id, text in _fts_best_chunks(conn, list(pending), terms).items():
            chosen[doc_id] = _trim_snippet(text, terms, width)
        pending -= chosen.keys()
    if pending:
        chosen.update(_raw_text_snippets(conn, list(pending), terms, width))

    for r in results:
        r.snippet = chosen.get(r.id, "")
    return results
