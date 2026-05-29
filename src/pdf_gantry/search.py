"""Full-text search (FTS5), semantic search, hybrid search, and chunk-level search."""

import re
import sqlite3
from collections import defaultdict

from .models import ChunkResult, SearchResult


def _sanitize_fts_query(query: str) -> str:
    """
    Sanitize a query string for FTS5.

    Handles hyphenated terms (e.g. "cross-national") which FTS5 misparses
    as column filters. Replaces hyphens with spaces except inside quoted phrases.
    """
    # Split on quoted sections to preserve them
    parts = re.split(r'(".*?")', query)
    sanitized = []
    for part in parts:
        if part.startswith('"') and part.endswith('"'):
            # Inside quotes: replace hyphens with spaces but keep quotes
            sanitized.append(part.replace('-', ' '))
        else:
            # Outside quotes: replace hyphens with spaces
            sanitized.append(part.replace('-', ' '))
    return "".join(sanitized)


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """Run FTS5 search and return ranked results with snippets."""
    safe_query = _sanitize_fts_query(query)
    # Contentless FTS5 can't use snippet() — we get snippets from paper_text instead
    rows = conn.execute(
        """SELECT
            p.id, p.filename, p.path, p.title,
            p.has_markdown, p.has_embeddings,
            rank,
            SUBSTR(pt.raw_text, 1, 200) as text_preview
        FROM papers_fts
        JOIN papers p ON p.id = papers_fts.rowid
        LEFT JOIN paper_text pt ON pt.paper_id = p.id
        WHERE papers_fts MATCH ?
        ORDER BY rank
        LIMIT ?""",
        (safe_query, limit),
    ).fetchall()

    results = []
    for row in rows:
        results.append(SearchResult(
            id=row["id"],
            filename=row["filename"],
            path=row["path"],
            title=row["title"],
            score=abs(row["rank"]),  # FTS5 rank is negative, lower = better
            snippet=row["text_preview"] or "",
            has_markdown=bool(row["has_markdown"]),
            has_embeddings=bool(row["has_embeddings"]),
        ))

    # Normalize scores: highest = 1.0
    if results:
        max_score = max(r.score for r in results)
        if max_score > 0:
            for r in results:
                r.score = round(r.score / max_score, 2)

    return results


def search_count(conn: sqlite3.Connection, query: str) -> int:
    """Return the number of FTS5 matches for a query."""
    safe_query = _sanitize_fts_query(query)
    row = conn.execute(
        "SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH ?",
        (safe_query,),
    ).fetchone()
    return row[0]


def semantic_search(
    conn: sqlite3.Connection,
    query_vector: bytes,
    limit: int = 20,
) -> list[SearchResult]:
    """Run vector similarity search using sqlite-vec."""
    rows = conn.execute(
        """SELECT
            p.id, p.filename, p.path, p.title,
            p.has_markdown, p.has_embeddings,
            e.distance,
            SUBSTR(pt.raw_text, 1, 200) as text_preview
        FROM paper_embeddings e
        INNER JOIN papers p ON p.id = e.paper_id
        LEFT JOIN paper_text pt ON pt.paper_id = p.id
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
            snippet=row["text_preview"] or "",
            has_markdown=bool(row["has_markdown"]),
            has_embeddings=bool(row["has_embeddings"]),
        ))

    return results


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    query_vector: bytes,
    limit: int = 20,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """
    Combine FTS5 and vector search using Reciprocal Rank Fusion (RRF).
    Component scores (FTS rank, vector cosine, ordinal positions) are
    preserved on each result for downstream composition.
    """
    fetch_limit = limit * 2

    # Get FTS5 results
    fts_results = fts_search(conn, query, limit=fetch_limit)

    # Get semantic results
    sem_results = semantic_search(conn, query_vector, limit=fetch_limit)

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
    # Stage 1: doc-level candidates
    doc_results = semantic_search(conn, query_vector, limit=doc_candidates)
    doc_ids_in_top = {r.id for r in doc_results}

    # Stage 2: chunk-level search
    chunk_results = chunk_search(conn, query_vector, limit=chunk_candidates)

    if not chunk_results:
        # Fall back to doc-level results if no chunks
        return doc_results[:limit]

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
            snippet=best_chunk.chunk_text[:300],
            has_markdown=True,
            has_embeddings=True,
        ))

    return results


def find_papers(
    conn: sqlite3.Connection,
    fragment: str,
    limit: int = 20,
) -> list[dict]:
    """Fuzzy filename lookup using LIKE matching (case-insensitive)."""
    rows = conn.execute(
        """SELECT id, filename, path, title, page_count, has_text, has_markdown,
                  has_embeddings, has_chunk_embeddings, is_scanned, citekey
        FROM papers
        WHERE filename LIKE ?
        ORDER BY filename
        LIMIT ?""",
        (f"%{fragment}%", limit),
    ).fetchall()
    return [dict(r) for r in rows]


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
