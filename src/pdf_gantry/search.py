"""Full-text search (FTS5), semantic search, and hybrid search."""

import sqlite3

from .models import SearchResult


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """Run FTS5 search and return ranked results with snippets."""
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
        (query, limit),
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
    row = conn.execute(
        "SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH ?",
        (query,),
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
    """
    fetch_limit = limit * 2

    # Get FTS5 results
    fts_results = fts_search(conn, query, limit=fetch_limit)

    # Get semantic results
    sem_results = semantic_search(conn, query_vector, limit=fetch_limit)

    # Build RRF scores
    rrf_scores: dict[int, float] = {}
    result_map: dict[int, SearchResult] = {}

    for rank, r in enumerate(fts_results):
        rrf_scores[r.id] = rrf_scores.get(r.id, 0) + 1.0 / (rrf_k + rank + 1)
        result_map[r.id] = r

    for rank, r in enumerate(sem_results):
        rrf_scores[r.id] = rrf_scores.get(r.id, 0) + 1.0 / (rrf_k + rank + 1)
        if r.id not in result_map:
            result_map[r.id] = r

    # Sort by combined RRF score
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:limit]

    results = []
    for doc_id in sorted_ids:
        r = result_map[doc_id]
        r.score = round(rrf_scores[doc_id], 4)
        results.append(r)

    return results
