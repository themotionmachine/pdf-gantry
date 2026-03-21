"""Full-text search (FTS5), semantic search, hybrid search, and chunk-level search."""

import sqlite3
from collections import defaultdict

from .models import ChunkResult, SearchResult


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
