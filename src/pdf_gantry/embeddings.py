"""Embedding generation and storage."""

import sqlite3
import struct
import time
from pathlib import Path

from .models import ProcessStats
from .utils import now_iso


def _get_embedding_model(model_name: str):
    """Load the sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. Run: pip install pdf-gantry[embeddings]"
        )
    return SentenceTransformer(model_name, trust_remote_code=True)


def _serialize_vector(vector) -> bytes:
    """Serialize a float vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def _prepare_text(title: str | None, abstract: str | None, raw_text: str | None, max_chars: int = 30000) -> str:
    """Prepare text for embedding: title + abstract + truncated text."""
    parts = []
    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)
    if raw_text:
        remaining = max_chars - sum(len(p) for p in parts)
        if remaining > 0:
            parts.append(raw_text[:remaining])
    return "\n\n".join(parts) if parts else ""


def embed_documents(
    conn: sqlite3.Connection,
    db_path: Path,
    paper_ids: list[int] | None = None,
    model_name: str = "nomic-ai/nomic-embed-text-v2-moe",
    dimensions: int = 768,
    batch_size: int = 32,
    limit: int | None = None,
    progress_callback=None,
) -> ProcessStats:
    """Generate embeddings for documents."""
    stats = ProcessStats()
    start = time.time()

    # Get documents to embed
    if paper_ids is not None:
        placeholders = ",".join("?" * len(paper_ids))
        rows = conn.execute(
            f"""SELECT p.id, p.title, p.abstract, pt.raw_text
            FROM papers p
            JOIN paper_text pt ON pt.paper_id = p.id
            WHERE p.id IN ({placeholders})""",
            paper_ids,
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT p.id, p.title, p.abstract, pt.raw_text
            FROM papers p
            JOIN paper_text pt ON pt.paper_id = p.id
            WHERE p.has_text = 1 AND p.has_embeddings = 0"""
        ).fetchall()

    if limit:
        rows = rows[:limit]

    stats.total = len(rows)
    if stats.total == 0:
        return stats

    # Load model
    model = _get_embedding_model(model_name)
    model_version = model_name.split("/")[-1] if "/" in model_name else model_name

    # Process in batches
    completed = 0
    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]

        texts = [
            _prepare_text(row["title"], row["abstract"], row["raw_text"])
            for row in batch
        ]

        # Prefix for Nomic models
        prefixed = [f"search_document: {t}" for t in texts]

        try:
            vectors = model.encode(prefixed, show_progress_bar=False)
        except Exception as e:
            for row in batch:
                stats.failed += 1
                conn.execute(
                    """UPDATE papers SET
                        last_error = ?, error_count = error_count + 1,
                        last_error_at = ?, updated_at = ?
                    WHERE id = ?""",
                    (str(e), now_iso(), now_iso(), row["id"]),
                )
            conn.commit()
            completed += len(batch)
            if progress_callback:
                progress_callback(completed, stats.total)
            continue

        now = now_iso()
        for i, row in enumerate(batch):
            try:
                vec_bytes = _serialize_vector(vectors[i])
                conn.execute(
                    "INSERT OR REPLACE INTO paper_embeddings (paper_id, embedding) VALUES (?, ?)",
                    (row["id"], vec_bytes),
                )
                conn.execute(
                    """UPDATE papers SET
                        has_embeddings = 1,
                        embedding_model = ?,
                        embedding_model_version = ?,
                        embedding_computed_at = ?,
                        updated_at = ?
                    WHERE id = ?""",
                    (model_name, model_version, now, now, row["id"]),
                )
                stats.succeeded += 1
            except Exception as e:
                stats.failed += 1
                conn.execute(
                    """UPDATE papers SET
                        last_error = ?, error_count = error_count + 1,
                        last_error_at = ?, updated_at = ?
                    WHERE id = ?""",
                    (str(e), now_iso(), now_iso(), row["id"]),
                )

        conn.commit()
        completed += len(batch)
        if progress_callback:
            progress_callback(completed, stats.total)

    stats.elapsed_seconds = round(time.time() - start, 1)
    return stats


def embed_chunks(
    conn: sqlite3.Connection,
    db_path: Path,
    paper_ids: list[int] | None = None,
    model_name: str = "nomic-ai/nomic-embed-text-v2-moe",
    dimensions: int = 768,
    batch_size: int = 64,
    limit: int | None = None,
    progress_callback=None,
) -> ProcessStats:
    """Generate chunk-level embeddings with contextual metadata prepended."""
    from .chunking import prepare_chunk_text

    stats = ProcessStats()
    start = time.time()

    # Get papers that need chunk embeddings
    if paper_ids is not None:
        placeholders = ",".join("?" * len(paper_ids))
        paper_rows = conn.execute(
            f"SELECT id, title FROM papers WHERE id IN ({placeholders}) AND has_text = 1",
            paper_ids,
        ).fetchall()
    else:
        paper_rows = conn.execute(
            "SELECT id, title FROM papers WHERE has_text = 1 AND has_chunk_embeddings = 0"
        ).fetchall()

    if limit:
        paper_rows = paper_rows[:limit]

    if not paper_rows:
        return stats

    # Collect all chunks across selected papers
    all_chunks = []
    paper_titles = {}
    for paper in paper_rows:
        paper_titles[paper["id"]] = paper["title"]
        chunks = conn.execute(
            "SELECT chunk_id, doc_id, chunk_index, section_header, text FROM chunks WHERE doc_id = ?",
            (paper["id"],),
        ).fetchall()
        all_chunks.extend(chunks)

    stats.total = len(all_chunks)
    if stats.total == 0:
        return stats

    # Load model
    model = _get_embedding_model(model_name)
    model_version = model_name.split("/")[-1] if "/" in model_name else model_name

    # Track which papers have all chunks embedded successfully
    paper_chunk_counts: dict[int, int] = {}
    paper_success_counts: dict[int, int] = {}
    for chunk in all_chunks:
        doc_id = chunk["doc_id"]
        paper_chunk_counts[doc_id] = paper_chunk_counts.get(doc_id, 0) + 1
        paper_success_counts.setdefault(doc_id, 0)

    # Process in batches
    completed = 0
    for batch_start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[batch_start:batch_start + batch_size]

        texts = [
            f"search_document: {prepare_chunk_text(paper_titles.get(c['doc_id']), c['section_header'], c['text'])}"
            for c in batch
        ]

        try:
            vectors = model.encode(texts, show_progress_bar=False)
        except Exception as e:
            stats.failed += len(batch)
            completed += len(batch)
            if progress_callback:
                progress_callback(completed, stats.total)
            continue

        for i, chunk in enumerate(batch):
            try:
                vec_bytes = _serialize_vector(vectors[i])
                conn.execute(
                    "INSERT OR REPLACE INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                    (chunk["chunk_id"], vec_bytes),
                )
                stats.succeeded += 1
                paper_success_counts[chunk["doc_id"]] += 1
            except Exception as e:
                stats.failed += 1

        conn.commit()
        completed += len(batch)
        if progress_callback:
            progress_callback(completed, stats.total)

    # Update paper flags for papers where all chunks succeeded
    now = now_iso()
    for doc_id, total in paper_chunk_counts.items():
        if paper_success_counts.get(doc_id, 0) == total:
            conn.execute(
                """UPDATE papers SET
                    has_chunk_embeddings = 1,
                    embedding_model = ?,
                    embedding_model_version = ?,
                    embedding_computed_at = ?,
                    updated_at = ?
                WHERE id = ?""",
                (model_name, model_version, now, now, doc_id),
            )
    conn.commit()

    stats.elapsed_seconds = round(time.time() - start, 1)
    return stats


def embed_query(model_name: str, query: str) -> bytes:
    """Embed a query string and return serialized vector."""
    model = _get_embedding_model(model_name)
    vector = model.encode(f"search_query: {query}", show_progress_bar=False)
    return _serialize_vector(vector)
