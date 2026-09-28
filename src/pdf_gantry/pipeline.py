"""Single-command pipeline: ingest → process → [enrich →] embed."""

import sqlite3
from pathlib import Path

from .ingest import ingest_directory
from .process import process_documents
from .queue import DEFAULT_MAX_RETRIES, not_quarantined_condition


def run_pipeline(
    conn: sqlite3.Connection,
    papers_dir: Path,
    db_path: Path,
    workers: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    filename: str | None = None,
    scan_threshold: float = 0.05,
    progress_callback=None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    enrich: bool = False,
    provider: str = "openalex",
    mailto: str | None = None,
) -> dict:
    """
    Run the full ingestion pipeline: ingest → process → [enrich →] embed (chunks).

    If filename is given, only that file is processed end-to-end.
    Limit applies to total pipeline throughput, not per-stage.

    With ``enrich``, metadata is fetched for never-enriched papers after
    extraction (enrich needs the text) and BEFORE embedding: chunk text is
    prefixed with the paper title (chunking.prepare_chunk_text), so a paper
    embedded before enrichment carries no title in its vectors. Papers that
    gain a title here but were already chunk-embedded are listed in
    ``titled_after_embed`` for ``embed --chunk --ids ... --force``.

    Returns stats dict with counts per stage.
    """
    stats = {
        "ingested": 0,
        "processed": 0,
        "enriched": 0,
        "embedded": 0,
        "errors": 0,
        "new_files": [],
    }

    # --- Stage 1: Ingest ---
    if filename:
        # Single-file mode: ingest the whole dir (fast — skips existing)
        # but only operate on the named file afterward
        file_path = papers_dir / filename
        if not file_path.exists():
            return {"error": f"File not found: {filename}", **stats}

        # Check if already in DB
        existing = conn.execute(
            "SELECT id FROM papers WHERE filename = ?", (filename,)
        ).fetchone()

        if not existing:
            # Ingest to register the file
            ingest_stats = ingest_directory(
                conn, papers_dir, dry_run=dry_run, scan_threshold=scan_threshold,
            )
            stats["ingested"] = 1 if ingest_stats.new > 0 else 0
        else:
            stats["ingested"] = 0

        if dry_run:
            stats["ingested"] = 0 if existing else 1
            return stats

        row = conn.execute(
            "SELECT id FROM papers WHERE filename = ?", (filename,)
        ).fetchone()
        if not row:
            return stats
        target_ids = [row["id"]]
        stats["ids"] = target_ids
    else:
        ingest_stats = ingest_directory(
            conn, papers_dir,
            dry_run=dry_run,
            scan_threshold=scan_threshold,
            progress_callback=progress_callback,
        )
        stats["ingested"] = ingest_stats.new + ingest_stats.changed

        if dry_run:
            return stats

        target_ids = None  # Process all unprocessed

    # --- Stage 2: Process ---
    if filename and target_ids:
        # Check if this file needs processing
        row = conn.execute(
            "SELECT has_text FROM papers WHERE id = ?", (target_ids[0],)
        ).fetchone()
        # A changed file is reset to has_text=0 by ingest, so has_text alone
        # decides whether this paper needs (re-)processing.
        target_ids_to_process = [] if row and row["has_text"] else target_ids

        if target_ids_to_process:
            proc_stats = process_documents(
                conn, papers_dir, db_path,
                paper_ids=target_ids_to_process,
                workers=workers,
                limit=limit,
                progress_callback=progress_callback,
            )
            stats["processed"] = proc_stats.succeeded
            stats["errors"] += proc_stats.failed
    else:
        # Get unprocessed papers (skip quarantined — see queue.quarantine_condition
        # — and encrypted papers, which classify_document() reports as "digital"
        # but can never actually be extracted; see process.py's identical filter)
        rows = conn.execute(
            "SELECT id FROM papers WHERE has_text = 0 "
            "AND (is_scanned = 0 OR is_scanned IS NULL) "
            "AND is_encrypted = 0 "
            f"AND {not_quarantined_condition(max_retries)}"
        ).fetchall()
        ids_to_process = [r["id"] for r in rows]

        if limit:
            ids_to_process = ids_to_process[:limit]

        if ids_to_process:
            proc_stats = process_documents(
                conn, papers_dir, db_path,
                paper_ids=ids_to_process,
                workers=workers,
                progress_callback=progress_callback,
            )
            stats["processed"] = proc_stats.succeeded
            stats["errors"] += proc_stats.failed

    # --- Stage 2b: Enrich (before embed, so titles reach the vectors) ---
    if enrich:
        _enrich_stage(conn, stats, target_ids if filename else None,
                      limit=limit, provider=provider, mailto=mailto)

    # --- Stage 3: Embed chunks ---
    # Skip embedding if sentence-transformers isn't installed
    try:
        from .embeddings import embed_chunks
    except ImportError:
        return stats

    if filename and target_ids:
        ids_to_embed = target_ids
    else:
        rows = conn.execute(
            "SELECT id FROM papers WHERE has_text = 1 AND has_chunk_embeddings = 0 "
            f"AND {not_quarantined_condition(max_retries)}"
        ).fetchall()
        ids_to_embed = [r["id"] for r in rows]
        if limit:
            ids_to_embed = ids_to_embed[:limit]

    if ids_to_embed:
        try:
            embed_stats = embed_chunks(
                conn, db_path,
                paper_ids=ids_to_embed,
                progress_callback=progress_callback,
            )
            stats["embedded"] = embed_stats.succeeded
            stats["errors"] += embed_stats.failed
        except ImportError:
            pass  # sentence-transformers not installed

    return stats


def _enrich_stage(
    conn: sqlite3.Connection,
    stats: dict,
    target_ids: list[int] | None,
    limit: int | None,
    provider: str,
    mailto: str | None,
) -> None:
    """Enrich never-attempted papers (the target file only, in file mode)."""
    from .metadata import default_enrich_ids, enrich_documents

    eligible = default_enrich_ids(conn)
    if target_ids is not None:
        wanted = set(target_ids)
        eligible = [i for i in eligible if i in wanted]
    if limit:
        eligible = eligible[:limit]

    untitled_embedded: set[int] = set()
    if eligible:
        placeholders = ",".join("?" * len(eligible))
        untitled_embedded = {r[0] for r in conn.execute(
            f"SELECT id FROM papers WHERE id IN ({placeholders}) "
            "AND (title IS NULL OR title = '') AND has_chunk_embeddings = 1",
            eligible,
        )}

    es = enrich_documents(conn, paper_ids=eligible, provider=provider, mailto=mailto) \
        if eligible else None
    stats["enriched"] = es.matched if es else 0
    stats["enrich"] = {
        "total": es.total if es else 0,
        "matched": es.matched if es else 0,
        "no_match": es.no_match if es else 0,
        "api_errors": es.api_errors if es else 0,
        "suspect": es.suspect if es else 0,
        "skipped_manual": es.skipped_manual if es else 0,
    }

    titled: list[int] = []
    if untitled_embedded:
        placeholders = ",".join("?" * len(untitled_embedded))
        titled = sorted(r[0] for r in conn.execute(
            f"SELECT id FROM papers WHERE id IN ({placeholders}) "
            "AND title IS NOT NULL AND title != ''",
            sorted(untitled_embedded),
        ))
    stats["titled_after_embed"] = titled
