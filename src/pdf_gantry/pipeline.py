"""Single-command pipeline: ingest → process → embed."""

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
) -> dict:
    """
    Run the full ingestion pipeline: ingest → process → embed (chunks).

    If filename is given, only that file is processed end-to-end.
    Limit applies to total pipeline throughput, not per-stage.
    Returns stats dict with counts per stage.
    """
    stats = {
        "ingested": 0,
        "processed": 0,
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
        if row and row["has_text"]:
            # Already processed — check if it needs re-processing (changed file)
            if ingest_stats.changed == 0:
                target_ids_to_process = []
            else:
                target_ids_to_process = target_ids
        else:
            target_ids_to_process = target_ids

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
        # Get unprocessed papers (skip quarantined — see queue.quarantine_condition)
        rows = conn.execute(
            "SELECT id FROM papers WHERE has_text = 0 "
            "AND (is_scanned = 0 OR is_scanned IS NULL) "
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
