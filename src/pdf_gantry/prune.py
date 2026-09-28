"""Remove database entries for files no longer on disk."""

import sqlite3
from pathlib import Path

from .fts import delete_row


def prune_missing(
    conn: sqlite3.Connection,
    papers_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Remove DB entries for papers whose files no longer exist on disk.

    Cleans up: papers, paper_text, papers_fts, chunks, chunk_vec, paper_embeddings.
    Returns stats dict with pruned count, remaining count, and pruned filenames.
    """
    if not papers_dir.is_dir():
        raise FileNotFoundError(
            f"papers_dir does not exist or is not a directory: {papers_dir}"
        )

    rows = conn.execute("SELECT id, path, filename FROM papers").fetchall()

    missing = []
    for row in rows:
        file_path = papers_dir / row["path"]
        if not file_path.exists():
            missing.append(row)

    if not dry_run and missing:
        for row in missing:
            paper_id = row["id"]

            # vec0 tables have no CASCADE — delete explicitly
            conn.execute(
                "DELETE FROM chunk_vec WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                (paper_id,),
            )
            conn.execute(
                "DELETE FROM paper_embeddings WHERE paper_id = ?",
                (paper_id,),
            )

            # FTS5 contentless: no CASCADE; delete before the papers row goes.
            delete_row(conn, paper_id)

            # Regular tables — CASCADE handles chunks and paper_text
            conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))

        conn.commit()

    remaining = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    return {
        "pruned": len(missing),
        "remaining": remaining if not dry_run else remaining - len(missing),
        "pruned_files": [r["filename"] for r in missing],
    }
