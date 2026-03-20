"""Read-only Obsidian vault integration."""

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .utils import now_iso


@dataclass
class VaultStats:
    """Results from a vault check."""
    total_pdfs: int = 0
    with_notes: int = 0
    without_notes: int = 0
    orphan_references: list[dict] = field(default_factory=list)


def _extract_pdf_references(content: str) -> set[str]:
    """Extract PDF filename references from markdown content."""
    refs = set()

    # Wikilinks: [[filename.pdf]] or [[filename.pdf|display]]
    for m in re.finditer(r'\[\[([^|\]]+\.pdf)(?:\|[^\]]+)?\]\]', content, re.IGNORECASE):
        refs.add(m.group(1))

    # Markdown links: [text](filename.pdf) or [text](path/filename.pdf)
    for m in re.finditer(r'\[[^\]]*\]\(([^)]*\.pdf)\)', content, re.IGNORECASE):
        # Extract just the filename from path
        refs.add(Path(m.group(1)).name)

    # Frontmatter source field: source: filename.pdf
    for m in re.finditer(r'^source:\s*(.+\.pdf)\s*$', content, re.MULTILINE | re.IGNORECASE):
        refs.add(m.group(1).strip())

    return refs


def check_vault(
    conn: sqlite3.Connection,
    vault_dir: Path,
) -> VaultStats:
    """Cross-reference PDFs in database with vault source notes. Read-only."""
    stats = VaultStats()

    # Get all indexed PDFs
    papers = conn.execute("SELECT id, filename FROM papers").fetchall()
    pdf_lookup = {row["filename"]: row["id"] for row in papers}
    stats.total_pdfs = len(pdf_lookup)

    # Track which PDFs have notes
    found_pdfs: set[str] = set()

    # Scan vault markdown files
    for md_file in vault_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        refs = _extract_pdf_references(content)
        rel_path = str(md_file.relative_to(vault_dir))

        for ref in refs:
            if ref in pdf_lookup:
                found_pdfs.add(ref)
            else:
                stats.orphan_references.append({
                    "reference": ref,
                    "note_path": rel_path,
                })

    # Update database
    now = now_iso()
    for filename, paper_id in pdf_lookup.items():
        if filename in found_pdfs:
            # Find the note that references this PDF (for vault_note_path)
            conn.execute(
                "UPDATE papers SET vault_note_path = 'found', vault_checked_at = ?, updated_at = ? WHERE id = ?",
                (now, now, paper_id),
            )
        else:
            conn.execute(
                "UPDATE papers SET vault_note_path = NULL, vault_checked_at = ?, updated_at = ? WHERE id = ?",
                (now, now, paper_id),
            )
    conn.commit()

    stats.with_notes = len(found_pdfs)
    stats.without_notes = stats.total_pdfs - stats.with_notes
    return stats


def get_orphan_references(conn: sqlite3.Connection, vault_dir: Path) -> list[dict]:
    """Find vault notes referencing PDFs not in the library."""
    papers = {row["filename"] for row in conn.execute("SELECT filename FROM papers").fetchall()}
    orphans = []

    for md_file in vault_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        refs = _extract_pdf_references(content)
        rel_path = str(md_file.relative_to(vault_dir))
        for ref in refs:
            if ref not in papers:
                orphans.append({"reference": ref, "note_path": rel_path})

    return orphans


def get_uncovered_pdfs(conn: sqlite3.Connection) -> list[dict]:
    """Find PDFs without any vault note referencing them."""
    rows = conn.execute(
        "SELECT id, filename FROM papers WHERE vault_note_path IS NULL AND vault_checked_at IS NOT NULL"
    ).fetchall()
    return [dict(row) for row in rows]
