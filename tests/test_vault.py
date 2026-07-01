"""Tests for read-only vault integration."""

import pytest

from pdf_gantry.db import get_connection
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.vault import (
    _extract_pdf_references,
    check_vault,
    get_orphan_references,
    get_uncovered_pdfs,
)


@pytest.fixture
def vault_dir(tmp_path):
    """Creates a mock Obsidian vault with source notes."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Note referencing a PDF via wikilink
    (vault / "climate_note.md").write_text(
        "# Climate Research\n\nSee [[test_climate.pdf]] for details.\n"
    )

    # Note referencing a PDF via markdown link
    (vault / "research").mkdir()
    (vault / "research" / "ml_note.md").write_text(
        "# ML Paper\n\n[paper](ml_nlp_paper.pdf)\n"
    )

    # Note referencing a PDF that doesn't exist in library
    (vault / "orphan_note.md").write_text(
        "# Missing Paper\n\nSee [[nonexistent_paper.pdf]]\n"
    )

    # Note with frontmatter source
    (vault / "frontmatter_note.md").write_text(
        "---\nsource: test_climate.pdf\ntags: climate\n---\n\nSome content.\n"
    )

    return vault


def test_extract_wikilink():
    """Wikilink PDF references are detected."""
    refs = _extract_pdf_references("See [[paper.pdf]] here")
    assert "paper.pdf" in refs


def test_extract_wikilink_with_alias():
    """Wikilinks with display aliases are detected."""
    refs = _extract_pdf_references("See [[paper.pdf|My Paper]] here")
    assert "paper.pdf" in refs


def test_extract_markdown_link():
    """Markdown-style PDF links are detected."""
    refs = _extract_pdf_references("[click here](paper.pdf)")
    assert "paper.pdf" in refs


def test_extract_markdown_link_with_path():
    """Markdown links with paths extract just the filename."""
    refs = _extract_pdf_references("[click](attachments/paper.pdf)")
    assert "paper.pdf" in refs


def test_extract_frontmatter_source():
    """Frontmatter source field is detected."""
    content = "---\nsource: paper.pdf\n---\nContent"
    refs = _extract_pdf_references(content)
    assert "paper.pdf" in refs


def test_extract_no_references():
    """No references returns empty set."""
    refs = _extract_pdf_references("Just some text with no PDFs")
    assert len(refs) == 0


def test_vault_check(tmp_path, papers_dir, vault_dir):
    """Vault check finds matching source notes."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    stats = check_vault(conn, vault_dir)

    assert stats.total_pdfs == 2
    assert stats.with_notes == 2  # Both test PDFs are referenced
    assert stats.without_notes == 0
    conn.close()


def test_vault_orphans(tmp_path, papers_dir, vault_dir):
    """Orphan references identify missing PDFs."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    orphans = get_orphan_references(conn, vault_dir)
    refs = [o["reference"] for o in orphans]
    assert "nonexistent_paper.pdf" in refs
    conn.close()


def test_vault_coverage(tmp_path, papers_dir, vault_dir):
    """Coverage shows PDFs without notes after vault check."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    # Before check, vault_checked_at is NULL so get_uncovered_pdfs returns nothing
    uncovered = get_uncovered_pdfs(conn)
    assert len(uncovered) == 0

    # After check
    check_vault(conn, vault_dir)
    uncovered = get_uncovered_pdfs(conn)
    assert len(uncovered) == 0  # Both referenced
    conn.close()


def test_vault_read_only(tmp_path, papers_dir, vault_dir):
    """Vault operations don't modify vault files."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)

    # Record vault file contents before
    vault_files = {}
    for f in vault_dir.rglob("*.md"):
        vault_files[f] = f.read_text()

    check_vault(conn, vault_dir)

    # Verify vault files unchanged
    for f, content in vault_files.items():
        assert f.read_text() == content
    conn.close()
