"""Tests for fuzzy filename lookup."""


from pdf_gantry.search import find_papers


def test_find_exact_match(populated_db):
    """Exact filename match returns the paper."""
    results = find_papers(populated_db, "test_climate.pdf")
    assert len(results) == 1
    assert results[0]["filename"] == "test_climate.pdf"


def test_find_partial_match(populated_db):
    """Partial filename fragment matches."""
    results = find_papers(populated_db, "climate")
    assert len(results) >= 1
    assert any("climate" in r["filename"].lower() for r in results)


def test_find_case_insensitive(populated_db):
    """Search is case-insensitive."""
    results = find_papers(populated_db, "CLIMATE")
    assert len(results) >= 1


def test_find_multiple_results(populated_db):
    """Fragment matching multiple files returns all."""
    results = find_papers(populated_db, ".pdf")
    assert len(results) == 2  # Both test PDFs


def test_find_no_match(populated_db):
    """No match returns empty list."""
    results = find_papers(populated_db, "nonexistent_xyz")
    assert results == []


def test_find_respects_limit(populated_db):
    """Limit parameter caps results."""
    results = find_papers(populated_db, ".pdf", limit=1)
    assert len(results) == 1
