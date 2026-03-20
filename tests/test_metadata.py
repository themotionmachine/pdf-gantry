"""Tests for metadata enrichment."""

import pytest

from pdf_gantry.metadata import extract_doi


def test_extract_doi_standard():
    """Standard DOI pattern is extracted."""
    text = "Available at https://doi.org/10.1234/test.paper.2024"
    doi = extract_doi(text)
    assert doi == "10.1234/test.paper.2024"


def test_extract_doi_in_text():
    """DOI embedded in text is extracted."""
    text = "The paper (doi: 10.1038/nature12373) discusses..."
    doi = extract_doi(text)
    assert doi == "10.1038/nature12373"


def test_extract_doi_none():
    """No DOI returns None."""
    text = "This is a document without any DOI"
    doi = extract_doi(text)
    assert doi is None


def test_extract_doi_trailing_punctuation():
    """Trailing punctuation is stripped from DOI."""
    text = "See 10.1234/test.paper."
    doi = extract_doi(text)
    assert doi == "10.1234/test.paper"


def test_extract_doi_with_parenthesis():
    """DOI followed by closing paren is cleaned."""
    text = "(10.1234/test.paper)"
    doi = extract_doi(text)
    assert doi == "10.1234/test.paper"
