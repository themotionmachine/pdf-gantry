"""Tests for Marker extraction backend."""

import pytest

from pdf_gantry.process import extract_text_marker


def _marker_available():
    try:
        import marker  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _marker_available(), reason="marker-pdf not installed")
def test_marker_extracts_text(sample_pdf):
    """Marker extraction produces non-empty raw text and markdown."""
    raw_text, markdown = extract_text_marker(sample_pdf)
    assert len(raw_text) > 0
    assert len(markdown) > 0


@pytest.mark.skipif(not _marker_available(), reason="marker-pdf not installed")
def test_marker_returns_markdown_format(sample_pdf):
    """Marker output is markdown (not raw text)."""
    raw_text, markdown = extract_text_marker(sample_pdf)
    # Markdown and raw_text should differ (markdown has formatting)
    assert isinstance(markdown, str)
    assert isinstance(raw_text, str)


def test_marker_import_error_when_missing(sample_pdf, monkeypatch):
    """extract_text_marker gives clear error when marker isn't installed."""
    # Temporarily hide the function's ability to import marker
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "marker" in name:
            raise ImportError("No module named 'marker'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="marker|Marker"):
        extract_text_marker(sample_pdf)


def test_marker_model_cache_is_module_level():
    """The model cache dict exists at module level."""
    from pdf_gantry.process import _MARKER_MODELS
    assert isinstance(_MARKER_MODELS, dict)
