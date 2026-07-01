"""Tests for structure-aware markdown chunking."""


from pdf_gantry.chunking import RawChunk, chunk_markdown, prepare_chunk_text


def test_basic_splitting():
    """Long text is split into multiple chunks."""
    text = "This is a sentence. " * 200  # ~4000 chars
    chunks = chunk_markdown(text, max_chars=1800)
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, RawChunk)
        assert len(chunk.text) > 0


def test_header_based_sections():
    """Markdown headers create section boundaries with tracked headers."""
    text = """## Introduction

This is the introduction section with enough text to fill a chunk.

## Methods

This is the methods section with different content about methodology.

## Results

This is the results section discussing findings and outcomes.
"""
    chunks = chunk_markdown(text, max_chars=5000)
    headers = [c.section_header for c in chunks]
    assert "Introduction" in headers
    assert "Methods" in headers
    assert "Results" in headers


def test_short_sections_not_merged_across_headers():
    """Each section produces its own chunk(s) even if short."""
    text = """## Section A

Short text A.

## Section B

Short text B.
"""
    chunks = chunk_markdown(text, max_chars=5000, min_chars=10)
    assert len(chunks) >= 2
    assert chunks[0].section_header == "Section A"
    assert chunks[1].section_header == "Section B"


def test_table_preservation():
    """Markdown tables are not split across chunks."""
    table = "| Col1 | Col2 |\n| --- | --- |\n| a | b |\n| c | d |"
    text = f"Some intro text.\n\n{table}\n\nSome trailing text."
    chunks = chunk_markdown(text, max_chars=50, min_chars=10)

    # Find the chunk containing the table
    table_chunks = [c for c in chunks if "| Col1 |" in c.text]
    assert len(table_chunks) >= 1
    # The table should be intact (all rows in one chunk)
    for tc in table_chunks:
        assert "| a | b |" in tc.text
        assert "| c | d |" in tc.text


def test_minimum_size_merging():
    """Trailing chunks smaller than min_chars merge with previous."""
    text = "A" * 1500 + "\n\n" + "B" * 100
    chunks = chunk_markdown(text, max_chars=1800, min_chars=512)
    # The 100-char trailing part should be merged, not standalone
    assert all(len(c.text) >= 100 for c in chunks)


def test_overlap():
    """Consecutive chunks have overlapping content."""
    text = ". ".join(f"Sentence number {i} with some extra words" for i in range(50))
    chunks = chunk_markdown(text, max_chars=500, overlap_chars=100, min_chars=50)
    if len(chunks) >= 2:
        # End of first chunk should appear at start of second
        end_of_first = chunks[0].text[-50:]
        assert end_of_first in chunks[1].text or chunks[1].text[:100] in chunks[0].text[-200:]


def test_prepare_chunk_text_full():
    """Metadata prepending with all fields."""
    result = prepare_chunk_text("My Paper Title", "Introduction", "Some text here")
    assert result == "My Paper Title | Introduction | Some text here"


def test_prepare_chunk_text_no_title():
    """Metadata prepending without title."""
    result = prepare_chunk_text(None, "Methods", "Some text")
    assert result == "Methods | Some text"


def test_prepare_chunk_text_no_metadata():
    """Metadata prepending with only text."""
    result = prepare_chunk_text(None, None, "Just the text")
    assert result == "Just the text"


def test_empty_input():
    """Empty markdown returns no chunks."""
    assert chunk_markdown("") == []
    assert chunk_markdown("   ") == []


def test_single_short_section():
    """Short text produces exactly one chunk."""
    chunks = chunk_markdown("A short paragraph.", max_chars=1800, min_chars=10)
    assert len(chunks) == 1
    assert chunks[0].text == "A short paragraph."


def test_char_offset_tracking():
    """Chunk offsets reference positions in the original markdown."""
    text = "First section.\n\n## Methods\n\nSecond section with more text."
    chunks = chunk_markdown(text, max_chars=5000)
    for chunk in chunks:
        assert isinstance(chunk.char_offset, int)
        assert chunk.char_offset >= 0


def test_h3_headers_split():
    """### headers also create section boundaries."""
    text = """### Subsection A

Content A here.

### Subsection B

Content B here.
"""
    chunks = chunk_markdown(text, max_chars=5000, min_chars=10)
    headers = [c.section_header for c in chunks]
    assert "Subsection A" in headers
    assert "Subsection B" in headers
