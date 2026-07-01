"""Structure-aware markdown chunking for embedding."""

import re
from dataclasses import dataclass


@dataclass
class RawChunk:
    """Intermediate chunk before DB storage."""
    text: str
    section_header: str | None
    page_start: int | None
    char_offset: int


# Sentence-ending patterns for splitting
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\(])')
_HEADER_PATTERN = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)
_TABLE_LINE = re.compile(r'^\|.*\|$')


def _is_table_block(text: str) -> bool:
    """Check if text is a markdown table."""
    lines = text.strip().split('\n')
    return len(lines) >= 2 and all(_TABLE_LINE.match(ln.strip()) for ln in lines if ln.strip())


def _split_into_sections(markdown: str) -> list[tuple[str | None, str, int]]:
    """
    Split markdown by ## and ### headers.
    Returns list of (header_text, section_body, char_offset).
    """
    sections = []
    matches = list(_HEADER_PATTERN.finditer(markdown))

    if not matches:
        # No headers — entire text is one section
        return [(None, markdown, 0)]

    # Text before first header
    if matches[0].start() > 0:
        preamble = markdown[:matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble, 0))

    for i, match in enumerate(matches):
        header_text = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()
        if body:
            sections.append((header_text, body, match.start()))

    return sections


def _split_text_recursive(
    text: str,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[str]:
    """
    Recursively split text at sentence boundaries, then paragraph boundaries,
    then line boundaries. Preserves tables as atomic units.
    """
    if len(text) <= max_chars:
        return [text]

    # Check for paragraph boundaries first
    paragraphs = text.split('\n\n')

    if len(paragraphs) > 1:
        return _merge_splits(paragraphs, '\n\n', max_chars, overlap_chars, min_chars)

    # Try sentence boundaries
    sentences = _SENTENCE_END.split(text)
    if len(sentences) > 1:
        return _merge_splits(sentences, ' ', max_chars, overlap_chars, min_chars)

    # Fall back to line boundaries
    lines = text.split('\n')
    if len(lines) > 1:
        return _merge_splits(lines, '\n', max_chars, overlap_chars, min_chars)

    # Can't split further — return as-is even if over max
    return [text]


def _merge_splits(
    parts: list[str],
    separator: str,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[str]:
    """
    Merge parts into chunks up to max_chars, with overlap.
    Tables are kept as atomic units.
    """
    chunks = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Tables are atomic — never split them
        if _is_table_block(part):
            if current:
                chunks.append(current)
                current = ""
            chunks.append(part)
            continue

        candidate = (current + separator + part).strip() if current else part

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If this single part exceeds max_chars, recursively split it
            if len(part) > max_chars:
                sub_chunks = _split_text_recursive(part, max_chars, overlap_chars, min_chars)
                chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1] if sub_chunks else ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Apply overlap between consecutive chunks
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            # Find a clean break point in the overlap
            space_idx = overlap_text.find(' ')
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            overlapped.append(overlap_text + separator + chunks[i])
        chunks = overlapped

    # Merge undersized trailing chunk
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        chunks[-2] = chunks[-2] + separator + chunks[-1]
        chunks.pop()

    return chunks


def chunk_markdown(
    markdown: str,
    title: str | None = None,
    max_chars: int = 1800,
    overlap_chars: int = 200,
    min_chars: int = 512,
) -> list[RawChunk]:
    """
    Split markdown into chunks preserving document structure.

    Strategy:
    1. Split on ## and ### headers to get sections
    2. Within each section, recursively split at sentence/paragraph boundaries
    3. Never split markdown tables across chunks
    4. Merge undersized trailing chunks with the previous chunk
    5. Track char_offset relative to the original markdown

    Args:
        markdown: The full markdown text
        title: Document title (stored for metadata, not included in chunk text)
        max_chars: Target maximum chunk size (~450 Nomic tokens)
        overlap_chars: Overlap between consecutive chunks (~50 tokens)
        min_chars: Minimum chunk size; smaller chunks merge with previous (~128 tokens)
    """
    if not markdown or not markdown.strip():
        return []

    sections = _split_into_sections(markdown)
    chunks = []

    for header, body, section_offset in sections:
        text_chunks = _split_text_recursive(body, max_chars, overlap_chars, min_chars)

        # Track char offsets within the section
        search_start = section_offset
        for text in text_chunks:
            # Find approximate offset in original markdown
            # Use the start of the non-overlap portion for accuracy
            offset = markdown.find(text[:80], search_start)
            if offset == -1:
                offset = search_start
            else:
                search_start = offset + len(text) // 2

            chunks.append(RawChunk(
                text=text,
                section_header=header,
                page_start=None,  # Could be derived from page markers if present
                char_offset=offset,
            ))

    return chunks


def prepare_chunk_text(
    title: str | None,
    section_header: str | None,
    chunk_text: str,
) -> str:
    """
    Build the text to embed with contextual metadata prepended.

    Format: "title | section_header | chunk_text"
    The "search_document:" prefix is added at embedding time.
    """
    parts = []
    if title:
        parts.append(title)
    if section_header:
        parts.append(section_header)
    parts.append(chunk_text)
    return " | ".join(parts)
