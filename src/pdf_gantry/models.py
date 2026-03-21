"""Data classes for Paper, SearchResult, and related types."""

from dataclasses import dataclass, field


@dataclass
class Paper:
    """Represents a paper record from the database."""
    id: int
    path: str
    filename: str
    file_hash: str
    file_size: int
    file_modified: str
    page_count: int | None = None
    has_text: bool = False
    has_markdown: bool = False
    has_embeddings: bool = False
    needs_ocr: bool = False
    is_scanned: bool | None = None
    text_method: str | None = None
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    abstract: str | None = None
    vault_note_path: str | None = None
    error_count: int = 0
    last_error: str | None = None


@dataclass
class SearchResult:
    """A single search result."""
    id: int
    filename: str
    path: str
    score: float
    snippet: str = ""
    title: str | None = None
    has_markdown: bool = False
    has_embeddings: bool = False


@dataclass
class Chunk:
    """A text chunk from a document."""
    chunk_id: int
    doc_id: int
    chunk_index: int
    section_header: str | None
    page_start: int | None
    text: str
    char_offset: int = 0


@dataclass
class ChunkResult:
    """A search result at the chunk level."""
    chunk_id: int
    doc_id: int
    chunk_index: int
    section_header: str | None
    page_start: int | None
    chunk_text: str
    score: float
    filename: str
    path: str
    title: str | None = None


@dataclass
class IngestStats:
    """Statistics from an ingest operation."""
    total_pdfs: int = 0
    already_indexed: int = 0
    new: int = 0
    changed: int = 0
    missing: int = 0
    evicted: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class ProcessStats:
    """Statistics from a processing operation."""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class StatusInfo:
    """Database status information."""
    total: int = 0
    with_text: int = 0
    with_markdown: int = 0
    with_embeddings: int = 0
    needs_ocr: int = 0
    has_errors: int = 0
    with_chunk_embeddings: int = 0
    db_size_bytes: int = 0
    db_path: str = ""
