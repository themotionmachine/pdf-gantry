# pdf_gantry PRD

## Claude Code Implementation Guide

**Project:** pdf_gantry (CLI name: `gantry`)
**Location:** `~/Desktop/code/pdf-gantry`
**Repository:** `rwm/pdf-gantry` (private)
**Language:** Python 3.11+
**Author:** Ryan

---

## 1. Project Overview

gantry is a CLI tool for managing a library of ~2000 academic PDFs stored in a flat iCloud folder. It replaces DEVONthink in a PKM stack built around Obsidian.

What it does: indexes PDFs into a single SQLite database (using sqlite-vec), extracts text and markdown, generates embeddings, and provides full-text and semantic search. It integrates read-only with an Obsidian vault to cross-reference source notes.

Who it's for: a single user running macOS on Apple Silicon (M1 Max), invoking gantry directly from the terminal or through Claude Code / AI agent sessions.

### Design Philosophy

**The PDF folder is the source of truth.** ~2000 PDFs live in a flat iCloud directory. gantry never moves, renames, or organizes them. All organization lives in the Obsidian vault via source notes and wikilinks.

**Queue-driven, property-based processing.** Every document in the database carries a set of boolean and versioned flags indicating what processing it has received (has_text, has_markdown, has_embeddings, embedding_model_version, needs_ocr, etc.). The core architectural pattern is: filter documents by property, then apply an operation to the resulting queue. This means you can always ask "what still needs X?" and get an answer, and you can always say "do X to everything that needs it."

**Agent-first output.** Default output is terse and token-efficient. Every command supports `--json` for structured output. Exit codes are meaningful. This tool is designed to be called by Claude Code sessions and AI agents as much as by humans.

**Vault integration is read-only.** gantry can read Obsidian vault files to check which PDFs have source notes, but it never writes to the vault.

**Easy cases first, hard cases later.** Development proceeds in phases: digital PDFs with embedded text first, then complex layouts via Marker, then scanned PDFs via OCR. Each phase tags documents so the next phase knows what remains.

---

## 2. Architecture

### 2.1 Package Structure

```
pdf-gantry/
  pyproject.toml
  README.md
  src/
    pdf_gantry/
      __init__.py
      cli.py              # Click CLI entry point and command group
      config.py            # Config loading and validation
      db.py                # Database connection, schema, migrations
      models.py            # Data classes for Paper, SearchResult, etc.
      ingest.py            # File scanning and registration
      process.py           # Text extraction, markdown conversion
      search.py            # FTS5 and semantic search
      embeddings.py        # Embedding generation and storage
      queue.py             # Property-based filtering and queue display
      vault.py             # Read-only vault integration
      metadata.py          # Semantic Scholar API integration
      ocr.py               # OCR detection and processing
      utils.py             # Hashing, formatting, shared helpers
  tests/
    conftest.py
    test_config.py
    test_db.py
    test_ingest.py
    test_process.py
    test_search.py
    test_queue.py
    test_embeddings.py
    test_vault.py
    fixtures/
      sample.pdf           # A small digital PDF for testing
      scanned.pdf          # A scanned PDF for OCR testing
      two_column.pdf       # A two-column layout PDF
```

### 2.2 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pdf-gantry"
version = "0.1.0"
description = "Agent-friendly CLI for managing academic PDF libraries"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "click>=8.1",
    "sqlite-vec>=0.1.6",
    "pymupdf>=1.25",
    "pymupdf4llm>=0.0.17",
    "pyyaml>=6.0",
    "rich>=13.0",
]

[project.optional-dependencies]
quality = [
    "marker-pdf>=1.10",
]
ocr = [
    "surya-ocr>=0.8",
]
embeddings = [
    "sentence-transformers>=3.0",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.8",
]
all = ["pdf-gantry[quality,ocr,embeddings,dev]"]

[project.scripts]
gantry = "pdf_gantry.cli:cli"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### 2.3 Database Schema

gantry uses a single SQLite database file (default: `~/.gantry/index.db`). The database loads the `sqlite-vec` extension at connection time.

```sql
-- Core papers table
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,          -- relative path from papers_dir
    filename TEXT NOT NULL,             -- basename for display
    file_hash TEXT NOT NULL,            -- SHA-256 of file contents
    file_size INTEGER NOT NULL,         -- bytes
    file_modified TEXT NOT NULL,        -- ISO 8601 timestamp from filesystem
    page_count INTEGER,                 -- number of pages (from PyMuPDF)

    -- Processing state flags
    has_text INTEGER NOT NULL DEFAULT 0,        -- raw text extracted
    has_markdown INTEGER NOT NULL DEFAULT 0,     -- markdown generated
    has_embeddings INTEGER NOT NULL DEFAULT 0,   -- embeddings computed
    needs_ocr INTEGER NOT NULL DEFAULT 0,        -- detected as scanned
    is_scanned INTEGER,                          -- NULL = unknown, 0 = digital, 1 = scanned

    -- Processing metadata
    text_method TEXT,                    -- 'pymupdf4llm' | 'marker' | 'surya'
    text_extracted_at TEXT,              -- ISO 8601
    markdown_method TEXT,                -- 'pymupdf4llm' | 'marker'
    markdown_extracted_at TEXT,          -- ISO 8601
    embedding_model TEXT,                -- e.g. 'nomic-embed-text-v2'
    embedding_model_version TEXT,        -- version string for reprocessing
    embedding_computed_at TEXT,          -- ISO 8601
    ocr_method TEXT,                     -- 'surya' | NULL
    ocr_completed_at TEXT,              -- ISO 8601

    -- External metadata
    title TEXT,
    authors TEXT,                        -- JSON array of author names
    year INTEGER,
    doi TEXT,
    abstract TEXT,
    semantic_scholar_id TEXT,
    metadata_source TEXT,                -- 'semantic_scholar' | 'crossref' | 'manual'
    metadata_enriched_at TEXT,           -- ISO 8601

    -- Vault integration (read-only cache)
    vault_note_path TEXT,                -- path to source note in vault, if found
    vault_checked_at TEXT,               -- ISO 8601, last time we checked

    -- Timestamps
    indexed_at TEXT NOT NULL,            -- ISO 8601, when first added
    updated_at TEXT NOT NULL,            -- ISO 8601, last modified in db

    -- Error tracking
    last_error TEXT,                     -- last processing error message
    error_count INTEGER NOT NULL DEFAULT 0,
    last_error_at TEXT                   -- ISO 8601
);

CREATE INDEX IF NOT EXISTS idx_papers_hash ON papers(file_hash);
CREATE INDEX IF NOT EXISTS idx_papers_has_text ON papers(has_text);
CREATE INDEX IF NOT EXISTS idx_papers_has_markdown ON papers(has_markdown);
CREATE INDEX IF NOT EXISTS idx_papers_has_embeddings ON papers(has_embeddings);
CREATE INDEX IF NOT EXISTS idx_papers_needs_ocr ON papers(needs_ocr);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);

-- Full-text search virtual table
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    filename,
    title,
    authors,
    abstract,
    text_content,
    content='',                          -- contentless: we manage content ourselves
    tokenize='porter unicode61'
);

-- Extracted text storage (separate to keep papers table lean)
CREATE TABLE IF NOT EXISTS paper_text (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    raw_text TEXT,                        -- plain text extraction
    markdown TEXT,                        -- markdown conversion
    text_length INTEGER,                 -- char count of raw_text
    markdown_length INTEGER              -- char count of markdown
);

-- Embeddings via sqlite-vec
-- vec0 virtual table: each row is one embedding keyed by paper_id
CREATE VIRTUAL TABLE IF NOT EXISTS paper_embeddings USING vec0(
    paper_id INTEGER PRIMARY KEY,
    embedding FLOAT[768]                 -- Nomic Embed V2 outputs 768-d vectors
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL
);
```

Key design decisions in the schema:

The `papers` table carries all processing state as indexed boolean columns. This makes queue queries trivial: `SELECT * FROM papers WHERE has_text = 0` gives you every document that still needs text extraction. The `text_method`, `embedding_model_version`, and similar fields let you reprocess when tools change: `SELECT * FROM papers WHERE embedding_model_version != 'v2.0'`.

Text content lives in a separate `paper_text` table so that queries against `papers` metadata stay fast and don't drag megabytes of text through memory. The FTS5 table is contentless -- gantry manages inserts and deletes explicitly, which avoids the complexity of content-sync triggers.

Embeddings use `vec0` from sqlite-vec. The dimension (768) matches Nomic Embed V2. If you later switch models, you add a migration that drops and recreates the virtual table with the new dimension.

### 2.4 Config System

Config file location: `~/.gantry/config.yaml` (created on first run if missing).

```yaml
# ~/.gantry/config.yaml

# Required: where the PDFs live
papers_dir: ~/Library/Mobile Documents/com~apple~CloudDocs/Papers

# Where gantry stores its index database
index_dir: ~/.gantry

# Obsidian vault root (for read-only vault integration)
vault_dir: ~/obsidian-vault

# Embedding settings
embedding:
  model: nomic-ai/nomic-embed-text-v2-moe
  dimensions: 768
  batch_size: 32
  # Alternative: use ollama instead of sentence-transformers
  # backend: ollama
  # ollama_model: nomic-embed-text

# Processing settings
processing:
  # Number of concurrent workers for batch operations
  workers: 4
  # Threshold for scanned PDF detection (ratio of image area to page area)
  scan_threshold: 0.8
  # Default text extraction method: 'pymupdf4llm' or 'marker'
  default_method: pymupdf4llm

# Semantic Scholar API (no key required for basic lookups)
semantic_scholar:
  rate_limit: 10  # requests per second (free tier)

# Display
output:
  # Default output format: 'text' or 'json'
  format: text
  # Max results to show by default
  default_limit: 20
```

Config resolution order: CLI flags > environment variables (`GANTRY_PAPERS_DIR`, etc.) > config file > defaults.

The config module (`config.py`) should:
- Look for `~/.gantry/config.yaml` on startup
- Create `~/.gantry/` directory if it doesn't exist
- Validate that `papers_dir` exists and is a directory
- Expand `~` in all paths
- Provide a `Config` dataclass with typed access to all values
- Support `gantry config show` to print current resolved config
- Support `gantry config set <key> <value>` for simple overrides

### 2.5 The Queue/Filter System

This is the architectural backbone. The filter system lets any command select a subset of documents based on their processing state.

The filter syntax is a set of named flags that map to columns on the `papers` table:

```
--needs text          -> has_text = 0
--needs markdown      -> has_markdown = 0
--needs embeddings    -> has_embeddings = 0
--needs ocr           -> needs_ocr = 1 AND ocr_completed_at IS NULL
--needs metadata      -> doi IS NULL OR metadata_enriched_at IS NULL
--has text            -> has_text = 1
--has markdown        -> has_markdown = 1
--has embeddings      -> has_embeddings = 1
--is scanned          -> is_scanned = 1
--is digital          -> is_scanned = 0
--has errors          -> error_count > 0
--stale-embeddings    -> embedding_model_version != <current_model_version>
```

Multiple filters combine with AND. So `gantry queue --needs text --is digital` returns digital PDFs that still need text extraction.

The `queue` module exposes a function:

```python
def build_filter_query(
    needs: list[str] | None = None,
    has: list[str] | None = None,
    is_prop: list[str] | None = None,
    has_errors: bool = False,
    stale_embeddings: bool = False,
) -> tuple[str, list]:
    """
    Returns (WHERE clause, params) for filtering the papers table.
    """
```

This function is used by `gantry queue` to display matching documents, and by `gantry process`, `gantry embed`, and other batch commands to select their work queue. Every batch-capable command accepts the same `--needs` / `--has` / `--is` flags through shared Click option decorators.

Implementation pattern for shared filter options in Click:

```python
import click
from functools import wraps

def filter_options(f):
    """Shared Click options for document filtering."""
    @click.option('--needs', multiple=True,
                  type=click.Choice(['text', 'markdown', 'embeddings', 'ocr', 'metadata']),
                  help='Filter to documents missing this property')
    @click.option('--has', multiple=True,
                  type=click.Choice(['text', 'markdown', 'embeddings', 'errors']),
                  help='Filter to documents with this property')
    @click.option('--is', 'is_prop', multiple=True,
                  type=click.Choice(['scanned', 'digital']),
                  help='Filter by document type')
    @click.option('--stale-embeddings', is_flag=True,
                  help='Documents with outdated embedding model version')
    @click.option('--limit', type=int, default=None,
                  help='Max documents to process')
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper
```

---

## 3. Phase 1: Foundation + Easy Cases

Phase 1 gets the project running with a working CLI, database, config, ingest pipeline, text extraction for digital PDFs, full-text search, and the queue system. After Phase 1, you can ingest the full corpus, extract text from most documents, and search across them.

### 3.1 Project Scaffold

**Task:** Create the project directory structure, pyproject.toml, and package skeleton.

Create every file listed in the package structure (section 2.1). The `__init__.py` should export a `__version__` string. Each module file should start as a stub with its docstring and key imports, to be filled in as the phase progresses.

Install in editable mode:

```bash
cd ~/Desktop/code/pdf-gantry
pip install -e ".[dev]"
```

Verify: `gantry --help` prints the help text.

### 3.2 Config System

**Task:** Implement `config.py` and the `gantry config` command group.

```
gantry config show
```

Output (text mode):
```
papers_dir: /Users/ryan/Library/Mobile Documents/com~apple~CloudDocs/Papers
index_dir:  /Users/ryan/.gantry
vault_dir:  /Users/ryan/obsidian-vault
database:   /Users/ryan/.gantry/index.db
```

Output (JSON mode, `gantry config show --json`):
```json
{
  "papers_dir": "/Users/ryan/Library/Mobile Documents/com~apple~CloudDocs/Papers",
  "index_dir": "/Users/ryan/.gantry",
  "vault_dir": "/Users/ryan/obsidian-vault",
  "database": "/Users/ryan/.gantry/index.db"
}
```

```
gantry config set papers_dir ~/Papers
```

Writes the updated value to `~/.gantry/config.yaml`.

```
gantry config init
```

Interactive first-run setup. Prompts for papers_dir (required), vault_dir (optional), creates `~/.gantry/` and writes the config file.

### 3.3 Database Module

**Task:** Implement `db.py` with schema creation and connection management.

The `db.py` module provides:

```python
def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """
    Open a connection, load sqlite-vec extension, enable WAL mode,
    set pragmas (journal_mode=WAL, foreign_keys=ON, busy_timeout=5000),
    and run migrations if needed.
    """

def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables, indices, and virtual tables if they don't exist."""

def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return current schema version, 0 if no schema_version table."""

def migrate(conn: sqlite3.Connection) -> None:
    """Apply any pending schema migrations."""
```

Loading sqlite-vec:

```python
import sqlite3
import sqlite_vec

db = sqlite3.connect(db_path)
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
```

### 3.4 `gantry ingest`

**Task:** Implement `ingest.py` and the `gantry ingest` command.

`gantry ingest` scans `papers_dir` for PDF files and registers each one in the database. For each PDF:

1. Compute SHA-256 hash of the file.
2. Check if a record with this path already exists.
   - If yes and hash matches: skip (already indexed).
   - If yes and hash differs: update the record, clear processing flags that depend on content (has_text, has_markdown, has_embeddings), set updated_at.
   - If no: insert a new record with basic file metadata.
3. For new files, use PyMuPDF to get page_count and run the scanned-vs-digital heuristic (section 3.7).
4. Detect files in the database that no longer exist on disk. Mark them (do not delete -- flag with a `removed_at` timestamp or print a warning).

CLI signature:

```
gantry ingest [--dry-run] [--json]
```

Default text output:
```
Scanning /Users/ryan/Library/Mobile Documents/.../Papers
Found 2,041 PDFs (1,893 indexed, 148 new, 0 changed, 3 missing)
Indexed 148 new documents in 12.3s
```

With `--dry-run`, report what would happen without writing to the database.

With `--json`:
```json
{
  "total_pdfs": 2041,
  "already_indexed": 1893,
  "new": 148,
  "changed": 0,
  "missing": 3,
  "elapsed_seconds": 12.3
}
```

**Performance note:** Hashing 2000 PDFs (~20GB) will take a few minutes. Use a progress indicator (Rich progress bar) in text mode. Consider caching: if file path + size + mtime match the existing record, skip the hash computation (fast path).

### 3.5 `gantry status`

**Task:** Implement the `gantry status` command.

```
gantry status [--json]
```

Text output:
```
pdf_gantry index: /Users/ryan/.gantry/index.db

Documents:     2,041
  With text:   1,842 (90.2%)
  With markdown: 0 (0.0%)
  With embeddings: 0 (0.0%)
  Needs OCR:   199 (9.8%)
  Has errors:  3 (0.1%)

Database size: 45.2 MB
Last ingest:   2026-03-20 14:32:01
```

JSON output: same data as a flat JSON object with integer counts and float percentages.

### 3.6 `gantry process`

**Task:** Implement `process.py` and the `gantry process` command.

In Phase 1, `gantry process` uses PyMuPDF4LLM to extract text and markdown from digital PDFs.

```
gantry process [PATH] [--method pymupdf4llm] [--workers N] [--limit N] [FILTER_OPTIONS] [--json]
```

If `PATH` is given, process that single PDF. Otherwise, process all documents matching the filter (default: `--needs text --is digital` if no filter specified).

For each document:

1. Open the PDF with PyMuPDF.
2. Extract raw text via `page.get_text()` concatenated across pages.
3. Extract markdown via `pymupdf4llm.to_markdown()`.
4. Write raw_text and markdown to the `paper_text` table.
5. Populate the FTS5 index: insert into `papers_fts` with the filename, title (if known), authors, abstract, and text_content.
6. Update the `papers` record: set `has_text = 1`, `has_markdown = 1`, `text_method = 'pymupdf4llm'`, `text_extracted_at` and `markdown_extracted_at` to now.
7. If extraction fails, increment `error_count`, set `last_error` and `last_error_at`, and continue to the next document.

Text output during processing:
```
Processing 1,842 documents with pymupdf4llm (4 workers)
[################    ] 1,200/1,842  65.1%  ~4m remaining
```

Completion output:
```
Processed 1,842 documents in 3h 12m
  Succeeded: 1,839
  Failed: 3 (use 'gantry queue --has errors' to see failures)
```

**Concurrency:** Use `concurrent.futures.ProcessPoolExecutor` for CPU-bound PyMuPDF work. Each worker opens its own database connection. Batch commits every 50 documents to avoid holding long transactions.

### 3.7 Scanned PDF Detection

During ingest or as part of processing, run a heuristic to classify each PDF as scanned or digital:

```python
def classify_document(pdf_path: str) -> str:
    """
    Returns 'scanned', 'digital', or 'mixed'.

    Heuristic: For each page, extract text blocks via page.get_text("blocks").
    Calculate the ratio of text area to page area. If the average ratio
    across all pages is below a threshold (default 0.05), classify as scanned.
    If above 0.5, classify as digital. Otherwise mixed.
    """
```

Set `is_scanned` and `needs_ocr` accordingly on the papers record. Phase 1 skips scanned PDFs during processing and leaves them for Phase 3.

### 3.8 `gantry search`

**Task:** Implement `search.py` and the `gantry search` command.

```
gantry search <query> [--limit N] [--json]
```

Uses FTS5 `MATCH` syntax. The query is passed directly to FTS5, which supports boolean operators (`AND`, `OR`, `NOT`), phrase queries (`"exact phrase"`), prefix queries (`term*`), and column filters (`title:climate`).

Text output:
```
Found 23 results for "climate adaptation"

 1. [0.92] Hansen_2016_climate_scenarios.pdf
    "...discusses climate adaptation strategies for coastal..."

 2. [0.87] IPCC_AR6_Chapter4.pdf
    "...framework for climate adaptation in developing..."

 3. [0.81] Adger_2005_social_vulnerability.pdf
    "...social dimensions of climate adaptation and..."
```

Each result shows: rank score (from FTS5 `rank`), filename, and a snippet (from FTS5 `snippet()` function).

JSON output:
```json
{
  "query": "climate adaptation",
  "total": 23,
  "results": [
    {
      "id": 142,
      "filename": "Hansen_2016_climate_scenarios.pdf",
      "path": "Hansen_2016_climate_scenarios.pdf",
      "score": 0.92,
      "snippet": "...discusses climate adaptation strategies for coastal...",
      "has_markdown": true,
      "has_embeddings": false
    }
  ]
}
```

### 3.9 `gantry queue`

**Task:** Implement `queue.py` and the `gantry queue` command.

```
gantry queue [FILTER_OPTIONS] [--count] [--json]
```

Displays documents matching the given filters. Without filters, shows all documents that need any processing.

```
gantry queue --needs text
```

Output:
```
199 documents need text extraction

  1. Scanned_thesis_2019.pdf (scanned, 342 pages)
  2. Old_conference_paper.pdf (scanned, 12 pages)
  ...
```

With `--count`, just print the number:
```
199
```

This is the command agents will use most. A Claude Code session can run `gantry queue --needs text --is digital --count --json` to get `{"count": 47}` and decide whether to kick off processing.

### 3.10 Phase 1 Testing Strategy

Tests for Phase 1 should cover:

**test_config.py:**
- Config loading from YAML file
- Config defaults when no file exists
- CLI flag overrides
- Path expansion (~)
- Validation (papers_dir must exist in non-test mode)

**test_db.py:**
- Schema creation on fresh database
- sqlite-vec extension loading
- WAL mode is active
- Migration from version 0 to current
- FTS5 table creation and basic insert/query
- vec0 table creation

**test_ingest.py:**
- Ingest a directory with sample PDFs
- Skip already-indexed files (by hash)
- Detect changed files (path exists, hash differs)
- Detect missing files
- Dry-run mode produces no database changes
- Page count extraction
- Scanned vs digital classification

**test_process.py:**
- Extract text from a digital PDF
- Extract markdown from a digital PDF
- FTS5 index is populated after processing
- Processing flags are set correctly
- Error handling: corrupt PDF increments error_count

**test_search.py:**
- FTS5 search returns ranked results
- Boolean queries work (AND, OR, NOT)
- Phrase queries work
- Column-specific queries work (title:X)
- Empty results return empty list, not error
- Snippet generation

**test_queue.py:**
- Filter by --needs text returns correct documents
- Filter by --has text returns correct documents
- Multiple filters combine with AND
- --count returns integer
- --json output is valid JSON

Use pytest fixtures to create a temporary database with known test data. The `conftest.py` should provide:

```python
@pytest.fixture
def tmp_db(tmp_path):
    """Provides a temporary database with schema initialized."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    init_schema(conn)
    return conn

@pytest.fixture
def sample_pdf(tmp_path):
    """Creates a minimal valid PDF for testing."""
    # Use PyMuPDF to create a one-page PDF with known text
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a test document about climate change.")
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
```

### 3.11 Phase 1 Definition of Done

Phase 1 is complete when:
- `gantry --help` works
- `gantry config init` creates a config file
- `gantry ingest` scans the papers folder and populates the database
- `gantry status` shows accurate coverage stats
- `gantry process` extracts text from digital PDFs using PyMuPDF4LLM
- `gantry search "some query"` returns ranked results with snippets
- `gantry queue --needs text` shows documents still needing processing
- All tests pass
- The full ~2000 PDF corpus can be ingested and the digital subset processed

---

## 4. Phase 2: Semantic Search + Quality Processing

Phase 2 adds embedding-based semantic search, hybrid search combining FTS5 and vectors, and Marker integration for higher-quality markdown from complex PDFs.

### 4.1 Embedding Pipeline

**Task:** Implement `embeddings.py` and the `gantry embed` command.

Model: Nomic Embed V2 (MoE), 768 dimensions, 8K context. Load via sentence-transformers:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
```

For each document, embed the concatenation of title + abstract + first N tokens of text content (fitting within 8K token context). Store the resulting 768-d vector in the `paper_embeddings` vec0 table.

CLI signature:

```
gantry embed [FILTER_OPTIONS] [--batch-size N] [--json]
```

Default filter: `--needs embeddings --has text` (only embed documents that have text but no embeddings yet).

Processing:
1. Load the embedding model once.
2. Query for documents matching the filter.
3. For each document, retrieve text from `paper_text`.
4. Batch documents (default batch_size=32) and encode.
5. Insert/update vectors in `paper_embeddings`.
6. Update `papers` record: `has_embeddings = 1`, `embedding_model`, `embedding_model_version`, `embedding_computed_at`.

Output:
```
Embedding 1,839 documents with nomic-embed-text-v2-moe (batch_size=32)
[####################] 1,839/1,839  100%  elapsed: 58s
Done. 1,839 documents embedded.
```

**Performance:** At ~30-50 docs/sec on M1 Max, the full corpus embeds in under 2 minutes. This is fast enough to re-embed everything when switching models.

### 4.2 `gantry semantic`

**Task:** Implement semantic search in `search.py`.

```
gantry semantic <query> [--limit N] [--json]
```

Process:
1. Encode the query string using the same embedding model.
2. Query `paper_embeddings` using sqlite-vec's distance function.
3. Join with `papers` to get metadata.
4. Return ranked results.

sqlite-vec query pattern:

```sql
SELECT
    p.id, p.filename, p.title,
    e.distance
FROM paper_embeddings e
INNER JOIN papers p ON p.id = e.paper_id
WHERE e.embedding MATCH ?
    AND k = ?
ORDER BY e.distance
```

The `?` for embedding is the query vector serialized as a blob (`struct.pack('768f', *query_vector)` or using `sqlite_vec.serialize_float32()`).

Text output follows the same format as `gantry search` but with cosine similarity scores instead of FTS5 rank.

### 4.3 Hybrid Search

**Task:** Implement `gantry search` with `--hybrid` flag.

```
gantry search <query> [--hybrid] [--limit N] [--json]
```

Hybrid search combines FTS5 and vector results using Reciprocal Rank Fusion (RRF):

```python
def hybrid_search(query: str, limit: int = 20, k: int = 60) -> list[SearchResult]:
    """
    1. Run FTS5 search, get top 2*limit results with ranks.
    2. Run semantic search, get top 2*limit results with distances.
    3. For each result set, compute RRF score: 1 / (k + rank).
    4. Merge: sum RRF scores for documents appearing in both lists.
    5. Sort by combined score descending, return top limit.
    """
```

The RRF constant `k=60` is standard. This avoids the need to normalize scores between the two very different ranking systems.

### 4.4 Marker Integration

**Task:** Add Marker as a quality processing option in `process.py`.

```
gantry process [FILTER_OPTIONS] --method marker [--workers N] [--json]
```

Or using a convenience flag:

```
gantry process --quality [FILTER_OPTIONS] [--json]
```

`--quality` is syntactic sugar for `--method marker`.

Marker processes PDFs through a vision model pipeline that handles two-column layouts, tables, equations, and figures much better than PyMuPDF4LLM. It's slower (~1 page/sec on M1 Max vs ~8 pages/sec) but produces substantially better markdown for complex academic papers.

Processing with Marker:
1. Call Marker's API to convert PDF to markdown.
2. Store the result in `paper_text.markdown`, overwriting any PyMuPDF4LLM markdown.
3. Update `markdown_method = 'marker'`, `markdown_extracted_at` to now.
4. Re-index the FTS5 table with the improved text.
5. If the document already had embeddings, consider whether to re-embed (flag `stale_embeddings` based on whether text changed significantly).

Marker is an optional dependency (`pip install pdf-gantry[quality]`). The `process.py` module should check for its availability at runtime and give a clear error if `--method marker` is requested but marker isn't installed.

### 4.5 Batch Operations

Phase 2 formalizes the pattern of running operations across filtered document sets:

```bash
# Process all digital PDFs that don't have text yet
gantry process --needs text --is digital

# Re-process everything with Marker for better quality
gantry process --quality --has text

# Embed everything that has text but no embeddings
gantry embed --needs embeddings --has text

# Re-embed documents with stale embeddings (model upgraded)
gantry embed --stale-embeddings
```

Every batch command:
- Accepts `FILTER_OPTIONS` to select documents
- Accepts `--limit N` to cap the batch size (useful for testing)
- Accepts `--dry-run` to report what would be processed
- Accepts `--json` for structured output
- Shows a progress bar in text mode
- Reports success/failure counts on completion
- Continues past individual document failures (logs error, moves on)

### 4.6 Phase 2 Testing Strategy

**test_embeddings.py:**
- Embedding a document produces a 768-d vector
- Vector is stored and retrievable from paper_embeddings
- Batch embedding processes multiple documents
- Re-embedding updates existing vectors
- --stale-embeddings filter works correctly

**test_search.py (additions):**
- Semantic search returns results ordered by similarity
- Hybrid search combines FTS5 and semantic results
- RRF fusion produces reasonable rankings
- Hybrid search handles documents with only FTS5 or only embeddings

**test_process.py (additions):**
- Marker processing produces markdown (if marker installed, else skip)
- --quality flag routes to Marker
- Marker unavailable gives clear error message
- Re-processing updates timestamps and methods

### 4.7 Phase 2 Definition of Done

Phase 2 is complete when:
- `gantry embed` generates embeddings for all documents with text
- `gantry semantic "query"` returns semantically similar documents
- `gantry search "query" --hybrid` combines FTS5 and vector search
- `gantry process --quality` uses Marker for markdown conversion
- Batch operations with filters work across all commands
- Full corpus is embedded (should take ~1-2 minutes)
- All tests pass

---

## 5. Phase 3: Hard Cases + Vault Integration

Phase 3 handles scanned PDFs, enriches metadata from external APIs, and adds read-only vault integration.

### 5.1 OCR Pipeline

**Task:** Implement `ocr.py` and extend `gantry process` to handle scanned PDFs.

```
gantry process --needs ocr [--ocr-method surya] [--json]
```

Surya OCR is the default OCR engine. It handles 90+ languages and produces layout-aware text output with Metal acceleration on M1.

Processing flow for a scanned PDF:
1. The document has `needs_ocr = 1` and `is_scanned = 1` (set during ingest).
2. Run Surya OCR on the PDF.
3. Store the OCR'd text as `raw_text` in `paper_text`.
4. Convert OCR'd text to markdown (basic formatting from Surya's layout detection).
5. Update processing flags: `has_text = 1`, `has_markdown = 1`, `text_method = 'surya'`, `ocr_method = 'surya'`, `ocr_completed_at` to now, `needs_ocr = 0`.
6. Index in FTS5.

Surya is an optional dependency (`pip install pdf-gantry[ocr]`). Runtime check and clear error if missing.

For mixed documents (some pages scanned, some digital), process each page with the appropriate method and concatenate.

### 5.2 Improved Scanned Detection

Refine the Phase 1 heuristic:

```python
def classify_pages(pdf_path: str) -> list[dict]:
    """
    Returns per-page classification:
    [
        {"page": 0, "type": "digital", "text_ratio": 0.72},
        {"page": 1, "type": "scanned", "text_ratio": 0.01},
        ...
    ]
    """
```

This enables mixed-mode processing where digital pages use PyMuPDF and scanned pages use OCR.

### 5.3 Vault Integration

**Task:** Implement `vault.py` and the `gantry vault` command group.

```
gantry vault check [--json]
```

This command reads the Obsidian vault (at `vault_dir`) and cross-references PDFs in the database with source notes in the vault. It is strictly read-only -- gantry never creates, modifies, or deletes vault files.

The check process:
1. Scan the vault for markdown files.
2. In each file, look for references to PDF filenames (in wikilinks like `[[filename.pdf]]`, markdown links, or frontmatter fields like `source: filename.pdf`).
3. For each PDF in the database, record whether a matching vault note was found. Update `vault_note_path` and `vault_checked_at`.

Output:
```
Vault: /Users/ryan/obsidian-vault
PDFs with source notes:  843 / 2,041 (41.3%)
PDFs without source notes: 1,198

Orphaned notes (reference PDFs not in library): 12
  - [[Smith_2020_methodology.pdf]] in research/methods.md
  ...
```

```
gantry vault orphans [--json]
```

Lists vault notes that reference PDFs not found in the library.

```
gantry vault coverage [--json]
```

Lists PDFs without any vault note referencing them -- useful for finding papers you've collected but never written about.

### 5.4 Metadata Enrichment

**Task:** Implement `metadata.py` and the `gantry enrich` command.

```
gantry enrich [FILTER_OPTIONS] [--json]
```

Default filter: `--needs metadata` (documents without DOI or without enriched metadata).

For each document:
1. Attempt to extract DOI from the PDF text (regex for `10.\d{4,}/` patterns).
2. If DOI found, query Semantic Scholar API: `GET https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,authors,year,abstract,citationCount,influentialCitationCount`
3. Store returned metadata in the `papers` record.
4. Respect rate limits (10 req/sec for unauthenticated).

If no DOI is found, attempt a title-based search (extract likely title from first page text, query Semantic Scholar's search endpoint). Mark confidence level in `metadata_source`.

Output:
```
Enriching metadata for 1,198 documents
[################    ] 800/1,198  66.8%  ~4m remaining
  DOI found in text: 623
  Matched via title search: 89
  No match found: 88
  API errors: 0
```

### 5.5 Error Handling and Retry Queues

By Phase 3, some documents will have accumulated errors from earlier phases. Formalize the error/retry system:

```
gantry retry [--max-attempts N] [--json]
```

This command re-processes all documents where `error_count > 0` and `error_count < max_attempts` (default 3). It uses the same processing pipeline as `gantry process` but selects only errored documents.

```
gantry errors [--json]
```

Shows all documents with errors, grouped by error type:

```
12 documents with errors

  Extraction failed (8):
    corrupt_file.pdf: PyMuPDF error: invalid PDF structure
    encrypted_paper.pdf: PyMuPDF error: document is encrypted
    ...

  OCR failed (3):
    blurry_scan.pdf: Surya error: confidence below threshold
    ...

  API error (1):
    Unknown_2019.pdf: Semantic Scholar: 429 Too Many Requests
```

### 5.6 Phase 3 Testing Strategy

**test_ocr.py:**
- OCR a scanned PDF produces text
- Mixed document handles per-page routing
- Surya unavailable gives clear error
- Processing flags updated correctly after OCR

**test_vault.py:**
- Vault check finds matching source notes
- Wikilink patterns are detected
- Markdown link patterns are detected
- Frontmatter source field is detected
- Orphaned notes identified correctly
- Coverage stats are accurate
- Vault integration is truly read-only (no writes to vault_dir)

**test_metadata.py:**
- DOI extraction from PDF text
- Semantic Scholar API integration (mock HTTP responses)
- Title-based fallback search
- Rate limiting respected
- Metadata stored correctly in papers table

**test_queue.py (additions):**
- Retry queue selects correct documents
- Error grouping works
- max_attempts respected

### 5.7 Phase 3 Definition of Done

Phase 3 is complete when:
- `gantry process --needs ocr` runs Surya OCR on scanned PDFs
- `gantry vault check` cross-references PDFs with vault source notes
- `gantry enrich` fetches metadata from Semantic Scholar
- `gantry retry` re-processes failed documents
- `gantry errors` shows error details
- The full corpus (including scanned PDFs) is processed and searchable
- All tests pass

---

## 6. CLI Design Principles

### 6.1 Token-Efficient Output

Default output is concise. One line per result in lists. Summary stats, not verbose logs. A Claude Code session reading gantry output should spend minimal tokens understanding it.

Verbose mode (`--verbose` or `-v`) adds detail: full paths, timestamps, processing metadata. This is for human debugging, not for agents.

### 6.2 JSON Output Mode

Every command supports `--json`. JSON output:
- Is always valid JSON (parseable by `json.loads`)
- Uses consistent key naming (snake_case)
- Includes the same data as text mode, structured
- Writes to stdout (text mode progress bars and chrome go to stderr)

When `--json` is active, suppress all Rich formatting, progress bars, and interactive elements. Only emit the final JSON object to stdout.

### 6.3 Exit Codes

```
0  Success
1  General error (bad config, missing dependencies)
2  No results (search found nothing, queue is empty)
3  Partial failure (some documents in batch failed)
4  Database error
```

Agents can check `$?` after a gantry command and branch accordingly. Exit code 2 is not an error -- it's information. Exit code 3 means "the batch ran, some succeeded, some failed -- check `gantry errors` for details."

### 6.4 Consistent Filter Syntax

Every command that operates on document sets accepts the shared filter options (`--needs`, `--has`, `--is`, `--stale-embeddings`, `--limit`). The options work identically across `queue`, `process`, `embed`, `enrich`, and `retry`.

### 6.5 Command Summary

```
gantry ingest          Scan papers folder, register new/changed PDFs
gantry status          Show index coverage and database stats
gantry process         Extract text and markdown from PDFs
gantry embed           Generate embeddings for documents
gantry search          Full-text search (FTS5), with optional --hybrid
gantry semantic        Semantic similarity search (vector)
gantry queue           Show documents matching a filter
gantry enrich          Fetch metadata from Semantic Scholar
gantry vault check     Cross-reference PDFs with vault source notes
gantry vault orphans   Find vault notes referencing missing PDFs
gantry vault coverage  Find PDFs without vault source notes
gantry errors          Show documents with processing errors
gantry retry           Re-process documents that previously failed
gantry config show     Print current configuration
gantry config set      Update a config value
gantry config init     Interactive first-run setup
```

---

## 7. Implementation Notes for Claude Code Sessions

### Session Workflow

A Claude Code session implementing gantry should:

1. Read this PRD first.
2. Work through one phase at a time. Don't skip ahead.
3. After implementing each module, run the tests for that module.
4. After completing a phase, run the full test suite.
5. Use `gantry --help` and actual CLI invocations to verify behavior.

### Key Dependencies and Imports

```python
# Always needed
import click
import sqlite3
import sqlite_vec
import yaml
import hashlib
from pathlib import Path
from datetime import datetime, timezone

# Phase 1
import fitz  # PyMuPDF
import pymupdf4llm

# Phase 2
from sentence_transformers import SentenceTransformer  # optional
import marker  # optional

# Phase 3
from surya.ocr import run_ocr  # optional
import httpx  # for Semantic Scholar API
```

### sqlite-vec Gotchas

- `sqlite_vec.load(conn)` must be called before any vec0 operations.
- `enable_load_extension(True)` is required first, then disable after loading.
- vec0 virtual tables use `MATCH` for nearest-neighbor queries, not `WHERE distance <`.
- Vectors must be serialized as packed floats: `struct.pack(f'{dim}f', *vector)` or use `sqlite_vec.serialize_float32()`.
- The `k` parameter in vec0 queries controls how many neighbors to return.

### Testing Without the Full Corpus

All tests should work with synthetic fixtures (small PDFs created by PyMuPDF in test setup). Never depend on the user's actual PDF library for tests. The `conftest.py` fixtures should be self-contained.

### Error Handling Pattern

Every processing function should follow this pattern:

```python
def process_document(paper_id: int, conn: sqlite3.Connection) -> bool:
    """Process a single document. Returns True on success, False on failure."""
    try:
        # ... do work ...
        conn.execute(
            "UPDATE papers SET has_text = 1, text_method = ?, ... WHERE id = ?",
            (method, paper_id)
        )
        return True
    except Exception as e:
        conn.execute(
            """UPDATE papers SET
                last_error = ?, error_count = error_count + 1,
                last_error_at = ? WHERE id = ?""",
            (str(e), datetime.now(timezone.utc).isoformat(), paper_id)
        )
        return False
```

Never let a single document failure abort a batch operation. Log it, record it, move on.
