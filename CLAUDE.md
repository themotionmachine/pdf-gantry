# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

pdf-gantry is an agent-friendly CLI (`gantry`) for managing ~2000 academic PDFs in a flat iCloud folder. It indexes PDFs into a single SQLite database (with sqlite-vec for vectors), extracts text/markdown, generates embeddings, and provides FTS5 + semantic search. Read-only Obsidian vault integration cross-references source notes.

## Commands

```bash
# Install (uses uv)
uv venv --python 3.11
uv pip install -e ".[dev]" --python .venv/bin/python

# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run a single test
.venv/bin/python -m pytest tests/test_search.py::test_fts_search_returns_results

# Lint
.venv/bin/ruff check src/ tests/

# CLI
.venv/bin/gantry --help
```

## Architecture

Source layout: `src/pdf_gantry/` with flat modules (no sub-packages).

Entry point: `gantry` → `pdf_gantry.cli:cli()` (Click group).

**Core modules:**
- `db.py` — SQLite connection with sqlite-vec extension, WAL mode, schema creation/migration. Single DB at `~/.gantry/index.db`
- `config.py` — YAML config at `~/.gantry/config.yaml`. Dataclass-based with env var overrides (`GANTRY_*`)
- `ingest.py` — Scans papers_dir, registers PDFs, computes SHA-256 hashes, classifies scanned vs digital
- `process.py` — Text/markdown extraction via PyMuPDF4LLM; populates `paper_text` table and FTS5 index. Uses `ProcessPoolExecutor` for concurrency
- `search.py` — FTS5 search, semantic search (sqlite-vec), hybrid search (RRF fusion)
- `embeddings.py` — Nomic Embed V2 via sentence-transformers; stores 768-d vectors in `vec0` table
- `queue.py` — `build_filter_query()` maps `--needs`/`--has`/`--is` flags to SQL WHERE clauses. Used by all batch commands
- `vault.py` — Read-only vault check: parses wikilinks, markdown links, frontmatter source fields
- `metadata.py` — DOI extraction + Semantic Scholar API lookups
- `ocr.py` — Surya OCR for scanned PDFs (optional dependency)

**Key design pattern:** The `papers` table carries boolean processing state columns (`has_text`, `has_markdown`, `has_embeddings`, `needs_ocr`). Queue/filter queries are trivial: `WHERE has_text = 0`. Every batch command uses shared `filter_options` Click decorator.

**FTS5 is contentless** — gantry manages inserts/deletes explicitly. Can't use `snippet()` function; text previews come from `paper_text` table via JOIN.

**Exit codes:** 0=success, 1=error, 2=no results, 3=partial failure, 4=db error.

**Every command supports `--json`** for structured output (to stdout; progress bars go to stderr).
