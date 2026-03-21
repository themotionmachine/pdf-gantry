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

## Design Direction: Composable Agent Primitives

Gantry's primary role is as a retrieval interface that shapes what an agent can know and how efficiently it can know it. The CLI should evolve toward a *query algebra* — composable operations that let agents chain retrieval steps without full serialization/deserialization round-trips or redundant context at each step.

Current commands are independent. The gap is in composition: an agent today must parse full JSON output from one command to feed IDs into the next. Future primitives should support patterns like:

- **ID-set piping** — `gantry search "X" --ids-only` outputs bare IDs that chain directly into other commands, avoiding full result parsing between steps
- **Batch context retrieval** — "give me the top-scoring chunk from each of these N papers" in one call, not N calls
- **Scoped context windows** — "give me K tokens of context around this chunk" so agents can zoom in without retrieving whole documents
- **Cross-paper queries** — "which of these papers discuss topic X?" as a single filtered operation rather than N searches composed post-hoc

Chunk-level embeddings (#2) are the substrate for this. Once chunks are addressable units, operations can be composed at chunk granularity — retrieve, filter, expand, compare — without agents touching raw PDFs or managing their own context windows. Every new command or flag should be evaluated against this principle: does it admit composition, and does it save tokens?

## Next Steps

All three PRD phases are structurally complete but untested against real data. Anticipated work:

1. **Real corpus testing.** Ingest the ~2000 PDFs and fix edge cases: encrypted PDFs, zero-byte files, exotic encodings, special-character filenames. The iCloud dataless file fix (#1) was the first of these.
2. **Tune scanned/digital classifier.** Current thresholds (0.05/0.15) were calibrated on synthetic PDFs. Two-column layouts, figure-heavy papers, and sparse title pages will likely misclassify. Spot-check with `gantry queue --is scanned`.
3. **Test optional dependencies end-to-end.** Embeddings (sentence-transformers + Nomic), Marker, and Surya are wired up but haven't run on real data with real models. Expect integration issues around model downloads, memory, and Metal acceleration.
4. **Search quality iteration.** Whole-document embeddings lose nuance for long papers — may need text chunking. RRF k parameter may need tuning. Possible new commands: `gantry read <filename>` to dump markdown, `gantry find <fragment>` for fuzzy filename lookup.
5. **Workflow integration.** Test usage from Claude Code sessions. May surface needs for richer `--json` output, a `gantry summary` command, or other agent-oriented features.
6. **Vault integration polish.** Real vaults have aliases, nested folders, varied reference conventions. May need refined reference extraction and a `gantry vault suggest` command for papers without notes.
