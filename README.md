# pdf-gantry

Agent-friendly CLI for indexing and searching academic PDF libraries. Designed to work well with AI agents, shell pipelines, and structured output (`--json` on every command).

## What it does

Point gantry at a folder of PDFs and it builds a SQLite index with full-text search (FTS5), semantic search (sqlite-vec embeddings), and metadata enrichment. It handles the full pipeline — ingestion, text extraction, OCR for scanned papers, chunk-level embeddings, and optional Obsidian vault cross-referencing.

## Quickstart

```bash
# Clone and install (requires Python 3.11+)
git clone https://github.com/themotionmachine/pdf-gantry.git
cd pdf-gantry
uv venv --python 3.11
uv pip install -e ".[all]" --python .venv/bin/python

# Configure your papers folder
.venv/bin/gantry config init

# Index everything
.venv/bin/gantry pipeline
```

The `pipeline` command runs the full chain: ingest → process → embed. You can also run each step individually.

### Optional dependencies

The base install handles ingestion, text extraction, and search. Heavier features are opt-in:

```bash
uv pip install -e ".[embeddings]"  # Semantic search (sentence-transformers)
uv pip install -e ".[ocr]"        # OCR for scanned PDFs (Surya)
uv pip install -e ".[quality]"    # Higher-quality extraction (Marker)
uv pip install -e ".[all]"        # Everything
```

## Commands

### Ingestion & processing

| Command | Description |
|---------|-------------|
| `gantry ingest` | Scan papers folder, register new/changed PDFs |
| `gantry process` | Extract text and markdown from PDFs |
| `gantry embed` | Generate chunk-level embeddings |
| `gantry ocr` | Run OCR on scanned PDFs using Surya |
| `gantry enrich` | Fetch metadata from Semantic Scholar |
| `gantry pipeline` | Run the full chain: ingest → process → embed |

### Search & retrieval

| Command | Description |
|---------|-------------|
| `gantry search <query>` | Full-text search (FTS5) |
| `gantry semantic <query>` | Semantic similarity search |
| `gantry find <query>` | Fuzzy filename lookup |
| `gantry read <id>` | Read document text or chunks |
| `gantry info <id>` | Fetch metadata for specific papers |

### Index management

| Command | Description |
|---------|-------------|
| `gantry status` | Index coverage and database stats |
| `gantry queue` | Show documents matching a filter (`--needs`, `--has`, `--is`) |
| `gantry errors` | Show documents with processing errors |
| `gantry retry` | Re-process previously failed documents |
| `gantry prune` | Remove entries for files no longer on disk |

### Configuration & integration

| Command | Description |
|---------|-------------|
| `gantry config` | Show, set, or initialize configuration |
| `gantry vault check` | Cross-reference PDFs against an Obsidian vault (read-only) |

Every command supports `--json` for structured output to stdout (progress goes to stderr), making gantry straightforward to use from scripts or AI agents.

## Configuration

```bash
gantry config init          # Interactive first-run setup
gantry config show          # Print current config
gantry config set key val   # Set a value
```

Config lives at `~/.gantry/config.yaml`. Key settings:

- **`papers_dir`** — path to your PDF folder (any flat directory works)
- **`vault_dir`** — path to an Obsidian vault (optional, for `vault check`)

The SQLite index is stored at `~/.gantry/index.db`.

## How it works

- **Indexing:** PDFs are SHA-256 hashed for change detection. Text and markdown are extracted via PyMuPDF4LLM, with Marker as an optional higher-quality backend.
- **Search:** FTS5 for keyword search, sqlite-vec for semantic search (Nomic Embed V2, 768-d vectors), and hybrid search using reciprocal rank fusion.
- **Chunks:** Documents are split into addressable chunks with per-chunk embeddings, enabling fine-grained retrieval without reading entire papers.
- **Scanned PDFs:** Automatically classified during ingestion. Use `gantry ocr` with Surya to extract text before embedding.
- **Vault integration:** Read-only — gantry checks which papers have corresponding Obsidian notes but never writes to your vault.

## Development

```bash
uv pip install -e ".[dev]" --python .venv/bin/python

# Run tests
.venv/bin/python -m pytest tests/ -v

# Lint
.venv/bin/ruff check src/ tests/
```

## License

MIT
