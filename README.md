# pdf-gantry

Agent-friendly CLI for managing academic PDF libraries. Token-efficient commands for text search, semantic search, PDF processing, and vault integration.

## Overview

`gantry` is a replacement for DEVONthink designed for use with ~2000 academic PDFs stored in a flat iCloud folder. It provides fast, composable CLI commands that work well with AI agents and shell pipelines.

## Commands

```
gantry search <query>      Full-text search across indexed PDFs
gantry semantic <query>    Semantic/embedding-based search
gantry status              Index coverage and error state
gantry process <path>      Process a PDF (text extraction, markdown, embeddings)
gantry queue               Show PDFs pending processing
gantry ingest              Scan papers folder, queue new/changed PDFs
gantry config              Show or set configuration
gantry vault check         Compare PDFs against Obsidian vault source notes (read-only)
```

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

Default paths:
- **Papers:** `~/Library/Mobile Documents/com~apple~CloudDocs/papers/`
- **Index:** `~/.pdf-gantry/`
- **Vault:** *(configure with `gantry config set vault_dir <path>`)*
