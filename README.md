# pdf-gantry

A CLI that turns a folder of academic PDFs into a SQLite index an AI agent can search without blowing its context window.

I built gantry for my own library: about 2,000 papers in one flat iCloud folder. Agents read papers fine. The expensive part is deciding *what* to read, and most retrieval setups make the agent pay for that decision in tokens — full search results parsed at every step, whole documents fetched to check one section. Gantry treats retrieval as the interface that shapes what an agent can know and how cheaply it can know it. Every command returns structured output, trims to the fields you ask for, and composes with the next command through bare IDs.

No server, no cloud, no framework. One SQLite file (with FTS5 and [sqlite-vec](https://github.com/asg017/sqlite-vec)), a folder of PDFs, and a shell.

## What an agent session looks like

Find candidate papers, paying only for IDs:

```bash
$ gantry search "predictive processing" --fields id --json -n 5
{
  "query": "predictive processing",
  "total": 5,
  "results": [{"id": 942}, {"id": 1112}, {"id": 999}, {"id": 1616}, {"id": 182}]
}
```

Then pull the single best-matching chunk from each of those papers in one call. Not N searches, and no full documents:

```bash
$ gantry info --ids 942,1112,999 --query "prediction error minimization" \
    --fields id,filename,top_chunk --json
{
  "count": 3,
  "papers": [
    {
      "id": 942,
      "filename": "Laukkonen & Slagter 2021.pdf",
      "top_chunk": {
        "chunk_id": 76606,
        "section_header": "**2. Predictive processing**",
        "score": 0.4675,
        "text": "the free energy principle is founded that has made it so appealing..."
      }
    },
    ...
  ]
}
```

Zoom in on a chunk with a fixed token budget instead of fetching the paper:

```bash
$ gantry read 942 --chunk 76606 --context 2000
```

Each step passes an address (a paper ID, a chunk ID). Payloads only move when the agent asks for them. The agent decides how deep to go, and shallow is cheap.

## Quickstart

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/themotionmachine/pdf-gantry.git
cd pdf-gantry
uv venv --python 3.11
uv pip install -e ".[all]" --python .venv/bin/python

.venv/bin/gantry config init    # point it at your PDF folder
.venv/bin/gantry pipeline       # ingest → extract text → embed
.venv/bin/gantry search "your first query"
```

The base install covers ingestion, extraction, and keyword search. Heavier features are opt-in extras:

```bash
uv pip install -e ".[embeddings]"  # semantic + hybrid search (sentence-transformers)
uv pip install -e ".[ocr]"        # scanned PDFs (Surya)
uv pip install -e ".[quality]"    # higher-quality extraction (Marker)
uv pip install -e ".[all]"        # everything
```

## The agent contract

Gantry holds to a few rules so that agents (and scripts) can rely on it:

- **`--json` on every command.** Structured output goes to stdout; progress bars and chatter go to stderr. Pipes stay clean.
- **`--fields` trims payloads.** Ask for `id,filename,top_chunk` and that is all you get. Tokens are the budget; spend them on content.
- **Exit codes carry meaning.**

  | Code | Meaning |
  |------|---------|
  | 0 | success |
  | 1 | error |
  | 2 | ran fine, no results |
  | 3 | partial failure (some documents succeeded) |
  | 4 | database error |

  An agent can branch on "no results" without parsing anything.
- **State is queryable.** Processing status lives in boolean columns (`has_text`, `has_embeddings`, `needs_ocr`), so `gantry queue --needs embeddings` answers "what work is left?" in one call.

If you point an agent at gantry, a system-prompt note like this is enough: *"You have `gantry` for searching a local paper library. Use `gantry search <query> --fields id --json` to find papers, `gantry info --ids <ids> --query <topic> --json` to get each paper's most relevant passage, and `gantry read <id> --chunk <chunk_id> --context 2000` to expand. Exit code 2 means no results."*

## Why not Zotero, or a RAG framework?

Zotero manages references; it does not give an agent chunk-level retrieval over full text. RAG frameworks give you retrieval but bring a server, an orchestration layer, and their own opinions about your agent loop. Gantry is the thin middle: your files stay where they are, the index is one SQLite file at `~/.gantry/index.db`, and the interface is a shell command any agent can already call. If you stop using it, you delete one file.

## Commands

### Ingestion and processing

| Command | Description |
|---------|-------------|
| `gantry ingest` | Scan the papers folder, register new and changed PDFs |
| `gantry process` | Extract text and markdown |
| `gantry embed` | Generate chunk-level embeddings |
| `gantry ocr` | OCR scanned PDFs with Surya |
| `gantry enrich` | Fetch metadata from Semantic Scholar |
| `gantry pipeline` | Run the full chain: ingest → process → embed |

### Search and retrieval

| Command | Description |
|---------|-------------|
| `gantry search <query>` | Hybrid search (FTS5 + vector, fused with RRF); `--fts` for keyword-only |
| `gantry semantic <query>` | Pure vector similarity search |
| `gantry find <fragment>` | Fuzzy filename lookup |
| `gantry read <id>` | Read a document's text, list its chunks, or expand one chunk with `--context` |
| `gantry info --ids <ids>` | Metadata for specific papers; `--query` attaches each paper's best-matching chunk |

### Index management

| Command | Description |
|---------|-------------|
| `gantry status` | Coverage and database stats |
| `gantry queue` | Documents matching a filter (`--needs`, `--has`, `--is`) |
| `gantry errors` / `gantry retry` | Inspect and re-run failures |
| `gantry prune` | Drop entries for files no longer on disk |

### Bibliography and vault

| Command | Description |
|---------|-------------|
| `gantry link init <path>` | Generate a `.bib` file from enriched metadata |
| `gantry link check <bib>` | Reconcile PDFs against a BibTeX file, assign citekeys |
| `gantry vault check` | Cross-reference papers against an Obsidian vault (read-only) |

## Configuration

```bash
gantry config init          # interactive setup
gantry config show
gantry config set papers_dir ~/Papers
```

Config lives at `~/.gantry/config.yaml`; `GANTRY_*` environment variables override it (`GANTRY_PAPERS_DIR`, `GANTRY_INDEX_DIR`, `GANTRY_VAULT_DIR`). The two settings that matter: `papers_dir`, any flat folder of PDFs, and optionally `vault_dir` for Obsidian cross-referencing.

## How it works

- **Change detection:** PDFs are SHA-256 hashed at ingest; only new or changed files are reprocessed.
- **Extraction:** PyMuPDF4LLM by default, Marker as an optional higher-quality backend.
- **Search:** contentless FTS5 for keywords, 768-d Nomic Embed V2 vectors in sqlite-vec for semantics, reciprocal rank fusion for hybrid. Hybrid degrades gracefully to FTS if the embedding model is unavailable.
- **Chunks:** documents are split into addressable chunks with per-chunk embeddings, so retrieval can land on a passage instead of a paper.
- **Scanned PDFs:** classified at ingest and routed to the OCR queue.

## Status

I use gantry daily against one real corpus of ~2,000 papers. That is the extent of battle-testing, so calibrate accordingly:

- The scanned-vs-digital classifier was tuned on synthetic PDFs. It works on my corpus; figure-heavy or two-column papers may misclassify on yours.
- The OCR path is implemented and unit-tested, but I have not yet validated it end-to-end against real scanned PDFs with the production Surya models.
- Vault integration handles common Obsidian conventions (wikilinks, frontmatter source fields). Unusual vault layouts may need work.

Bug reports with a problem PDF attached are the most useful thing you can send.

## Development

Tests are mandatory here; the project runs red/green TDD.

```bash
uv pip install -e ".[dev]" --python .venv/bin/python
.venv/bin/python -m pytest tests/ -v
.venv/bin/ruff check src/ tests/
```

## License

MIT
