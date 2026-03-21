# Brev.md

Notes for Brev — the agent testing gantry against Ryan's real research library.

## Getting started

```bash
cd ~/Desktop/code/pdf-gantry
source .venv/bin/activate
gantry config init          # set papers_dir and vault_dir
gantry ingest               # scan the corpus
gantry process              # extract text from digital PDFs
gantry embed --chunk        # generate chunk-level embeddings
```

Chunk-level embedding requires `sentence-transformers`:
```bash
uv pip install -e ".[embeddings]" --python .venv/bin/python
```

## Migration from earlier commits

If you have an existing `~/.gantry/index.db` from before the chunk-level embedding work (schema v1), it will auto-migrate on first connection. The migration adds:
- `has_chunk_embeddings` column to `papers` (defaults to 0)
- `chunks` table
- `chunk_vec` vec0 virtual table

**After migration, you must re-run `gantry process`** to populate the `chunks` table. Existing text/markdown in `paper_text` is untouched, but chunks are only generated during processing. You can target specific papers:

```bash
gantry process --has text    # re-process everything that already has text
```

This will regenerate chunks and reset `has_chunk_embeddings = 0`, so follow with `gantry embed --chunk`.

## Non-obvious design choices

**FTS5 is contentless.** The `papers_fts` table does not store text — it only stores the inverted index. This means `snippet()` doesn't work. Search results pull text previews from `paper_text` via JOIN. If you see empty snippets, that's expected for documents where `paper_text.raw_text` is NULL.

**vec0 tables don't support CASCADE.** When re-processing a document, we explicitly `DELETE FROM chunk_vec WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)` before deleting from `chunks`. If you ever manually delete chunks, delete from `chunk_vec` first.

**Scanned/digital classification thresholds** (0.05 / 0.15 in `ingest.py`) were calibrated on synthetic test PDFs, not real academic papers. Expect misclassification on figure-heavy or two-column papers. Check with `gantry queue --is scanned` and spot-check. The threshold for "digital" (0.15 text area ratio) is deliberately low — real papers with embedded text should clear it easily, but sparse title pages or cover sheets may not.

**iCloud dataless files.** On macOS Sonoma+, evicted iCloud files throw `OSError errno 11` (EDEADLK) instead of using `.icloud` stubs. Ingest catches this, skips the file, and prints a `brctl download` command at the end. If you see "X files evicted from iCloud" — that's working as intended. Run the suggested command to force-download, then re-ingest.

**Chunk overlap is applied via string concatenation**, not by re-splitting. The last ~200 chars of chunk N are prepended to chunk N+1 after a word boundary. This means chunk text lengths may slightly exceed `max_chars`. This is intentional — we prioritize semantic coherence over strict size limits.

**`gantry semantic` auto-detects chunk embeddings.** If any papers have `has_chunk_embeddings = 1`, it uses the cascade search (doc-level filter → chunk-level retrieval → top-3 pooling). Use `--doc-only` to force the old doc-level-only behavior. During a transition period where only some papers are chunk-embedded, cascade search still works — it just won't find chunk-level matches for unprocessed papers.

**The cascade boost (0.05)** is a mild score addition for chunks whose parent document appeared in doc-level top-50. It's not a hard filter — chunks from documents outside the top-50 still appear, they just don't get the boost. If you notice retrieval quality issues, this is the first parameter to tune.

**`prepare_chunk_text` prepends metadata at embedding time, not storage time.** The `chunks.text` column stores raw chunk text. The title and section header are prepended when calling the embedding model: `"search_document: {title} | {section} | {text}"`. This means if you later enrich metadata (adding titles via `gantry enrich`), you'd want to re-embed chunks to benefit from the improved context.

## Useful commands for testing

```bash
# How many papers need processing?
gantry queue --needs text --count

# How many are chunk-embedded?
gantry queue --has chunk_embeddings --count

# See chunks for a specific paper
gantry read "some_paper.pdf" --chunks

# Read a specific chunk
gantry read x --chunk 42

# Full-text search
gantry search "adaptation strategies"

# Semantic search (uses chunks if available)
gantry semantic "what papers discuss ROI of coastal infrastructure"

# Hybrid (FTS5 + vectors via RRF)
gantry search "climate policy" --hybrid

# What's the overall state?
gantry status

# JSON everywhere — for your own consumption
gantry status --json
gantry search "climate" --json
gantry semantic "methodology" --json
```

## Exit codes

These matter for your control flow:
- **0**: success
- **1**: error (bad config, missing deps, etc.)
- **2**: no results (search found nothing, queue empty) — not an error
- **3**: partial failure (some docs in batch failed, or iCloud files skipped)
- **4**: database error

Exit code 2 is information, not failure. An empty search result is a valid outcome.
