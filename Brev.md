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

Chunk-level embedding requires `sentence-transformers` and `einops`:
```bash
uv pip install -e ".[embeddings]" --python .venv/bin/python
```

(Both are included in the `[embeddings]` extra. If you hit an `einops` import error, you're on an older install — re-run the command above.)

## Migration from earlier commits

The database auto-migrates on first connection. Current schema is **v3**.

**From v1 (pre-chunk):** Adds `has_chunk_embeddings` column, `chunks` table, `chunk_vec` table. You must re-run `gantry process --has text` to populate chunks, then `gantry embed --chunk`.

**From v2 (pre-cosine fix):** Recreates `paper_embeddings` and `chunk_vec` tables with `distance_metric=cosine`. **Existing vectors are preserved** — they're read out, tables recreated, vectors re-inserted. No re-embedding needed. This fixes `score_vector` values: previously they were `1 - L2_distance` (compressed, often negative), now they're true cosine similarity (0–1 range, 0.7+ means strong match).

**After any migration, verify:**
```bash
gantry status               # check counts
gantry process --has text    # re-process to generate chunks (if coming from v1)
gantry embed --chunk         # embed chunks (if coming from v1 or v2)
```

## Non-obvious design choices

**FTS5 is contentless.** The `papers_fts` table does not store text — it only stores the inverted index. This means `snippet()` doesn't work. Search results pull text previews from `paper_text` via JOIN. If you see empty snippets, that's expected for documents where `paper_text.raw_text` is NULL.

**vec0 tables don't support CASCADE.** When re-processing a document, we explicitly `DELETE FROM chunk_vec WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)` before deleting from `chunks`. If you ever manually delete chunks, delete from `chunk_vec` first.

**Scanned/digital classification thresholds** (0.05 / 0.15 in `ingest.py`) were calibrated on synthetic test PDFs, not real academic papers. Expect misclassification on figure-heavy or two-column papers. Check with `gantry queue --is scanned` and spot-check. The threshold for "digital" (0.15 text area ratio) is deliberately low — real papers with embedded text should clear it easily, but sparse title pages or cover sheets may not.

**iCloud dataless files.** On macOS Sonoma+, evicted iCloud files throw `OSError errno 11` (EDEADLK) instead of using `.icloud` stubs. Ingest catches this, skips the file, and prints a `brctl download` command at the end. If you see "X files evicted from iCloud" — that's working as intended. Run the suggested command to force-download, then re-ingest.

**Chunk overlap is applied via string concatenation**, not by re-splitting. The last ~200 chars of chunk N are prepended to chunk N+1 after a word boundary. This means chunk text lengths may slightly exceed `max_chars`. This is intentional — we prioritize semantic coherence over strict size limits.

**`gantry semantic` auto-detects chunk embeddings.** If any papers have `has_chunk_embeddings = 1`, it uses the cascade search (doc-level filter → chunk-level retrieval → top-3 pooling). Use `--doc-only` to force the old doc-level-only behavior. During a transition period where only some papers are chunk-embedded, cascade search still works — it just won't find chunk-level matches for unprocessed papers.

**The cascade boost (0.05)** is a mild score addition for chunks whose parent document appeared in doc-level top-50. It's not a hard filter — chunks from documents outside the top-50 still appear, they just don't get the boost. If you notice retrieval quality issues, this is the first parameter to tune.

**`prepare_chunk_text` prepends metadata at embedding time, not storage time.** The `chunks.text` column stores raw chunk text. The title and section header are prepended when calling the embedding model: `"search_document: {title} | {section} | {text}"`. This means if you later enrich metadata (adding titles via `gantry enrich`), you'd want to re-embed chunks to benefit from the improved context.

**Hyphenated search terms.** FTS5 misparses hyphens as column filters (`cross-national` → `no such column: national`). Gantry sanitizes these automatically — hyphens become spaces before hitting MATCH. This means `cross-national` and `cross national` produce the same results. If you need an exact hyphenated match, it won't find one (FTS5 doesn't support this natively). This is an acceptable tradeoff.

**Hybrid search RRF scores are ordinal, not absolute.** The fused `score` from `--hybrid` lives in a narrow ~0.016–0.033 band and is meaningless for thresholding or cross-query comparison. If you need absolute relevance signals, use `--components` to get the raw `score_fts` (BM25) and `score_vector` (cosine similarity). Cosine similarity is on a fixed 0–1 scale and survives cross-query comparison. BM25 scores are corpus-relative but still more informative than RRF.

**`--fields` filters the result objects, not the top-level response.** `gantry search "X" --json --fields id,score` still returns `{"query": ..., "total": ..., "results": [{id, score}, ...]}`. The `query` and `total` wrapper fields are always present. Only the per-result objects are filtered.

**`gantry info` is for targeted follow-up, not discovery.** It takes paper IDs (from a prior search) and returns metadata without re-running search. The pattern: run a broad `--fields id,score` search first, filter IDs in your logic, then `gantry info --ids 1,2,3 --json` for the survivors. Add `--chunks` if you need the actual chunk texts for those papers.

## Composable search patterns

For multi-step workflows, here's how the pieces fit together:

```bash
# 1. Broad pass — IDs and scores only
gantry search "bounded rationality" --hybrid --json --fields id,score,filename

# 2. With component scores — identify semantic-only vs keyword-only hits
gantry search "bounded rationality" --hybrid --json --components --fields id,score_fts,score_vector

# 3. Targeted fetch for papers that survived your filtering
gantry info --ids 142,587,923 --json --fields filename,title,snippet

# 4. Deep dive with chunk texts
gantry info --ids 142 --json --chunks

# 5. Read a specific chunk directly
gantry read x --chunk 47

# 6. Expand context around a chunk (scoped window)
gantry read x --chunk 47 --context 4000

# 7. Fuzzy filename lookup (don't need the exact name)
gantry find "Friston" --json
```

The key insight: `--fields` makes early passes cheap, `gantry info` makes follow-up targeted, `--components` makes cross-query reasoning possible, and `--context` lets you zoom in without pulling the whole document. You don't need to pull full snippets until you know which papers you care about.

**`gantry find` resolves partial filenames.** If you have a fragment like "Friston" or "2019_climate", `gantry find` does a case-insensitive LIKE match against all indexed filenames. Use this when you know roughly what paper you want but don't have the exact filename. Returns IDs you can feed into `gantry info` or `gantry read`.

**`--context` expands outward from a chunk.** `gantry read x --chunk 47 --context 4000` gives you the target chunk plus neighboring chunks, expanding alternately before and after, until the character budget is reached. The target chunk is always included even if it exceeds the budget. Use this after a semantic search surfaces a relevant chunk — you can see what comes before and after without reading the whole document.

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

# Expand context around a chunk
gantry read x --chunk 42 --context 4000

# Fuzzy filename lookup
gantry find "Friston"
gantry find "2019" --json

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
