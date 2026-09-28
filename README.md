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
  "returned": 5,
  "mode": "hybrid",
  "results": [{"id": 942}, {"id": 1112}, {"id": 999}, {"id": 1616}, {"id": 182}]
}
```

When the agent needs enough to judge the hits but not a full JSON payload, `--format oneline` prints one tab-separated line per hit: id, score, year, citekey, title (the filename when there is no title):

```bash
$ gantry search "Deepfakes and Social Media: Implications" --fts --format oneline -n 3
1014	1			Mashinini 2020.pdf
499	0.98	2020		Deepfakes and Disinformation: Exploring the Impact of Synthetic Political Video on Deception, Uncertainty, and Trust in News
347	0.98	2021		AI and the Future of Disinformation Campaigns: Part 2: A Threat Model
```

Paste titles as they are. Query text is literal by default, so colons, parentheses, apostrophes and `C++` are safe. `"quoted phrases"`, `word*` prefixes and an `AND`/`OR`/`NOT` between two terms still work. Pass `--fts-syntax` if you want raw FTS5 syntax such as column filters or `NEAR`.

Or skip JSON entirely and emit bare ranked IDs, one per line, to pipe straight into the next step:

```bash
$ gantry search "predictive processing" --ids-only -n 5
942
1112
999
1616
182
```

Loading the embedding model takes about 12 s, so don't loop over queries in the shell. Put them in a file (or on stdin with `-`) and the model loads once. On the real ~2,000-paper index, 20 hybrid queries took 15 s this way, against about 12 s for each separate call:

```bash
$ printf '%s\n' "deepfakes elections" "content moderation" | gantry search --queries-file - --format oneline -n 3
0	1014	0.0328			Mashinini 2020.pdf
0	500	0.032			Deepfakes and the New Disinformation War- The Coming Age of Post-Truth Geopolitics.pdf
0	944	0.0318			Law of Ukraine on AI Technologies.pdf
1	650	0.0325	2024		Decentralised content moderation
1	1304	0.032			SSRN-id4213674.pdf
1	531	0.031	2021		Do Platform Migrations Compromise Content Moderation? Evidence from r/The_Donald and r/Incels
```

With `--json` this returns `{"queries": [{"query", "mode", "total", "returned", "results", "not_found"}, ...]}`. With `--format oneline`, each line starts with the 0-based query index. `--ids-only` prints the de-duplicated union of hits, in the order they were first seen.

To ask "which of *these* papers discuss X?", scope a search to a candidate set instead of searching globally and filtering client-side:

```bash
$ gantry search "free energy" --restrict-to-ids 942,1112,999 --ids-only
942
999
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

Check that a quote is verbatim in a paper, and get its page for the citation:

```bash
$ gantry grep "the predicative self-identification" --ids 1902 --json
{
  "query": "the predicative self-identification",
  "count": 2,
  "truncated": false,
  "hits": [
    {
      "doc_id": 1902,
      "filename": "Habermas1987.pdf",
      "chunk_id": 152773,
      "chunk_index": 229,
      "page_start": 92,
      "page_end": 93,
      "offset": 506,
      "length": 34,
      "matched_text": "the predicative selfidentification",
      "match": "normalized",
      "source": "chunks",
      "context": "...I argue for the following thesis: the predicative selfidentification that a person undertakes is..."
    },
    ...
  ],
  "not_found": []
}
```

`match` says whether the text was found as typed (`exact`) or only after normalising line breaks, line-end hyphenation, ligatures, markdown emphasis and curly quotes (`normalized`); `matched_text` is the span as it appears in the source. `--ids` takes `search --ids-only` output as is, or `-` to read it from stdin.

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
- **`--fields` trims payloads.** Ask for `id,filename,top_chunk` and that is all you get. Tokens are the budget; spend them on content. An unknown field name gets a warning on stderr that lists the valid names; it is not dropped silently.
- **One shape per concept.** `authors` is always a JSON list of names (`[]` when unknown) in `search`, `semantic`, `info`, `find` and `read`. Search results carry `title`, `year`, `authors` and `citekey`.
- **Counts say what they count.** In `search`/`semantic` JSON, `returned` is the number of results in the response. `total` is the global FTS5 match count for `search --fts` without `--restrict-to-ids`, regardless of `-n`. Hybrid and semantic search rank rather than match, so there `total` equals `returned`. The `mode` field reports the retrieval path that actually ran: `hybrid`, `vector_fallback` (FTS5 rejected the query, so the results are vector-only), `fts` (embeddings unavailable), `fts_only` (`--fts`), or, for `semantic`, `cascade` or `doc`.
- **Exit codes carry meaning.**

  | Code | Meaning |
  |------|---------|
  | 0 | success |
  | 1 | error |
  | 2 | ran fine, no results |
  | 3 | partial failure (some documents succeeded, or some requested IDs/indices were missing) |
  | 4 | database error |
  | 64 | usage error: unknown option or command, bad choice, missing argument |

  An agent can branch on "no results" without parsing anything. A typo is never exit 2: usage errors exit 64, and when `--json` appears anywhere on the command line they print `{"error": "...", "usage": "...", "exit_code": 64}` on stdout.
- **Stable JSON envelopes.** Keys are only ever added, never renamed. Errors from any command under `--json` are `{"error": "..."}`. Paper-level records carry `id` (the paper ID); `read` also keeps its older `paper_id`, and chunk records carry both `id`/`paper_id` and the chunk's own `chunk_id`.

  | Command | Envelope |
  |---------|----------|
  | `search`, `semantic` | `{query, total, mode (search only), results: [{id, filename, score, snippet, citekey, …}], not_found}` |
  | `find` | `{fragment, count, results: [{id, filename, title, year, citekey, page_count, has_text, matched_fields}]}` |
  | `info` | `{count, papers: [{id, title, authors, year, doi, citekey, …, top_chunk?, chunks?}], not_found}` |
  | `read ID` | `{id, paper_id, filename, title, text}` |
  | `read ID --chunks` | `{id, paper_id, filename, chunks: [{chunk_id, chunk_index, section_header, text_length}]}` |
  | `read ID --index A-B` | `{id, paper_id, filename, title, total_chunks, chunks: [{chunk_id, chunk_index, section_header, char_offset, char_end, page_start, text}], missing_indices}` |
  | `read ID --chunk C` | `{id, paper_id, doc_id, chunk_id, chunk_index, section_header, page_start, filename, title, text}`; with `--context` adds `context`, `total_chunks` and names the chunk text `chunk_text` |
  | `queue` | `{count, documents: [...]}` |
  | `errors` | `{count, errors: [...]}` |
  | `schema` | `{schema_version, database, tables, views, relationships, common_joins}` |
- **State is queryable.** Processing status lives in boolean columns (`has_text`, `has_embeddings`, `needs_ocr`), so `gantry queue --needs embeddings` answers "what work is left?" in one call.

If you point an agent at gantry, a system-prompt note like this is enough: *"You have `gantry` for searching a local paper library. Use `gantry search <query> --ids-only` to find papers (bare IDs, one per line; add `--restrict-to-ids <ids>` to scope a search to a candidate set), `gantry info --ids <ids> --query <topic> --json` to get each paper's most relevant passage, `gantry find "<title or author>"` for a known paper, and `gantry read <id> --index <a-b>` (or `--chunk <chunk_id> --context 2000`) to read passages. Exit code 2 means no results; 64 means the command itself was malformed."*

## Why not Zotero, or a RAG framework?

Zotero manages references; it does not give an agent chunk-level retrieval over full text. RAG frameworks give you retrieval but bring a server, an orchestration layer, and their own opinions about your agent loop. Gantry is the thin middle: your files stay where they are, the index is one SQLite file at `~/.gantry/index.db`, and the interface is a shell command any agent can already call. If you stop using it, you delete one file.

## Commands

### Ingestion and processing

| Command | Description |
|---------|-------------|
| `gantry ingest` | Scan the papers folder, register new and changed PDFs |
| `gantry process` | Extract text and markdown |
| `gantry embed` | Generate chunk-level embeddings; `--ids <ids> --force` re-embeds specific papers (deletes their old vectors and resets the flag) |
| `gantry ocr` | OCR scanned PDFs with Surya |
| `gantry enrich` | Fetch metadata (title, authors, year, DOI, abstract) from OpenAlex or Semantic Scholar |
| `gantry pipeline [FILE...]` | Run the full chain: ingest → process → embed. `--enrich` adds metadata lookup before embedding. Positional files (or `--file`) scope it to those PDFs: `gantry pipeline paper.pdf --enrich --json` adds one paper end to end and reports its `ids` |
| `gantry meta set\|clear\|normalize` | Hand metadata edits in code rather than raw SQL (see below) |

#### Enriching metadata

Most flat PDF folders carry almost no metadata, so `gantry enrich` backfills it from a scholarly index and writes it onto each paper (`title`, `authors`, `year`, `doi`, `abstract`). With metadata in place, `gantry link init` can generate a real `.bib`.

It defaults to **OpenAlex**: no API key, broad coverage, and a "polite pool" that runs faster when requests carry your email. Set it once and every run uses it:

```bash
gantry config set openalex_mailto you@example.com
gantry enrich --dry-run        # how many papers would be touched
gantry enrich --limit 20       # small batch to eyeball quality first
gantry enrich                  # the whole library
```

For each paper it tries, in order: exact DOI lookup (from the `doi` column, else a DOI found in the first page of text), then an author+year search derived from the filename, then a title search. DOI matches are reliable; title-search fallbacks on opaque filenames are worth a skeptical pass.

Semantic Scholar is still available with `--provider semantic-scholar`, but without an API key it rate-limits hard, which is why OpenAlex is the default. By default `enrich` only touches papers it has never tried; pass `--ids`, `--has`/`--needs`/`--is` filters or `--limit` to scope it.

A paper the provider can't match is recorded as `metadata_source = 'none:<provider>'`, so "tried and missed" is distinguishable from "never tried". Misses aren't retried by default; `gantry enrich --retry-misses` re-attempts them, and `gantry queue --is enrich-miss` lists them. A lookup that fails on a network or API error leaves the paper untouched, so the next run tries it again.

`gantry pipeline --enrich` runs enrichment after text extraction and before embedding, because each chunk is embedded with its paper's title prepended. Its JSON reports `enriched`, an `enrich` block of counts, and `titled_after_embed`: papers that were already embedded before they got a title, ready for `gantry embed --chunk --ids … --force`.

#### Editing metadata by hand

When a provider gets a paper wrong, or has nothing, set the metadata with `gantry meta` rather than SQL:

```bash
gantry meta set --id 53 --title "Mind Games" --authors "Ann Author; Ben Author" \
    --year 2021 --citekey author2021mind --by ryan
gantry meta clear --ids 861,862          # null a wrong match so enrich retries it
gantry meta normalize --dry-run --json   # report repairs to hand-written rows
```

`meta set` writes only the fields you pass, stores authors as a JSON list (`"A; B"` or a JSON list both work), stamps ISO timestamps, and records `metadata_source = 'manual:<by>'`. `enrich` never overwrites manual metadata, even with `--ids`; `gantry queue --is manual-metadata` lists it. `meta clear` nulls title, authors, year, DOI and abstract, resets the enrich and verify state, and leaves the citekey alone unless you pass `--citekey`. `meta normalize` is idempotent: it converts `;`-separated authors to JSON lists, converts `YYYY-MM-DD HH:MM:SS` timestamps to ISO, clears abstracts left on untitled `metadata_suspect` papers (it only reports other untitled papers with abstracts unless you pass `--all-orphans`), and tags pre-2026-09 misses as `none:legacy`. All three take `--dry-run` and `--json`.

### Search and retrieval

| Command | Description |
|---------|-------------|
| `gantry search <query>` | Hybrid search (FTS5 + vector, fused with RRF); `--fts` for keyword-only, `--fts-syntax` for raw FTS5 syntax. `--ids-only` emits bare ranked IDs for piping; `--format oneline` emits `id score year citekey title`; `--restrict-to-ids` scopes the search to a candidate set; `--queries-file PATH\|-` runs many queries with one model load |
| `gantry semantic <query>` | Pure vector similarity search; also supports `--ids-only`, `--format oneline`, `--restrict-to-ids` and `--queries-file` |
| `gantry find <words>` | Known-item lookup over title, authors, year, citekey, DOI and filename: `find "Mind games"`, `find "Zhang 2022"`, `find "Flew & Martin"`. Every word must match some field; title matches rank first; each result reports `matched_fields`. `--ids-only` for piping |
| `gantry read <id>` | Read a document's text, list its chunks (`--chunks`), read chunks by per-paper position (`--index 28-41`), or read one chunk by `chunk_id` (`--chunk`, which must belong to that paper) and expand it with `--context` |
| `gantry info <ids…>` | Metadata for specific papers; takes IDs (`info 12 13`, `info 12,13`), filenames or citekeys (`info @smith2020`), or `--ids`. `--query` attaches each paper's best-matching chunk |
| `gantry schema` | Tables, columns, relationships and common joins, for raw-SQL callers |
| `gantry grep "<text>"` | Find a literal string (a quote) in chunk text; each hit gives doc, chunk, `page_start`/`page_end`, offset and whether the match was exact or normalised. `--ids`, `-i/--ignore-case`, `--limit`, `--context`. Papers without chunks are searched in raw text |

`<id>` in `read` and `info` accepts a paper ID, a filename, or a citekey (with or without `@`).

**Raw SQL.** If you query `~/.gantry/index.db` directly, use the views `v_papers` and `v_chunks`. Both are keyed by `paper_id` (the underlying `chunks` table calls it `doc_id`, and a chunk's key is `chunk_id`, not `id`). `v_chunks` exposes `paper_id, chunk_id, chunk_index, section_header, page_start, text, char_offset`. Full text lives in `paper_text.raw_text` / `paper_text.markdown`. `gantry schema` prints all of this.

### Index management

| Command | Description |
|---------|-------------|
| `gantry status` | Coverage and database stats, plus pending-work counts (`needs_text`, `needs_embeddings`, `needs_chunk_embeddings`, `needs_enrich`, `enrich_misses`, `metadata_suspect`, `manual_metadata`) matching what the next default run would pick up |
| `gantry queue` | Documents matching a filter (`--needs`, `--has`, `--is`) |
| `gantry errors` / `gantry retry` | Inspect and re-run failures |
| `gantry queue --is broken` | Papers quarantined after too many failures (`error_count >= processing.max_retries`, default 3) |
| `gantry retry --ids <ids>` | Clear a quarantined paper's error count and re-process it |
| `gantry prune` | Drop entries for files no longer on disk |
| `gantry fts rebuild` | Rebuild the keyword index from current metadata and text, swapping it in atomically (search keeps working until the swap). Run it once on an index created before 2026-09-27 so metadata edits reach keyword search; `status --json` reports `fts_contentless_delete: true` afterwards. About 4 s for 1,800 papers. `--dry-run`, `--json` |
| `gantry chunks backfill-pages` | Set `page_start`/`page_end` on chunks indexed before pages were recorded, from the PDFs, without re-chunking or re-embedding. `--ids`, `--limit`, `--force`, `--dry-run`; reports coverage and papers with unplaced chunks |

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

Config lives at `~/.gantry/config.yaml`; `GANTRY_*` environment variables override it (`GANTRY_PAPERS_DIR`, `GANTRY_INDEX_DIR`, `GANTRY_VAULT_DIR`, `GANTRY_OPENALEX_MAILTO`). The two settings that matter: `papers_dir`, any flat folder of PDFs, and optionally `vault_dir` for Obsidian cross-referencing. Set `openalex_mailto` to use OpenAlex's faster polite pool during `gantry enrich`.

## How it works

- **Change detection:** PDFs are SHA-256 hashed at ingest; only new or changed files are reprocessed.
- **Extraction:** PyMuPDF4LLM by default, Marker as an optional higher-quality backend.
- **Search:** contentless FTS5 (`contentless_delete=1`) for keywords, over filename, title, authors, abstract and text, refreshed whenever `process`, `ocr`, `enrich` or `meta` changes any of them, 768-d Nomic Embed V2 vectors in sqlite-vec for semantics, reciprocal rank fusion for hybrid. Hybrid degrades gracefully to FTS if the embedding model is unavailable, and to vector-only if FTS5 rejects a raw `--fts-syntax` query. Each result's `snippet` is the passage that matched: the best vector chunk, else the chunk containing the most query terms, else a window of text around the first term. It is not the first 200 characters of the file, which are usually a masthead.
- **Chunks:** documents are split into addressable chunks with per-chunk embeddings, so retrieval can land on a passage instead of a paper.
- **Chunk pages:** each chunk records the physical PDF pages it came from (`page_start`/`page_end`, 1-based). The markdown has no page breaks, so `process` locates each chunk's text in PyMuPDF's per-page text (head/tail probes, then word overlap, then interpolation between neighbours); OCR chunks take the page from their `## Page N` section. On a 150k-chunk corpus this places 99.9% of chunks, 97.6% by direct text match. Pages appear in `grep`, `info --query`, `info --chunks` and `read --chunk(s)`.
- **Scanned PDFs:** classified at ingest and routed to the OCR queue.
- **Quarantine:** a paper that fails processing/embedding `processing.max_retries` times (default 3) is skipped by default selection so a permanently-broken PDF isn't re-attempted on every run. Find them with `gantry queue --is broken`; un-quarantine a fixed file with `gantry retry --ids <ids>`.

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
