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

A library collects the same paper more than once: a publisher PDF and a preprint, a renamed re-download. `--works` folds those copies into one result, so `-n 10` means ten papers rather than ten files:

```bash
$ gantry search "niche construction free energy" --works --fields id,work_key,canonical_id,copies --json -n 5
```

Each result then carries, for example, `{"id": 645, "work_key": "work:326", "canonical_id": 326, "copies": [326]}`: two downloads of one paper, returned once.

The result keeps the id of the copy that matched (so page numbers in that copy stay right). `work_key` is `work:<smallest member id>`, `canonical_id` is the copy to prefer (it has a citekey, then the best metadata source: manual, then bib, then OpenAlex; then not suspect, then has text, then the newest), and `copies` lists the work's other ids across the whole library. Plain text output lists the copies after each hit; `--format oneline` is unchanged. Without `--works`, output is exactly as before.

Two rows are the same work when they have the same file hash; or the same DOI, with titles that agree (similarity at least 0.75, or one contained in the other) and page counts within 2; or the same metadata `title` and `year`, for titles of five or more words that aren't generic ("Introduction to the special issue…", "Book review…") and page counts within 2. A row flagged `metadata_suspect` takes neither DOI nor title evidence, and when both rows have a title read off page 1, those must agree too, so two chapters stamped with their book's DOI stay apart. A wrong merge would hide a paper, so the rules lean toward showing a duplicate. Python callers get the same through `pdf_gantry.works.works_for(conn, ids)` and `collapse(conn, results, limit)`, which only read and work on a `mode=ro` connection.

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
uv pip install -e ".[embeddings]" --python .venv/bin/python

.venv/bin/gantry config set papers_dir ~/Papers   # your folder of PDFs
.venv/bin/gantry pipeline       # ingest → extract text → embed
.venv/bin/gantry search "your first query"
```

`gantry config init` asks the same questions interactively; `config set` is the form an agent can run.

The base install covers ingestion, extraction, and keyword search. Heavier features are opt-in extras:

```bash
uv pip install -e ".[embeddings]"  # semantic + hybrid search (sentence-transformers)
uv pip install -e ".[ocr]"        # scanned PDFs (Surya)
uv pip install -e ".[quality]"    # higher-quality extraction (Marker)
uv pip install -e ".[all]"        # everything
```

### Setting up with an agent

You can point a coding agent at this repo and ask it to set gantry up. The Quickstart above is the whole procedure. A few things worth knowing before you do:

- **Install `.[embeddings]`, not `.[all]`.** `all` adds the OCR (Surya) and Marker stacks, which are much larger and only matter for scanned PDFs or higher-quality extraction.
- **The first run downloads the embedding model.** `pipeline` fetches Nomic Embed V2 (about 1.8 GB, plus its model code) from Hugging Face once. Embedding then runs locally. To make sure nothing reaches the network afterwards, set `HF_HUB_OFFLINE=1`.
- **`pipeline` keeps documents on your machine.** It reads, extracts and embeds locally. Only `gantry enrich` (or `pipeline --enrich`) calls an outside service: it sends each paper's DOI, title and filename-derived author and year to OpenAlex. Skip it for confidential documents. It is also built for scholarly papers, so it won't find patents or legal filings.
- **Whatever your agent reads, its model provider sees.** Gantry returns passages to the agent, and the agent sends them to its model. Decide that the same way you would for pasting the text in yourself.
- **Check the result with a semantic query.** `gantry status` should show every paper with text and chunk embeddings, and `gantry search` should find a document by meaning, not only by its exact words.

## The agent contract

Gantry holds to a few rules so that agents (and scripts) can rely on it:

- **`--json` on every command.** Structured output goes to stdout; progress bars and chatter go to stderr. Pipes stay clean.
- **`--fields` trims payloads.** Ask for `id,filename,top_chunk` and that is all you get. Tokens are the budget; spend them on content. An unknown field name gets a warning on stderr that lists the valid names; it is not dropped silently.
- **One shape per concept.** `authors` is always a JSON list of names (`[]` when unknown) in `search`, `semantic`, `info`, `find` and `read`. Search results carry `title`, `year`, `authors` and `citekey`.
- **Counts say what they count.** In `search`/`semantic` JSON, `returned` is the number of results in the response. `total` is the global FTS5 match count for `search --fts` without `--restrict-to-ids`, regardless of `-n`. Hybrid and semantic search rank rather than match, so there `total` equals `returned`. With `search --works`, `total` counts works, except a global `--fts` count larger than the over-fetch, which still counts matching files. The `mode` field reports the retrieval path that actually ran: `hybrid`, `vector_fallback` (FTS5 rejected the query, so the results are vector-only), `fts` (embeddings unavailable), `fts_only` (`--fts`), or, for `semantic`, `cascade` or `doc`.
- **Exit codes carry meaning.**

  | Code | Meaning |
  |------|---------|
  | 0 | success |
  | 1 | error |
  | 2 | ran fine, no results |
  | 3 | partial failure (some documents succeeded, or some requested IDs/indices were missing) |
  | 4 | database error |
  | 64 | usage error: unknown option or command, bad choice, missing argument, unknown collection or inbox item |

  An agent can branch on "no results" without parsing anything. A typo is never exit 2: usage errors exit 64, and when `--json` appears anywhere on the command line they print `{"error": "...", "usage": "...", "exit_code": 64}` on stdout.
- **Stable JSON envelopes.** Keys are only ever added, never renamed. Errors from any command under `--json` are `{"error": "..."}`; a refused `collection` write adds a stable `code` (`name_conflict`, `already_undone`, `undo_conflict`, `unknown_batch`), and so does a refused `inbox` write (`item_retired`, `already_undone`, `undo_conflict`, `unknown_batch`). Paper-level records carry `id` (the paper ID); `read` also keeps its older `paper_id`, and chunk records carry both `id`/`paper_id` and the chunk's own `chunk_id`.

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
  | `collection show` | `{collection: {id, name, note, created_by, created_at}, ids, members: [{id, filename, title, display_title, citekey, added_by, added_at, why, batch_id, …}], missing, ambiguous}` |
  | `collection list` | `{count, collections: [{id, name, note, count, resolved, missing, ambiguous, …}]}` |
  | `collection history` | `{count, history: [{batch_id, op, by, why, at, count, undone, collection, …}]}` |
  | `inbox list`, `inbox due` | `{count, items: [{item_id, paper_id, resolution, candidates, file_hash, path, title, citekey, source, by, why, added_at, passage, status, due_at, interval_days, reviews, last_verdict, …}]}` |
  | `inbox add` | `{batch_id, added, already_present, not_found, items: [{paper_id, item_id, status}]}` |
  | `inbox history` | `{count, history: [{batch_id, op, by, why, at, count, item_id, verdict, undone, …}]}` |
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
| `gantry titles backfill` | Read each paper's title from its page typography into `title_from_layout`. Opens PDFs; `--ids`, `--force`, `--limit`, `--dry-run`, `--json`. New papers get this from `process` and `pipeline` |
| `gantry pipeline [FILE...]` | Run the full chain: ingest → process → embed. `--enrich` adds metadata lookup before embedding. Positional files (or `--file`) scope it to those PDFs: `gantry pipeline paper.pdf --enrich --json` adds one paper end to end and reports its `ids` |
| `gantry meta set\|clear\|normalize` | Hand metadata edits in code rather than raw SQL (see below) |

#### Enriching metadata

Most flat PDF folders carry almost no metadata, so `gantry enrich` backfills it from a scholarly index and writes it onto each paper (`title`, `authors`, `year`, `doi`, `abstract`). With metadata in place, `gantry link init` can generate a real `.bib`.

It defaults to **OpenAlex**: broad coverage, and a "polite pool" that runs faster when requests carry your email. Without an API key OpenAlex gives each IP a small daily budget, enough for a trial batch but not a whole library, so for a full run get a free key and set it too. Set both once and every run uses them:

```bash
gantry config set openalex_mailto you@example.com
gantry config set openalex_api_key YOUR_KEY
gantry enrich --dry-run        # how many papers would be touched
gantry enrich --limit 20       # small batch to eyeball quality first
gantry enrich                  # the whole library
```

For each paper it tries, in order: exact DOI lookup (from the `doi` column, else a DOI found in the first page of text), then an author+year search derived from the filename, then a title search. DOI matches are reliable; title-search fallbacks on opaque filenames are worth a skeptical pass.

Semantic Scholar is still available with `--provider semantic-scholar`, but without an API key it rate-limits hard, which is why OpenAlex is the default. By default `enrich` only touches papers it has never tried; pass `--ids`, `--has`/`--needs`/`--is` filters or `--limit` to scope it.

A paper the provider can't match is recorded as `metadata_source = 'none:<provider>'`, so "tried and missed" is distinguishable from "never tried". Misses aren't retried by default; `gantry enrich --retry-misses` re-attempts them, and `gantry queue --is enrich-miss` lists them. A lookup that fails on a network or API error leaves the paper untouched, so the next run tries it again.

When OpenAlex answers with a rate limit (HTTP 429), `enrich` stops rather than recording the remaining papers as misses. Those papers stay untried for the next run, JSON reports `rate_limited: true`, and the exit code is 3.

The title search queries with the layout title when a paper has one. Under `--retry-misses` a title hit must also be present in the document and resemble the layout title (similarity at least 0.75); anything else stays a miss and writes nothing, so a retry can't leave a wrong title behind. The DOI tier is unchanged. JSON `titled_ids` lists the papers that gained a title, and `rejected_title` counts the hits turned down.

`gantry pipeline --enrich` runs enrichment after text extraction and before embedding, because each chunk is embedded with its paper's title prepended. Its JSON reports `enriched`, an `enrich` block of counts, and `titled_after_embed`: papers that were already embedded before they got a title, ready for `gantry embed --chunk --ids … --force`.

#### Layout titles

About half the library has no provider metadata, so `gantry titles backfill` reads a title off the page: the largest text in the top 60% of page 1, skipping corpus boilerplate, cover sheets, labels, bylines and citation lines, and returning nothing for a garbled text layer. It writes `title_from_layout` and `title_layout_confidence` (`high` when the font stands out from the body text, else `low`).

Treat it as an unverified hint. It scores 88% against bib titles (`eval/title_harness.py`), but those papers are the well-formed ones, and a blind sample of untitled papers scored far lower. So `title` keeps meaning verified metadata and is never filled from layout. `search`, `semantic`, `info` and `find` add `display_title` and `title_source` (`metadata`, `layout` or `filename`) to their JSON, plus `title_layout_confidence` when the source is layout; `--format oneline` and plain text show the same fallback. Keyword search indexes the layout title when a paper has no title, and `find` matches it one rank below a verified title. A `link check` match on a layout title is always `uncertain`.

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
| `gantry search <query>` | Hybrid search (FTS5 + vector, fused with RRF); `--fts` for keyword-only, `--fts-syntax` for raw FTS5 syntax. `--ids-only` emits bare ranked IDs for piping; `--format oneline` emits `id score year citekey title`; `--restrict-to-ids` scopes the search to a candidate set; `--collection NAME` scopes it to a collection (both together give the intersection; an unknown name exits 64); `--queries-file PATH\|-` runs many queries with one model load; `--works` folds copies of the same paper into one result (adds `work_key`, `canonical_id`, `copies`) |
| `gantry semantic <query>` | Pure vector similarity search; also supports `--ids-only`, `--format oneline`, `--restrict-to-ids` and `--queries-file` |
| `gantry find <words>` | Known-item lookup over title, authors, year, citekey, DOI and filename: `find "Mind games"`, `find "Zhang 2022"`, `find "Flew & Martin"`. Every word must match some field; title matches rank first; each result reports `matched_fields`. `--ids-only` for piping |
| `gantry read <id>` | Read a document's text, list its chunks (`--chunks`), read chunks by per-paper position (`--index 28-41`), or read one chunk by `chunk_id` (`--chunk`, which must belong to that paper) and expand it with `--context` |
| `gantry info <ids…>` | Metadata for specific papers; takes IDs (`info 12 13`, `info 12,13`), filenames or citekeys (`info @smith2020`), or `--ids`. `--query` attaches each paper's best-matching chunk |
| `gantry schema` | Tables, columns, relationships and common joins, for raw-SQL callers |
| `gantry grep "<text>"` | Find a literal string (a quote) in chunk text; each hit gives doc, chunk, `page_start`/`page_end`, offset and whether the match was exact or normalised. `--ids`, `-i/--ignore-case`, `--limit`, `--context`. Papers without chunks are searched in raw text |

`<id>` in `read` and `info` accepts a paper ID, a filename, or a citekey (with or without `@`).

**Raw SQL.** If you query `~/.gantry/index.db` directly, use the views `v_papers` and `v_chunks`. Both are keyed by `paper_id` (the underlying `chunks` table calls it `doc_id`, and a chunk's key is `chunk_id`, not `id`). `v_chunks` exposes `paper_id, chunk_id, chunk_index, section_header, page_start, text, char_offset`. Full text lives in `paper_text.raw_text` / `paper_text.markdown`. `gantry schema` prints all of this.

### Collections

A collection is a hand-curated set of papers. Agents can make them freely: every write names who made it (`--by`), gets a batch ID, and can be undone.

| Command | Description |
|---------|-------------|
| `gantry collection create <name>` | New empty collection. `--note`, `--by`. Names are unique among live collections, ignoring case |
| `gantry collection add <name> --ids <ids\|->` | Add papers. `--by`, `--why`. `-` reads IDs from stdin, so `search --ids-only` pipes straight in. Adding a member again is a no-op reported in `already_present` |
| `gantry collection remove <name> --ids <ids\|->` | Remove papers. `--by`. Keeps a tombstone |
| `gantry collection show <name>` | Members with who added them and why. `--ids-only` prints bare IDs and sends any missing count to stderr. Exit 2 when empty, 3 when members are missing or ambiguous |
| `gantry collection list` | Collections with their counts |
| `gantry collection rename <name> <new>` | Rename. `--by` |
| `gantry collection delete <name>` | Delete. `--by`. Keeps a tombstone, members included |
| `gantry collection undo <batch>` | Reverse one batch, and only what it did. `--by`. A membership re-added since is left alone and listed in `skipped` (exit 3). A taken name, or a collection renamed or filled since, refuses the undo (exit 1). Undoing twice is an error |
| `gantry collection history [<name>]` | One row per batch, newest first. `--limit` |

```bash
gantry collection create governance --by claude
gantry search "platform governance" --ids-only \
  | gantry collection add governance --ids - --by claude --why "for chapter 3"
gantry search "appeals" --collection governance
gantry collection show governance --ids-only | gantry grep "oversight board" --ids -
```

Collections live in `collections.db` beside the index, mode 0600, and that file is the only thing these commands write. Paper IDs are not stable: renaming a file gives it a new ID. So a member is stored as file hash plus path and matched back to the index on each read: hash and path, then hash, then path. Where the papers directory is reachable, a row whose file is gone loses to one whose file exists, so the row a rename left behind yields to the new one. A member matching nothing is reported as `missing` and kept, because the paper may come back. One matching several rows equally well is reported as `ambiguous`. `gantry pipeline` writes moved keys back as its last step.

### Reading inbox

The inbox is a spaced-repetition queue of papers to read, and of passages within them. An item enters due now; each `review` gives a verdict and schedules the next time it comes back. Like collections, every write names who made it (`--by`), gets a batch ID, and can be undone.

| Command | Description |
|---------|-------------|
| `gantry inbox add --ids <ids\|->` | Add papers, due now. `--by`, `--why`, `--source manual\|agent`. `--page N` (1-based) with optional `--quote` and `--note` adds a passage instead. A paper holds any number of passages but one whole-paper item: adding it again is a no-op reported in `already_present` |
| `gantry inbox due` | Items due now, longest-waiting first. `--limit`, `--ids-only`. Exit 2 when nothing is due |
| `gantry inbox list` | Items by `--status`: `active` (the default: due or scheduled), `due`, `scheduled`, `done` or `all`. `--ids-only` |
| `gantry inbox review <item> <verdict>` | `key` (!!) triples the interval, at least a week; `good` (!) doubles it; `doubt` (?) brings it back in 1–2 days; `later` (→) grows it 1.5×, no penalty; `done` (x) retires it. A first review starts from one day. `--by` |
| `gantry inbox undo <batch>` | Undoing an add tombstones its items. Undoing a review puts back the item's schedule from before it, so an undone `done` returns to the inbox. A review undo is refused (exit 1) if the item was reviewed again or removed since. `--by` |
| `gantry inbox history` | One row per batch, newest first. `--limit` |

```bash
gantry search "content moderation appeals" --ids-only -n 3 \
  | gantry inbox add --ids - --by claude --source agent --why "for chapter 3"
gantry inbox add --ids 412 --page 7 --quote "the appeal is the product" --by ryan
gantry inbox due
gantry inbox review 5 good --by ryan
```

The inbox lives in `reading.db` beside the index, mode 0600, and that file is the only thing these commands write. Items are keyed and matched exactly as collection members are, so they follow a paper through a rename, and `missing` or `ambiguous` items are reported (`resolution`, exit 3) rather than dropped. `reading.db` also holds an append-only log of reader opens (`reading.record_open`, used by Ramelli), which is not batched and not undoable. Times are UTC ISO strings. `gantry pipeline` writes moved item keys back as its last step, and never creates the file.

### Index management

| Command | Description |
|---------|-------------|
| `gantry status` | Coverage and database stats, plus pending-work counts (`needs_text`, `needs_embeddings`, `needs_chunk_embeddings`, `needs_enrich`, `enrich_misses`, `metadata_suspect`, `manual_metadata`, `bib_metadata`, `needs_layout_title`) matching what the next default run would pick up, and `layout_titled` and `display_untitled` (papers showing only a filename) |
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
| `gantry link suggest <bib>` | Candidate citekeys for papers with no citekey and no title, with evidence, a tier (`strong`, `title`, `weak`, `filename`) and any sibling entries. `--apply --ids 1,2` assigns the top candidate to those papers only and refuses a citekey already taken |
| `gantry link audit <bib>` | Linked papers whose bib title is not in the document while their stored or layout title is. `--clear --ids 1,2` removes those citekeys and appends the old values to `citekey-clears.jsonl` in the index directory |
| `gantry link pull <bib>` | Copy title, authors, year, DOI and abstract from entries onto linked papers. Precedence is hand-set > bib > provider > layout; only rows whose bib title is in the document change. Records `metadata_source = 'bib:<stem>'`, which `enrich` never overwrites, and reports `newly_titled_ids` for re-embedding |
| `gantry link missing <bib>` | Bib entries with no paper by citekey, `file=` basename or DOI. `--json` or `--format tsv` |
| `gantry vault check` | Cross-reference papers against an Obsidian vault (read-only) |

## Configuration

```bash
gantry config init          # interactive setup
gantry config show
gantry config set papers_dir ~/Papers
```

Config lives at `~/.gantry/config.yaml`; `GANTRY_*` environment variables override it (`GANTRY_PAPERS_DIR`, `GANTRY_INDEX_DIR`, `GANTRY_VAULT_DIR`, `GANTRY_OPENALEX_MAILTO`, `GANTRY_OPENALEX_API_KEY`). The two settings that matter: `papers_dir`, any flat folder of PDFs, and optionally `vault_dir` for Obsidian cross-referencing. Set `openalex_mailto` to use OpenAlex's faster polite pool during `gantry enrich`, and `openalex_api_key` for a daily budget large enough to enrich a whole library. `config show` never prints the key.

## How it works

- **Change detection:** PDFs are SHA-256 hashed at ingest; only new or changed files are reprocessed.
- **Extraction:** PyMuPDF4LLM by default, Marker as an optional higher-quality backend.
- **Search:** contentless FTS5 (`contentless_delete=1`) for keywords, over filename, title, authors, abstract and text, refreshed whenever `process`, `ocr`, `enrich`, `meta`, `link pull` or `titles backfill` changes any of them, 768-d Nomic Embed V2 vectors in sqlite-vec for semantics, reciprocal rank fusion for hybrid. Hybrid degrades gracefully to FTS if the embedding model is unavailable, and to vector-only if FTS5 rejects a raw `--fts-syntax` query. Each result's `snippet` is the passage that matched: the best vector chunk, else the chunk containing the most query terms, else a window of text around the first term. It is not the first 200 characters of the file, which are usually a masthead.
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
