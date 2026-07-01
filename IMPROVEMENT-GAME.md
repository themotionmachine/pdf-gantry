# The Improvement Game — pdf-gantry — 2c68bd9

## Palette
- **Bookends:** gantry is a working FTS5+semantic-search CLI over a real ~2000-PDF corpus, mid-real-world-shakeout, with core composable primitives shipped (`info --ids`, `info --query`, `best_chunk_per_doc()`) but the query algebra still shallow → a full composable query algebra where agents chain retrieval at chunk granularity (ID-set piping, scoped context windows, cross-paper filtered queries) without full serialize/deserialize round-trips, touching raw PDFs, or managing their own context windows.
- **Attractors:**
  1. **Composable retrieval** — the move adds or extends a demonstrable, composable capability in the query algebra (chaining, ID-piping, scoped windows, cross-paper ops) that saves agent tokens vs. today's round-trips.
  2. **Corpus fidelity** — the move optimizes for depth and correctness on Ryan's actual ~2000-PDF corpus and retrieval habits, not generic breadth or config surface justified only by "other users might want it."
- **Yes (grants):** may add a new Python dependency (core or a new/existing `pyproject.toml` optional-dependency group) if genuinely warranted; may call external APIs the codebase already integrates (Semantic Scholar, OpenAlex) but never from tests — fixtures only.
- **No (bans):** TDD is law (red→green, new behavior ships tested) · never merge to `main` without Ryan's consent · never force-push `main` · no secrets in the repo, no dashboard-only config · history-sacred (no greening by weakening a prior round's test, except a licensed Prune card, logged) · no PDF access added to read-path commands (search/info/read/status/queue/errors stay pure-DB, per the Quern remote-shim invariant) — a standing repo invariant, not this run's attractor, but still a hard boundary.
- **Rounds:** 5 · **Master seed:** `17673f212e9d9227b02a3fadecf82244` · **Debt K:** 4, starts −1 · **Deck:** 16, with replacement, feature-forward-weighted
- **Selection:** graded score (−2…+3/attractor) + elite-map placement (16×3 = 48 cells) + history-sacred · best-of-2 on Graft rounds only
- **Profile:** **feature-forward** — deck weighted toward Provocateur/Oracle/User-Advocate, operator Graft-heavy (50% flat, Splice 35→10% annealing, Prune the rest), memory Rooted-heavy (`aw=40-prog/3`), tone debt starts −1/K=4, best-of-2 on Graft rounds only, feature-forward Binding deck. Anchor surface-boost left **off** (default pool+churn only) — logged per profile spec.
- **Topology (house rule):** the pre-existing worktree/branch `new-ig-round` (this checkout, off `main` @ 2c68bd9) *is* the game trunk — no separate `improvement-game/trunk` branch created. Round branches: `r<N>-<perspective-slug>`, merged back to `new-ig-round` on advance.
- **Autonomy:** all 5 rounds played back-to-back, chain presented at the end.
- **Executor:** each round played by a fresh `sonnet`-model subagent inhabiting the drawn perspective (per Ryan's request this session).

## Scoreboard  (rewritten in place each round)
| Attractor | Cumulative | Last Δ |
| --- | --- | --- |
| 1 Composable retrieval | +2 | +0 |
| 2 Corpus fidelity | +4 | +3 |
- **Coverage:** 2/48 elite-map cells · **Tone:** 0 L / 2 D (debt −3) · **Ops:** 1 Sp / 1 Gr / 0 Pr · **Fouls:** 0

## Elite map — best move per (perspective × operator)  (rewritten in place each round)
| Perspective | Graft | Prune | Splice |
| --- | --- | --- | --- |
| 1 Cartographer | – | – | – |
| 2 Gardener | – | – | – |
| 3 Provocateur | – | – | – |
| 4 Adversary | **r2b · Δ+3 · encrypted PDF no longer aborts batch ingest · r2b-adversary** | – | – |
| 5 Minimalist | – | – | – |
| 6 Naturalist | – | – | – |
| 7 Diplomat | – | – | – |
| 8 Aesthete | – | – | – |
| 9 Archivist | – | – | – |
| 10 Trickster | – | – | – |
| 11 Engineer | – | – | – |
| 12 Harness Engineer | – | – | – |
| 13 Oracle | – | – | – |
| 14 User-Advocate | – | – | **r1 · Δ+2 · surfaced dropped IDs in `info --ids` (not_found + partial exit) · r1-user-advocate** |
| 15 Migrator | – | – | – |
| 16 Sentinel | – | – | – |

## Round 1 — User-Advocate · Splice · Dark
- **Roll:** R=1 P=14 OP=Splice TONE=Dark MEM=Amnesiac spine=3 ("What does this make the user understand that they shouldn't have to?") band=Tame anchor=src/pdf_gantry/utils.py · complication: none · reroll: none
- **Provocation (GM, scaffolding only):** User-Advocate, Splice ("reroute the path — recombine existing capabilities into the flow the user's task actually wants"), Dark register (confront, don't delight), spine question above, anchor `src/pdf_gantry/utils.py` as entry point only, both attractors eligible.
- **Focus (player's own words):** "`gantry info --ids` is the primitive CLAUDE.md brags about shipping... walking the path an agent actually walks — `search "X" --ids-only` piped into `info --ids ... --query Y` — exposes the seam: if any IDs no longer resolve, `info` silently returns fewer papers than requested and says nothing... I'm closing that loop: `info` now reports exactly which requested IDs it couldn't resolve (`not_found`) and downgrades its exit code to partial-failure (3) when some-but-not-all IDs resolve."
- **Played:** Added `missing_ids()` to `utils.py` (order-preserving, dedupe-safe set-diff, TDD'd in `tests/test_utils.py`). Wired into `gantry info --ids`: JSON and plain output now carry `not_found`; partial resolution now exits `EXIT_PARTIAL` (3) instead of silent success. New `tests/test_info_not_found.py` covers all-found / partial / all-missing / duplicate-ID cases. · **PR:** local branch `r1-user-advocate` (not pushed — game trunk only, per Palette)
- **Score:** Composable retrieval **+2** (a real reliability gap in the ID-piping composition path is closed — an agent chaining commands can now detect a silent drop via body or exit code) · Corpus fidelity **+1** (motivated by real pruning/staleness on the actual ~2000-PDF corpus, not a hypothetical) → **trunk: ADVANCED** (merged `--ff-only`) · **elite-map:** new champion of (User-Advocate × Splice)
- **Legacy hooks:** (1) same silent-drop pattern likely in `search --restrict-to-ids` / `semantic --restrict-to-ids`; (2) `ocr --ids` / `retry --ids` don't report unresolved target IDs either; (3) `missing_ids()` generalizes if a future `queue --ids-only`-style surface ships.

## Round 2 — Adversary (lead) × Sentinel (Multiclass modifier) · Graft · Dark
- **Roll:** R=2 P=4 OP=Graft TONE=Dark MEM=Amnesiac spine=2 ("What's the input the author never pictured — empty, enormous, malformed, hostile, concurrent?") band=Standard anchor=`IMPROVEMENT-GAME.md` · complication: Multiclass — Adversary lead, in the manner of card #16 Sentinel as modifier · reroll: none
- **Anchor note:** the drawn anchor was the game's own Chronicle file (0 code references, born minutes earlier) — not a legitimate Adversary/Sentinel target. Both players correctly reached past it into `src/pdf_gantry/` per the "entry point, not a cage" rule and recorded why in their Focus statements.
- **Provocation (GM, scaffolding only):** Adversary supplies the method (Graft — write the hostile-input test that locks behavior), Sentinel supplies the lens (aim at a trust boundary where *external/foreign data* crosses in — an API response, a file this system didn't produce). Dark register. Spine question above. Both attractors eligible.
- **Best-of-2 played** (feature-forward profile: best-of-2 fires on Graft rounds) — two independent players, same frame, sub-draws tagged `:a`/`:b`:
  - **Player A** (`r2-adversary`) — **Focus:** "...`metadata.py`/`openalex.py` pull real JSON off the wire and hand it to `parse_openalex_work`, which trusts `a.get('author', {}).get('display_name', '')`... OpenAlex legitimately returns `'author': null` for group/consortium authorships... one bad entry and the parser throws `AttributeError`, swallowed silently by the caller's bare `except Exception: return None` — a title, DOI, year, abstract we already paid a network round-trip for, gone, no error surfaced." Null-coalesced `a.get("author") or {}` in `openalex.py::parse_openalex_work`, 2 new tests (null-author entry, missing-author-key entry).
  - **Player B** (`r2b-adversary`) — **Focus:** "`ingest_directory()` is the front door — every one of Ryan's ~2000 PDFs... passes through `classify_document()`... PyMuPDF happily *opens* a password-protected PDF without raising, and only throws `ValueError` later when the per-page loop touches `page.get_text()` — a call the author never wrapped... One password-protected PDF anywhere in a 2000-file iCloud folder... takes down the entire batch ingest, not just that file." Added `doc.needs_pass` guard + per-page try/except + try/finally close in `ingest.py::classify_document`, new `encrypted_pdf` fixture, 2 new tests (classify doesn't raise; ingest survives a locked file mixed into the batch).
- **GM verification:** read `ingest_directory`/`_get_pdf_metadata` on the trunk before scoring — confirmed `classify_document()` was called with **no surrounding try/except** in the per-file loop, so Player B's claimed blast radius (one encrypted PDF aborts the *entire* directory scan) was real, not embellished.
- **Score comparison:** A — Composable retrieval +0, Corpus fidelity **+2** (real bug, but already contained: a bare `except Exception` upstream meant the failure mode was "one paper's enrichment silently drops," not a crash). B — Composable retrieval +0, Corpus fidelity **+3** (unguarded: one bad file crashes the whole batch scan across the real ~2000-PDF corpus — bigger blast radius, genuine stride).
- **Winner: Player B** (`r2b-adversary`, Δ+3 > Δ+2) → **trunk: ADVANCED** (merged `--ff-only`) · **elite-map:** new champion of (Adversary × Graft). Player A (`r2-adversary`) is **kept, not merged** — archived below as a real, gate-green, non-elite stepping stone.
- **Legacy hooks:** (from B) `process.py`'s PyMuPDF4LLM extraction likely hits the same `needs_pass` wall on a file ingest now lets through as "digital"; encrypted PDFs are silently classified `digital` with zero signal they're actually unreadable — worth a `queue`-visible "blocked/encrypted" status; other bare `fitz.open()` call sites may share the shape. (from A, still open) `metadata.py::_normalize_semantic_scholar` has the identical bug shape one file over; `openalex.py`'s three `fetch_by_*` functions swallow *any* parse failure into silent "no match" — worth splitting `stats.no_match` from `stats.parse_errors`.

## Archive (fouls + non-elite stepping stones)
- **r2-adversary** (Player A, round 2) — held, not merged (best-of-2 runner-up, Δ+2 vs winner's Δ+3). Real fix (OpenAlex null-author crash, gate green, TDD'd), same (Adversary × Graft) cell as the winner so it doesn't get its own elite slot. Branch kept on disk for reference; not on the trunk.
