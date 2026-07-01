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
| 1 Composable retrieval | +0 | – |
| 2 Corpus fidelity | +0 | – |
- **Coverage:** 0/48 elite-map cells · **Tone:** 0 L / 0 D (debt −1) · **Ops:** 0 Sp / 0 Gr / 0 Pr · **Fouls:** 0

## Elite map — best move per (perspective × operator)  (rewritten in place each round)
| Perspective | Graft | Prune | Splice |
| --- | --- | --- | --- |
| 1 Cartographer | – | – | – |
| 2 Gardener | – | – | – |
| 3 Provocateur | – | – | – |
| 4 Adversary | – | – | – |
| 5 Minimalist | – | – | – |
| 6 Naturalist | – | – | – |
| 7 Diplomat | – | – | – |
| 8 Aesthete | – | – | – |
| 9 Archivist | – | – | – |
| 10 Trickster | – | – | – |
| 11 Engineer | – | – | – |
| 12 Harness Engineer | – | – | – |
| 13 Oracle | – | – | – |
| 14 User-Advocate | – | – | – |
| 15 Migrator | – | – | – |
| 16 Sentinel | – | – | – |

## Archive (fouls + non-elite stepping stones)
(none yet)
