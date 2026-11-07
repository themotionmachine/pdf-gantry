"""Match a stored file_hash + path key back onto a row of ``papers``.

``papers.id`` is not stable (a rename gives a new row and leaves the old one
until a manual ``gantry prune``), so the sidecars (collections.db, reading.db)
key their rows on file_hash + path and resolve them at read time with
``match``. ``load_index`` reads rows by position, so ``index_conn`` may be a
plain ``mode=ro`` connection without a row factory.
"""

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paper:
    id: int
    path: str
    filename: str
    file_hash: str
    title: str | None
    layout_title: str | None
    citekey: str | None


@dataclass(frozen=True)
class Match:
    paper: Paper | None = None
    candidates: tuple[int, ...] = ()

    @property
    def ambiguous(self) -> bool:
        return bool(self.candidates)


class Index:
    """One ``papers`` query, keyed for matching, with a cached liveness check."""

    def __init__(self, papers: list[Paper], papers_dir=None):
        self.papers = papers
        self.by_id = {p.id: p for p in papers}
        self.by_path = {p.path: p for p in papers}
        self.by_hash: dict[str, list[Paper]] = defaultdict(list)
        for p in papers:
            self.by_hash[p.file_hash].append(p)
        # An unreachable papers_dir says nothing about any one file (Quern
        # calls over SSH): skip the check rather than call everything gone.
        self.papers_dir = Path(papers_dir) if papers_dir and Path(papers_dir).is_dir() else None
        self._live: dict[str, bool] = {}

    def is_live(self, paper: Paper) -> bool:
        if paper.path not in self._live:
            self._live[paper.path] = (self.papers_dir / paper.path).exists()
        return self._live[paper.path]

    def without_liveness(self) -> "Index":
        return Index(self.papers)


_COLUMNS = "id, path, filename, file_hash, title, title_from_layout, citekey"
_CHUNK = 400  # stays well under SQLite's bound-parameter limit


def load_index(index_conn: sqlite3.Connection, papers_dir=None) -> Index:
    rows = index_conn.execute(f"SELECT {_COLUMNS} FROM papers ORDER BY id").fetchall()
    return Index([Paper(*tuple(r)) for r in rows], papers_dir)


def load_index_for(index_conn: sqlite3.Connection, keys, papers_dir=None) -> Index:
    """An ``Index`` of just the rows that share a hash or a path with ``keys``.

    ``match`` only ever looks at ``by_hash[file_hash]`` and ``by_path[path]``,
    so for these keys it answers exactly as the full index would, without
    reading every paper. For hot read paths such as reading.state_for.
    """
    hashes = sorted({h for h, _ in keys})
    paths = sorted({p for _, p in keys})
    found: dict[int, Paper] = {}
    for column, values in (("file_hash", hashes), ("path", paths)):
        for i in range(0, len(values), _CHUNK):
            part = values[i:i + _CHUNK]
            rows = index_conn.execute(
                f"SELECT {_COLUMNS} FROM papers WHERE {column} IN"
                f" ({','.join('?' * len(part))})",
                part,
            ).fetchall()
            for r in rows:
                found.setdefault(r[0], Paper(*tuple(r)))
    return Index([found[i] for i in sorted(found)], papers_dir)


def decide(tier: list[Paper]) -> Match:
    if len(tier) == 1:
        return Match(paper=tier[0])
    return Match(candidates=tuple(p.id for p in tier))


def match(index: Index, file_hash: str, path: str) -> Match:
    """Find the paper a stored key points at now.

    Tiers: hash and path, then hash only, then path only. With a papers_dir, a
    live candidate in any tier beats a dead one in a better tier, so a stale
    row left by a rename yields to the live row with the same hash. If nothing
    is live the best dead row still resolves: its text and chunks serve reads,
    and an evicted file is not a lost paper.
    """
    same_hash = index.by_hash.get(file_hash, [])
    at_path = index.by_path.get(path)
    tiers = (
        [p for p in same_hash if p.path == path],
        [p for p in same_hash if p.path != path],
        [at_path] if at_path and at_path.file_hash != file_hash else [],
    )
    if index.papers_dir:
        for tier in tiers:
            live = [p for p in tier if index.is_live(p)]
            if live:
                return decide(live)
    for tier in tiers:
        if tier:
            return decide(tier)
    return Match()
