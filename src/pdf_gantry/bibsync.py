"""Reconciling the index with a hand-curated .bib, beyond ``link check``.

``link check`` needs a title, doi or matching filename to join a paper to an
entry, and a paper with no metadata has none of the first two. Here the document
itself is the evidence:

- ``suggest_links`` proposes citekeys for untitled, unlinked papers;
- ``audit_links`` finds linked papers whose bib title isn't in the document;
- ``pull_metadata`` copies entries onto linked papers (bib outranks providers);
- ``missing_entries`` lists entries with no paper at all.

The layout title is an unverified hint, so it only ever nominates. Every path
that writes needs the bib title to be present in the document text, or a person
naming the id. Precedence for metadata: hand-set > bib > provider > layout.
"""

import json
import re
import sqlite3
import unicodedata
from pathlib import Path

from .fts import refresh_row
from .link import BibEntry, clean_bib_text, extract_bib_filename, normalize_doi
from .meta import normalize_authors
from .metadata import (
    _CONTENT_WORD_RE,
    _TITLE_STOPWORDS,
    CONTAINMENT_HEAD_CHARS,
    CONTAINMENT_THRESHOLD,
    title_containment,
    title_similarity,
)
from .queue import BIB_SOURCE_PREFIX, manual_condition
from .utils import now_iso

SUGGEST_SOURCE = "suggest:reviewed"
SIMILARITY_GATE = 0.75  # layout title vs bib title
PREFIX_MIN_CHARS = 12  # "Main Title" of "Main Title: Subtitle"
FILENAME_CONTAINMENT = 0.8
MIN_PHRASE_CHARS = 8
TIER_ORDER = ("strong", "title", "weak", "filename")

_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_AUTHOR_SEP_RE = re.compile(r"\s+and\s+", re.I)


def _phrase(text: str | None) -> str:
    """Lowercase ASCII words, for phrase and surname tests."""
    folded = unicodedata.normalize("NFKD", text or "")
    return " ".join(_NON_WORD_RE.sub(" ", folded.lower()).split())


def split_authors(raw: str | None) -> list[str]:
    """A BibTeX author field as a list of clean names.

    Splits on " and " outside braces only: ``{Food and Agriculture
    Organization}`` is one author. A trailing "and others" is dropped.
    """
    if not raw:
        return []
    text = " ".join(raw.split())
    names: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0:
            m = _AUTHOR_SEP_RE.match(text, i)
            if m:
                names.append(text[start:i])
                i = start = m.end()
                continue
        i += 1
    names.append(text[start:])
    cleaned = [clean_bib_text(n) for n in names]
    return [n for n in cleaned if n and n.lower() != "others"]


def _surname(entry: BibEntry) -> str:
    names = split_authors(entry.authors)
    if not names:
        return ""
    first = names[0]
    last = first.split(",")[0] if "," in first else first
    words = _phrase(last).split()
    return words[-1] if words else ""


def _title_tokens(title: str) -> list[str]:
    words = _CONTENT_WORD_RE.findall(title.lower())
    return [w for w in words if w not in _TITLE_STOPWORDS] or words


class _Candidate:
    """A bib entry with its comparison keys computed once."""

    def __init__(self, entry: BibEntry):
        self.entry = entry
        self.phrase = _phrase(entry.title)
        self.tokens = _title_tokens(entry.title or "")
        self.surname = _surname(entry)
        self.year = str(entry.year) if entry.year else ""


def _candidates(entries: list[BibEntry]) -> list[_Candidate]:
    built = [_Candidate(e) for e in entries if e.title]
    return [c for c in built if len(c.phrase) >= MIN_PHRASE_CHARS and c.tokens]


def _judge(cand: _Candidate, layout: str | None, confidence: str | None, head: str,
           head_words: set[str], head_phrase: str, filename: str) -> dict | None:
    """Evidence for one paper/entry pair, or None if neither gate passes."""
    containment = sum(w in head_words for w in cand.tokens) / len(cand.tokens)
    surname_hit = bool(cand.surname) and cand.surname in filename
    year_hit = bool(cand.year) and cand.year in filename
    if containment < CONTAINMENT_THRESHOLD and not (surname_hit and year_hit):
        return None  # the cheap test first: this runs papers x entries times

    layout_phrase = _phrase(layout)
    similarity = title_similarity(layout, cand.entry.title) if layout else 0.0
    prefix = len(layout_phrase) >= PREFIX_MIN_CHARS and cand.phrase.startswith(layout_phrase)
    title_gate = (similarity >= SIMILARITY_GATE or prefix) \
        and containment >= CONTAINMENT_THRESHOLD
    phrase_in_head = cand.phrase in head_phrase
    filename_gate = surname_hit and year_hit \
        and (phrase_in_head or containment >= FILENAME_CONTAINMENT)
    if not (title_gate or filename_gate):
        return None

    if title_gate and filename_gate:
        tier = "strong"
    elif title_gate:
        tier = "title" if confidence == "high" else "weak"
    else:
        tier = "filename"
    return {
        "tier": tier,
        "rank": (-(title_gate + filename_gate), -similarity, cand.entry.citekey),
        "citekey": cand.entry.citekey,
        "bib_title": cand.entry.title,
        "entry_type": cand.entry.entry_type,
        "year": cand.entry.year,
        "evidence": {
            "title_similarity": round(similarity, 3),
            "title_prefix": prefix,
            "containment": round(containment, 3),
            "phrase_in_head": phrase_in_head,
            "filename_surname": surname_hit,
            "filename_year": year_hit,
            "layout_confidence": confidence,
        },
    }


def _key_owners(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        r["citekey"].lower(): r["id"]
        for r in conn.execute("SELECT id, citekey FROM papers WHERE citekey IS NOT NULL")
    }


def suggest_links(
    conn: sqlite3.Connection,
    entries: list[BibEntry],
    paper_ids: list[int] | None = None,
) -> dict:
    """Candidate citekeys for papers with no citekey and no title.

    Passes a pair on either of two gates: the layout title resembles the bib
    title (or is its main-title prefix) and the bib title is in the document;
    or the filename carries the first author's surname and the year and the
    title is in the document's head. Tier ``strong`` passed both, ``title``
    the first on a high-confidence layout title, ``weak`` the first on a low
    one, ``filename`` the second.

    The known false accepts are sibling entries (chapter and book, drafts of
    one report), so every other entry that passed is listed in ``alternates``
    and the suggestion is marked ``ambiguous``. Nothing here writes.
    """
    sql = (
        f"SELECT p.id, p.filename, p.title_from_layout, p.title_layout_confidence, "
        f"SUBSTR(pt.raw_text, 1, {CONTAINMENT_HEAD_CHARS}) AS head "
        "FROM papers p JOIN paper_text pt ON pt.paper_id = p.id "
        "WHERE p.citekey IS NULL AND (p.title IS NULL OR p.title = '') "
        "AND pt.raw_text IS NOT NULL AND pt.raw_text != ''"
    )
    params: list = []
    if paper_ids is not None:
        sql += f" AND p.id IN ({','.join('?' * len(paper_ids))})"
        params.extend(paper_ids)
    rows = conn.execute(sql + " ORDER BY p.id", params).fetchall()

    cands = _candidates(entries)
    owners = _key_owners(conn)
    suggestions = []
    for row in rows:
        head = row["head"]
        head_words = set(_CONTENT_WORD_RE.findall(head.lower()))
        head_phrase = _phrase(head)
        filename = _phrase(row["filename"])
        passed = [
            j for j in (
                _judge(c, row["title_from_layout"], row["title_layout_confidence"], head,
                       head_words, head_phrase, filename)
                for c in cands
            ) if j
        ]
        if not passed:
            continue
        passed.sort(key=lambda j: j["rank"])
        best, *others = passed
        best.pop("rank")
        suggestions.append({
            "id": row["id"],
            "filename": row["filename"],
            "layout_title": row["title_from_layout"],
            **best,
            "citekey_taken_by": owners.get(best["citekey"].lower()),
            "ambiguous": bool(others),
            "alternates": [
                {"citekey": o["citekey"], "bib_title": o["bib_title"], "tier": o["tier"]}
                for o in others
            ],
        })

    by_tier: dict[str, int] = {}
    for s in suggestions:
        by_tier[s["tier"]] = by_tier.get(s["tier"], 0) + 1
    return {
        "papers_considered": len(rows),
        "suggested": len(suggestions),
        "by_tier": {t: by_tier[t] for t in TIER_ORDER if t in by_tier},
        "ambiguous": sum(s["ambiguous"] for s in suggestions),
        "citekey_taken": sum(s["citekey_taken_by"] is not None for s in suggestions),
        "suggestions": suggestions,
    }


def apply_suggestions(
    conn: sqlite3.Connection,
    entries: list[BibEntry],
    paper_ids: list[int],
    dry_run: bool = False,
) -> dict:
    """Assign the top suggestion to exactly the named papers.

    Naming an id is the review, so ``citekey_source`` records it. A citekey
    already held by another paper, or by an earlier id in this call, is
    refused rather than shared.
    """
    report = suggest_links(conn, entries, paper_ids=paper_ids)
    by_id = {s["id"]: s for s in report["suggestions"]}
    owners = _key_owners(conn)
    result: dict = {"dry_run": dry_run, "applied": [], "refused": [], "no_suggestion": []}
    now = now_iso()
    for pid in paper_ids:
        s = by_id.get(pid)
        if s is None:
            result["no_suggestion"].append(pid)
            continue
        key = s["citekey"].lower()
        if key in owners:
            result["refused"].append(
                {"id": pid, "citekey": s["citekey"], "taken_by": owners[key]}
            )
            continue
        owners[key] = pid
        result["applied"].append({"id": pid, "citekey": s["citekey"], "tier": s["tier"]})
        if not dry_run:
            conn.execute(
                "UPDATE papers SET citekey = ?, citekey_source = ?, updated_at = ? WHERE id = ?",
                (s["citekey"], SUGGEST_SOURCE, now, pid),
            )
    conn.commit()
    return result


def audit_links(conn: sqlite3.Connection, entries: list[BibEntry]) -> dict:
    """Linked papers whose document contradicts their bib entry.

    Flags a paper when the bib title is not in the document but the stored
    title, or a high-confidence layout title, is. A layout title comes from the
    document, so it is in the document by construction: ``basis`` says which
    title carried the flag, and ``layout`` alone is the weakest case. A
    low-confidence layout title carries nothing. Listed strongest first.
    """
    by_key = {e.citekey: e for e in entries}
    rows = conn.execute(
        f"SELECT p.id, p.filename, p.title, p.title_from_layout, p.title_layout_confidence, "
        f"p.citekey, p.citekey_source, p.metadata_source, "
        f"SUBSTR(pt.raw_text, 1, {CONTAINMENT_HEAD_CHARS}) AS head "
        "FROM papers p JOIN paper_text pt ON pt.paper_id = p.id "
        "WHERE p.citekey IS NOT NULL ORDER BY p.id"
    ).fetchall()

    flagged = []
    checked = 0
    for row in rows:
        entry = by_key.get(row["citekey"])
        if entry is None or not entry.title or not row["head"]:
            continue
        checked += 1
        bib_score = title_containment(entry.title, row["head"])
        if bib_score >= CONTAINMENT_THRESHOLD:
            continue
        stored = title_containment(row["title"], row["head"]) >= CONTAINMENT_THRESHOLD
        hint = row["title_from_layout"] if row["title_layout_confidence"] == "high" else None
        layout = title_containment(hint, row["head"]) >= CONTAINMENT_THRESHOLD
        if not (stored or layout):
            continue
        flagged.append({
            "id": row["id"],
            "filename": row["filename"],
            "citekey": row["citekey"],
            "citekey_source": row["citekey_source"],
            "bib_title": entry.title,
            "stored_title": row["title"],
            "layout_title": row["title_from_layout"],
            "bib_containment": round(bib_score, 3),
            "basis": "both" if stored and layout else "stored" if stored else "layout",
            "title_similarity": round(
                max(title_similarity(row["title"], entry.title),
                    title_similarity(row["title_from_layout"], entry.title)), 3
            ),
        })
    order = {"both": 0, "stored": 1, "layout": 2}
    flagged.sort(key=lambda f: (order[f["basis"]], f["title_similarity"], f["id"]))
    by_basis: dict[str, int] = {}
    for f in flagged:
        by_basis[f["basis"]] = by_basis.get(f["basis"], 0) + 1
    return {"checked": checked, "count": len(flagged), "by_basis": by_basis, "flagged": flagged}


def clear_audited(
    conn: sqlite3.Connection,
    entries: list[BibEntry],
    paper_ids: list[int],
    log_path: Path,
    dry_run: bool = False,
) -> dict:
    """Remove citekeys from the named flagged papers, logging what was there.

    Ids the audit does not flag are refused. Each removal is appended to
    ``log_path`` as a JSON line before the write, so the old key is recoverable
    even if the commit is the last thing to happen.
    """
    flagged = {f["id"]: f for f in audit_links(conn, entries)["flagged"]}
    chosen = [flagged[i] for i in paper_ids if i in flagged]
    result = {
        "dry_run": dry_run,
        "cleared": chosen,
        "not_flagged": [i for i in paper_ids if i not in flagged],
        "log_path": str(log_path),
    }
    if dry_run or not chosen:
        return result

    now = now_iso()
    with open(log_path, "a", encoding="utf-8") as log:
        for f in chosen:
            log.write(json.dumps({"at": now, **f}) + "\n")
    conn.executemany(
        "UPDATE papers SET citekey = NULL, citekey_source = NULL, updated_at = ? WHERE id = ?",
        [(now, f["id"]) for f in chosen],
    )
    conn.commit()
    return result


def _entry_values(entry: BibEntry) -> dict:
    """Fields a bib entry can supply, without the ones it lacks."""
    values = {
        "title": entry.title,
        "authors": split_authors(entry.authors) or None,
        "year": entry.year,
        "doi": normalize_doi(entry.doi) if entry.doi else None,
        "abstract": entry.abstract,
    }
    return {k: v for k, v in values.items() if v}


def _field_changes(row, values: dict) -> dict:
    changes = {}
    for field, new in values.items():
        old = normalize_authors(row[field]) if field == "authors" else row[field]
        if field == "doi" and old and normalize_doi(old) == new:
            continue
        if old != new:
            changes[field] = {"old": old, "new": new}
    return changes


def pull_metadata(
    conn: sqlite3.Connection,
    entries: list[BibEntry],
    stem: str,
    paper_ids: list[int] | None = None,
    dry_run: bool = False,
) -> dict:
    """Copy title, authors, year, doi and abstract from entries to linked papers.

    Hand-set rows are skipped. Otherwise a value is filled or replaced only
    when the bib title is present in the document (containment >= 0.6), which
    stops a wrong citekey from importing a wrong record; those rows land in
    ``skipped.unverified`` for ``link audit``. A field the entry lacks keeps
    what the paper has. ``newly_titled_ids`` feeds ``embed --chunk --force``,
    since chunk text is prefixed with the title.
    """
    source = BIB_SOURCE_PREFIX + stem
    by_key = {e.citekey: e for e in entries}
    sql = (
        "SELECT p.*, "
        f"CASE WHEN p.metadata_source IS NOT NULL AND "
        f"{manual_condition().replace('metadata_source', 'p.metadata_source')} "
        "THEN 1 ELSE 0 END AS is_manual, "
        f"SUBSTR(pt.raw_text, 1, {CONTAINMENT_HEAD_CHARS}) AS head "
        "FROM papers p LEFT JOIN paper_text pt ON pt.paper_id = p.id "
    )
    params: list = []
    if paper_ids is not None:
        sql += f"WHERE p.id IN ({','.join('?' * len(paper_ids))}) "
        params.extend(paper_ids)
    else:
        sql += "WHERE p.citekey IS NOT NULL "
    rows = conn.execute(sql + "ORDER BY p.id", params).fetchall()

    skipped: dict[str, list[int]] = {
        "unlinked": [], "not_in_bib": [], "manual": [], "unverified": [], "no_change": [],
    }
    updated: list[dict] = []
    newly_titled: list[int] = []
    now = now_iso()

    for row in rows:
        pid = row["id"]
        if not row["citekey"]:
            skipped["unlinked"].append(pid)
            continue
        entry = by_key.get(row["citekey"])
        if entry is None:
            skipped["not_in_bib"].append(pid)
            continue
        if row["is_manual"]:
            skipped["manual"].append(pid)
            continue
        score = title_containment(entry.title, row["head"])
        if score < CONTAINMENT_THRESHOLD:
            skipped["unverified"].append(pid)
            continue
        values = _entry_values(entry)
        changes = _field_changes(row, values)
        if not changes and row["metadata_source"] == source and not row["metadata_suspect"]:
            skipped["no_change"].append(pid)
            continue

        updated.append({"id": pid, "citekey": row["citekey"], "changes": changes})
        if not row["title"] and "title" in changes:
            newly_titled.append(pid)
        if dry_run:
            continue
        assignments = {
            k: (json.dumps(v["new"]) if k == "authors" else v["new"])
            for k, v in changes.items()
        }
        assignments.update({
            "metadata_source": source,
            "metadata_enriched_at": now,
            "metadata_suspect": 0,
            "metadata_verify_score": round(score, 3),
            "metadata_verified_at": now,
            "updated_at": now,
        })
        sets = ", ".join(f"{col} = ?" for col in assignments)
        conn.execute(f"UPDATE papers SET {sets} WHERE id = ?", [*assignments.values(), pid])
        refresh_row(conn, pid)
    conn.commit()

    return {
        "dry_run": dry_run,
        "metadata_source": source,
        "considered": len(rows),
        "updated_ids": [u["id"] for u in updated],
        "newly_titled_ids": newly_titled,
        "skipped": skipped,
        "updates": updated,
    }


def missing_entries(conn: sqlite3.Connection, entries: list[BibEntry]) -> list[dict]:
    """Bib entries matching no paper by citekey, file= basename or doi."""
    keys, files, dois = set(), set(), set()
    for r in conn.execute("SELECT citekey, filename, doi FROM papers"):
        if r["citekey"]:
            keys.add(r["citekey"])
        files.add((r["filename"] or "").lower())
        if r["doi"]:
            dois.add(normalize_doi(r["doi"]))

    rows = []
    for e in sorted(entries, key=lambda x: x.citekey):
        filename = extract_bib_filename(e.file)
        if e.citekey in keys or (filename and filename.lower() in files):
            continue
        if e.doi and normalize_doi(e.doi) in dois:
            continue
        rows.append({
            "citekey": e.citekey,
            "type": e.entry_type,
            "year": e.year,
            "author": "; ".join(split_authors(e.authors)) or None,
            "title": e.title,
            "doi": e.doi,
        })
    return rows
