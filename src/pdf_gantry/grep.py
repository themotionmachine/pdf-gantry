"""Literal (quote) search over chunk text: `gantry grep`.

The quote-verification primitive: confirm a passage is verbatim in doc D and
get its chunk and page. Pure DB; never opens a PDF.

Matching runs in two tiers per chunk:

- ``exact``: the needle occurs as typed (optionally case-insensitive).
- ``normalized``: both sides are normalised first, so the extraction's line
  breaks, line-end hyphenation ("engage-\\nment"), ligatures ("\ufb01"), markdown
  emphasis ("**word**"), curly quotes and dashes don't defeat a real quote.
  Offsets map back to the original chunk text, so ``matched_text`` is the
  verbatim source span.

Papers with no chunks fall back to ``paper_text.raw_text`` (no chunk, no
page); every hit says which source it came from.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata

_SKIP = set("\u00ad\u200b\u200c\u200d\ufeff*_`")  # soft hyphen, zero-width, markdown
_HYPHENS = set("-\u2010\u2011")  # dropped between letters
_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ",
}
_BR = "<br>"  # pymupdf4llm writes line breaks inside table cells as <br>
_WORD = re.compile(r"[^\W\d_]+")
_LIGATURE_PAIRS = re.compile(r"ff|fi|fl")

DEFAULT_CONTEXT = 80


def normalize_with_map(text: str, fold: bool = False) -> tuple[str, list[int]]:
    """Normalise ``text`` for tolerant matching; also return, for every output
    character, the index of the source character it came from.

    Whitespace runs (and <br>) collapse to one space; a hyphen between letters
    is dropped along with any whitespace after it (so "self-determination",
    "selfdetermination" and "self-\\ndetermination" all agree); NFKC expands
    ligatures; curly quotes and dashes become ASCII; markdown emphasis, soft
    hyphens and zero-width characters vanish. ``fold`` casefolds.
    """
    out: list[str] = []
    idx: list[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "<" and text.startswith(_BR, i):
            if out and out[-1] != " ":
                out.append(" ")
                idx.append(i)
            i += len(_BR)
            continue
        if ch in _SKIP:
            i += 1
            continue
        if ch in _HYPHENS and out and out[-1].isalpha():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].isalpha():
                i = j
                continue
        ch = _MAP.get(ch, ch)
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                idx.append(i)
            i += 1
            continue
        s = unicodedata.normalize("NFKC", ch)
        if fold:
            s = s.casefold()
        for c in s:
            out.append(c)
            idx.append(i)
        i += 1
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def _find_exact(text: str, needle: str, ignore_case: bool) -> list[tuple[int, int]]:
    if ignore_case:
        return [(m.start(), m.end()) for m in
                re.finditer(re.escape(needle), text, re.IGNORECASE)]
    spans, pos = [], text.find(needle)
    while pos >= 0:
        spans.append((pos, pos + len(needle)))
        pos = text.find(needle, pos + 1)
    return spans


def _find_normalized(text: str, nneedle: str, ignore_case: bool) -> list[tuple[int, int]]:
    ntext, idx = normalize_with_map(text, fold=ignore_case)
    spans, pos = [], ntext.find(nneedle)
    while pos >= 0:
        spans.append((idx[pos], idx[pos + len(nneedle) - 1] + 1))
        pos = ntext.find(nneedle, pos + 1)
    return spans


_NON_ALNUM = re.compile(r"[\W_]+")


def _squash(s: str, fold: bool) -> str:
    """Letters and digits only (NFKC). A normalised match implies a match of
    the squashed forms, and squashing runs in C, so it screens candidates
    before the per-character normalisation."""
    s = _NON_ALNUM.sub("", unicodedata.normalize("NFKC", s))
    return s.casefold() if fold else s


def _find(text: str, needle: str, nneedle: str, ignore_case: bool):
    """(spans, match_kind) for one text: exact first, normalised if none."""
    spans = _find_exact(text, needle, ignore_case)
    if spans:
        return spans, "exact"
    squashed = _squash(needle, ignore_case)
    if squashed and squashed not in _squash(text, ignore_case):
        return [], None
    if nneedle:
        spans = _find_normalized(text, nneedle, ignore_case)
        if spans:
            return spans, "normalized"
    return [], None


def _prefilter(column: str, needle: str) -> tuple[str, list]:
    """SQL condition that keeps every row that could match, cheaply.

    A row survives if it contains the needle verbatim, or enough of the
    needle's longest words (lowercased). Words are split at ff/fi/fl so a
    ligature in the source can't hide them, and one of three may be missing
    so a line-end hyphenation inside a word can't either.
    """
    frags = []
    for w in _WORD.findall(unicodedata.normalize("NFKC", needle).lower()):
        frags.extend(p for p in _LIGATURE_PAIRS.split(w) if len(p) >= 3)
    frags = sorted(set(frags), key=len, reverse=True)[:3]
    cond = f"instr({column}, ?) > 0"
    params: list = [needle]
    if frags:
        need = 2 if len(frags) == 3 else 1
        score = " + ".join(f"(instr(lower({column}), ?) > 0)" for _ in frags)
        cond = f"({cond} OR ({score}) >= {need})"
        params.extend(frags)
    else:
        cond = f"({cond} OR instr(lower({column}), ?) > 0)"
        params.append(needle.strip().lower())
    return cond, params


def _context(text: str, start: int, end: int, chars: int) -> str:
    return re.sub(r"\s+", " ", text[max(0, start - chars):end + chars]).strip()


def grep(
    conn: sqlite3.Connection,
    needle: str,
    doc_ids: list[int] | None = None,
    ignore_case: bool = False,
    limit: int | None = 50,
    context_chars: int = DEFAULT_CONTEXT,
) -> dict:
    """Find ``needle`` in chunk text (raw_text for papers without chunks).

    Returns {"query", "count", "truncated", "hits", "not_found"}. Each hit:
    doc_id, filename, chunk_id, chunk_index, page_start, page_end, offset,
    length (offset/length index the chunk text, or raw_text when
    source == "raw_text"), matched_text (the verbatim source span), match
    ("exact" | "normalized"), source ("chunks" | "raw_text"), context.
    A quote that sits in the overlap between consecutive chunks is reported
    once, in the earlier chunk.
    """
    if not needle or not needle.strip():
        raise ValueError("empty search string")
    nneedle, _ = normalize_with_map(needle, fold=ignore_case)
    hits: list[dict] = []
    cap = None if limit is None else limit + 1  # one extra to detect truncation

    scope, scope_params = "", []
    not_found: list[int] = []
    if doc_ids is not None:
        ph = ",".join("?" * len(doc_ids))
        scope, scope_params = f" AND c.doc_id IN ({ph})", list(doc_ids)
        found = {r[0] for r in conn.execute(
            f"SELECT id FROM papers WHERE id IN ({ph})", doc_ids)}
        seen = set()
        for i in doc_ids:
            if i not in found and i not in seen:
                not_found.append(i)
                seen.add(i)

    # Scoped greps scan every chunk of the named papers; unscoped greps
    # prefilter in SQL so the normalised pass only touches plausible chunks.
    cond, cond_params = ("1", []) if doc_ids is not None else _prefilter("c.text", needle)
    rows = conn.execute(
        f"""SELECT c.chunk_id, c.doc_id, c.chunk_index, c.page_start, c.page_end,
                   c.text, p.filename
            FROM chunks c JOIN papers p ON p.id = c.doc_id
            WHERE {cond}{scope}
            ORDER BY c.doc_id, c.chunk_index""",
        cond_params + scope_params,
    )

    prev = None  # (doc_id, chunk_index, text) of the last chunk that had a hit
    for r in rows:
        if cap is not None and len(hits) >= cap:
            break
        text = r["text"]
        spans, kind = _find(text, needle, nneedle, ignore_case)
        if not spans:
            continue
        adjacent = prev is not None and prev[0] == r["doc_id"] and prev[1] == r["chunk_index"] - 1
        prev_norm = normalize_with_map(prev[2], fold=ignore_case)[0] if adjacent else ""
        for start, end in spans:
            if adjacent:
                head = normalize_with_map(text[:end], fold=ignore_case)[0]
                if head and head in prev_norm:
                    continue  # the same passage, repeated by chunk overlap
            hits.append({
                "doc_id": r["doc_id"], "filename": r["filename"],
                "chunk_id": r["chunk_id"], "chunk_index": r["chunk_index"],
                "page_start": r["page_start"], "page_end": r["page_end"],
                "offset": start, "length": end - start,
                "matched_text": text[start:end],
                "match": kind, "source": "chunks",
                "context": _context(text, start, end, context_chars),
            })
        prev = (r["doc_id"], r["chunk_index"], text)

    # Fallback: papers that have text but no chunks.
    if cap is None or len(hits) < cap:
        cond, cond_params = (("1", []) if doc_ids is not None
                             else _prefilter("pt.raw_text", needle))
        raw_scope = scope.replace("c.doc_id", "pt.paper_id")
        rows = conn.execute(
            f"""SELECT pt.paper_id, pt.raw_text, p.filename
                FROM paper_text pt JOIN papers p ON p.id = pt.paper_id
                WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = pt.paper_id)
                  AND pt.raw_text IS NOT NULL AND {cond}{raw_scope}
                ORDER BY pt.paper_id""",
            cond_params + scope_params,
        )
        for r in rows:
            if cap is not None and len(hits) >= cap:
                break
            text = r["raw_text"]
            spans, kind = _find(text, needle, nneedle, ignore_case)
            for start, end in spans:
                hits.append({
                    "doc_id": r["paper_id"], "filename": r["filename"],
                    "chunk_id": None, "chunk_index": None,
                    "page_start": None, "page_end": None,
                    "offset": start, "length": end - start,
                    "matched_text": text[start:end],
                    "match": kind, "source": "raw_text",
                    "context": _context(text, start, end, context_chars),
                })

    truncated = limit is not None and len(hits) > limit
    if truncated:
        hits = hits[:limit]
    return {
        "query": needle, "count": len(hits), "truncated": truncated,
        "hits": hits, "not_found": not_found,
    }
