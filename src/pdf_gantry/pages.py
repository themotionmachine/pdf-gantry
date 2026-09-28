"""Chunk page provenance: map chunk text back to the PDF pages it came from.

Chunks are cut from markdown (PyMuPDF4LLM or Marker), which carries no page
boundaries. The per-page plain text from PyMuPDF does. This module locates
each chunk in the per-page text by probing a normalised form of the chunk
(letters only, casefolded) against the normalised concatenation of all
pages, and reads the page off the hit's offset.

The method is upstreamed from p3_corpus/chunk_pages.py, which validated it on
179k chunks (98.6% located or partial). Passes:

1. Substring probes: a head probe (80/50/30 letters), confirmed by a tail
   probe searched forward. Probes search forward from the previous in-order
   hit, so running headers repeated on every page resolve to the right one.
   If the head misses, mid-chunk probes at 25/50/75%.
1b. A head hit without tail confirmation may run onto following pages;
   extend page_end while the next page still carries the chunk's words.
1c. Validate probe hits by word overlap; a running header or a table of
   contents line can match on the wrong page. Weak hits move to a clearly
   better page.
2. Word-overlap fallback (tables, reflowed columns) inside the window
   between located neighbours.
3. Neighbour interpolation for what is left (short chunks, image
   placeholders): page range = previous page_end .. next page_start.

Statuses: located, partial, interpolated, unlocated, no_text.

OCR output is different: ``ocr.ocr_document`` writes one ``## Page N``
section per page, so every OCR chunk already knows its page from its
section header. ``assign_pages_from_ocr_headers`` uses that directly.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PROBES = (80, 50, 30)   # head/tail probe lengths, longest first
MIN_CHARS = 20          # below this (normalised) a chunk can only be interpolated
MAX_SPAN = 6            # a chunk never spans more than this many pages
MAX_JUMP = 20           # a head-only hit this far past the last page needs word support
WORDS_MIN = 8           # word-overlap checks need at least this many distinct words
WORDS_ACCEPT = 0.5      # pass 2: fraction of words on one page to accept it
WORDS_EXTEND = 0.4      # pass 1b: extend onto the next page above this overlap
WORDS_WEAK = 0.6        # pass 1c: a probe hit below this overlap is re-checked ...
WORDS_STRONG = 0.8      # ... and moved only to a page at least this good (and +0.3)

# Letters only: digits go too, because PDFs with margin line numbers merge
# them into words ("commu22 nication") in one extraction but not the other.
_NORM = re.compile(r"[\W\d_]+")
_WORD = re.compile(r"[^\W\d_]{5,}")
_OCR_PAGE_HEADER = re.compile(r"^Page (\d+)$")

LOCATED = "located"
PARTIAL = "partial"
INTERPOLATED = "interpolated"
UNLOCATED = "unlocated"
NO_TEXT = "no_text"


@dataclass
class PageSpan:
    """Where one chunk sits in the PDF (1-based physical pages)."""
    page_start: int | None
    page_end: int | None
    status: str
    method: str = ""


def norm(s: str | None) -> str:
    """Letters only, casefolded, ligatures expanded (NFKC)."""
    return _NORM.sub("", unicodedata.normalize("NFKC", s or "").casefold())


def words(s: str | None) -> set[str]:
    return set(_WORD.findall(unicodedata.normalize("NFKC", s or "").casefold()))


def pdf_page_texts(pdf_path: Path | str) -> list[str]:
    """Per-page plain text from PyMuPDF (the same call process.py uses for raw_text)."""
    # `import pymupdf`, not `import fitz`: on PyMuPDF >= 1.26 the fitz alias
    # prints a deprecation line to stdout, which corrupts --json output.
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    doc = pymupdf.open(str(pdf_path))
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def page_from_ocr_header(section_header: str | None) -> int | None:
    """The page number from an OCR ``## Page N`` section header, else None."""
    if not section_header:
        return None
    m = _OCR_PAGE_HEADER.match(section_header.strip())
    return int(m.group(1)) if m else None


def assign_pages_from_ocr_headers(chunks: list) -> list:
    """Set page_start/page_end on OCR chunks from their ``Page N`` header, in place."""
    for c in chunks:
        page = page_from_ocr_header(c.section_header)
        if page is not None:
            c.page_start = c.page_end = page
    return chunks


def _page_at(starts: list[int], pos: int) -> int:
    """1-based page containing normalised-text offset pos (starts = cumulative)."""
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def _find_probe(full: str, probe: str, cursor: int) -> tuple[int, bool]:
    """First occurrence at/after cursor, else anywhere. Returns (pos, in_order)."""
    p = full.find(probe, cursor)
    if p >= 0:
        return p, True
    return full.find(probe), False


def _tail_page(full, n, hpos, hlen, starts, ps):
    """Confirm a head hit with a tail probe searched forward. Returns (page_end, plen)."""
    for plen in PROBES:
        if len(n) < plen + hlen:
            continue
        tpos = full.find(n[-plen:], hpos + hlen)
        if tpos >= 0:
            pt = _page_at(starts, tpos)
            if 0 <= pt - ps <= MAX_SPAN:
                return pt, plen
        return None, 0
    return None, 0


def _window(rows: list[PageSpan], k: int, n_pages: int, strict: bool = False):
    """Pages between the nearest located/partial neighbours of rows[k].
    strict=True returns (None, None) when neither neighbour exists."""
    lo = hi = None
    for j in range(k - 1, -1, -1):
        if rows[j].status in (LOCATED, PARTIAL):
            lo = rows[j].page_end
            break
    for j in range(k + 1, len(rows)):
        if rows[j].status in (LOCATED, PARTIAL):
            hi = rows[j].page_start
            break
    if lo is None and hi is None:
        return (None, None) if strict else (1, n_pages)
    if lo is None:
        lo = hi
    if hi is None:
        hi = lo
    if hi < lo:
        hi = lo
    if not strict:  # widen by one page each side for the word search
        lo, hi = max(1, lo - 1), min(n_pages, hi + 1)
    return lo, hi


def map_chunks_to_pages(pages_raw: list[str], chunks: list[str]) -> list[PageSpan]:
    """Map chunk texts (in document order) to 1-based page ranges.

    ``pages_raw`` is the per-page text of the PDF; ``chunks`` the chunk texts.
    Returns one PageSpan per chunk, in order.
    """
    pages = [norm(t) for t in pages_raw]
    starts, acc = [], 0
    for t in pages:
        starts.append(acc)
        acc += len(t)
    full = "".join(pages)
    if not full:
        return [PageSpan(None, None, NO_TEXT) for _ in chunks]

    page_words: list[set[str]] | None = None

    def pw() -> list[set[str]]:
        nonlocal page_words
        if page_words is None:
            page_words = [words(t) for t in pages_raw]
        return page_words

    rows: list[PageSpan] = []
    cursor = 0
    last_page = None  # page_end of the last in-order located chunk

    # --- pass 1: substring probes
    for text in chunks:
        n = norm(text)
        row = PageSpan(None, None, UNLOCATED)
        rows.append(row)
        if len(n) < MIN_CHARS:
            row.method = "short"
            continue
        hpos, hlen, in_order = -1, 0, True
        for plen in PROBES:
            if len(n) < plen:
                continue
            hpos, in_order = _find_probe(full, n[:plen], cursor)
            if hpos >= 0:
                hlen = plen
                break
        if hpos < 0 and len(n) < PROBES[-1] * 2:  # short chunk: try the whole text
            hpos, in_order = _find_probe(full, n, cursor)
            hlen = len(n) if hpos >= 0 else 0
        if hpos >= 0:
            ps = _page_at(starts, hpos)
            pe, tail_len = _tail_page(full, n, hpos, hlen, starts, ps)
            # A head-only hit far ahead of the last located page is suspect
            # (boilerplate repeated later); accept only with word support.
            far = pe is None and last_page is not None and ps - last_page > MAX_JUMP
            if far:
                cw = words(text)
                far = (len(cw) < WORDS_MIN
                       or len(cw & pw()[ps - 1]) / len(cw) < WORDS_STRONG)
            if (in_order or pe is not None) and not far:
                row.page_start = ps
                row.page_end = pe if pe is not None else ps
                row.status = LOCATED
                row.method = (f"head{hlen}" + (f"+tail{tail_len}" if pe is not None else "")
                              + ("" if in_order else "+ooo"))
                if in_order:
                    cursor = hpos
                    last_page = row.page_end
                continue
        # mid probes at 25/50/75 %, in order only
        hits = []
        for frac in (0.25, 0.5, 0.75):
            off = int(len(n) * frac)
            for plen in PROBES:
                if off + plen > len(n):
                    continue
                mpos = full.find(n[off:off + plen], cursor)
                if mpos >= 0:
                    hits.append(_page_at(starts, mpos))
                    break
        if hits and max(hits) - min(hits) <= MAX_SPAN:
            row.page_start, row.page_end = min(hits), max(hits)
            row.status, row.method = PARTIAL, f"mid{len(hits)}"

    n_pages = len(pages)

    # --- pass 1b: extend head-only hits onto following pages
    for k, row in enumerate(rows):
        if row.status != LOCATED or "tail" in row.method:
            continue
        cw = words(chunks[k])
        if len(cw) < WORDS_MIN:
            continue
        pe = row.page_end
        while pe < n_pages and pe - row.page_start < MAX_SPAN:
            if len(cw & pw()[pe]) / len(cw) < WORDS_EXTEND:  # index pe == page pe+1
                break
            pe += 1
        if pe != row.page_end:
            row.page_end = pe
            row.method += "+ext"

    # --- pass 1c: validate probe hits by word overlap
    for k, row in enumerate(rows):
        if row.status not in (LOCATED, PARTIAL):
            continue
        cw = words(chunks[k])
        if len(cw) < WORDS_MIN:
            continue
        on = set().union(*pw()[row.page_start - 1:row.page_end])
        ov = len(cw & on) / len(cw)
        if ov >= WORDS_WEAK:
            continue
        best, score = None, 0.0
        for p in range(1, n_pages + 1):
            s = len(cw & pw()[p - 1]) / len(cw)
            if s > score:
                best, score = p, s
        if best is None or score < WORDS_STRONG or score < ov + 0.3:
            continue
        ps = pe = best
        while (ps > 1 and pe - ps < MAX_SPAN
               and len(cw & pw()[ps - 2]) / len(cw) >= WORDS_EXTEND):
            ps -= 1
        while (pe < n_pages and pe - ps < MAX_SPAN
               and len(cw & pw()[pe]) / len(cw) >= WORDS_EXTEND):
            pe += 1
        row.page_start, row.page_end = ps, pe
        row.method += "+wfix"

    # --- pass 2: word overlap inside the neighbour window
    for k, row in enumerate(rows):
        if row.page_start is not None:
            continue
        cw = words(chunks[k])
        if len(cw) < WORDS_MIN:
            continue
        lo, hi = _window(rows, k, n_pages)
        best, score = None, 0.0
        for p in range(lo, hi + 1):
            s = len(cw & pw()[p - 1]) / len(cw)
            if s > score:
                best, score = p, s
        if best is not None and score >= WORDS_ACCEPT:
            row.page_start = row.page_end = best
            row.status, row.method = PARTIAL, f"words{int(score * 100)}"

    # --- pass 3: neighbour interpolation
    for k, row in enumerate(rows):
        if row.page_start is not None:
            continue
        lo, hi = _window(rows, k, n_pages, strict=True)
        if lo is None:
            continue
        row.page_start, row.page_end = lo, hi
        row.status, row.method = INTERPOLATED, "neighbors"

    return rows


# --- backfill ---------------------------------------------------------------

AMBIGUOUS_LIMIT = 50  # per-paper ambiguity rows returned in the report


def _paper_spans(conn, paper, chunk_rows, papers_dir: Path) -> list[PageSpan]:
    """Page spans for one paper's existing chunks (in chunk_index order)."""
    if paper["text_method"] == "surya":
        # OCR markdown: "## Page N" sections. The scan has no text layer to
        # probe, and doesn't need one.
        spans = []
        for c in chunk_rows:
            page = page_from_ocr_header(c["section_header"])
            spans.append(PageSpan(page, page, LOCATED if page else UNLOCATED, "ocr_header"))
        return spans
    page_texts = pdf_page_texts(Path(papers_dir) / paper["path"])
    return map_chunks_to_pages(page_texts, [c["text"] for c in chunk_rows])


def backfill_pages(
    conn,
    papers_dir: Path,
    paper_ids: list[int] | None = None,
    dry_run: bool = False,
    force: bool = False,
    limit: int | None = None,
    progress_callback=None,
) -> dict:
    """Compute page_start/page_end for chunks that already exist.

    Only the two page columns are written: chunk text, chunk ids, chunk_vec
    rows and has_chunk_embeddings are untouched, so nothing needs
    re-embedding. Opens the PDFs (write path only).

    By default only papers with at least one chunk lacking page_start are
    selected; ``force`` re-maps every selected paper.

    Returns a report dict: papers, chunks, assigned, updated, coverage,
    by_status, ambiguous (papers with unlocated/interpolated chunks, worst
    first), errors (papers that could not be mapped), dry_run.
    """
    where = ["EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = p.id)"]
    params: list = []
    if not force:
        where.append(
            "EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = p.id AND c.page_start IS NULL)"
        )
    if paper_ids is not None:
        where.append(f"p.id IN ({','.join('?' * len(paper_ids))})")
        params.extend(paper_ids)
    sql = (
        "SELECT p.id, p.path, p.filename, p.text_method FROM papers p WHERE "
        + " AND ".join(where) + " ORDER BY p.id"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    papers = conn.execute(sql, params).fetchall()

    by_status = {s: 0 for s in (LOCATED, PARTIAL, INTERPOLATED, UNLOCATED, NO_TEXT)}
    report = {
        "dry_run": dry_run, "papers": len(papers), "chunks": 0, "assigned": 0,
        "updated": 0, "coverage": None, "by_status": by_status,
        "ambiguous": [], "errors": [],
    }
    ambiguous = []

    for i, paper in enumerate(papers, 1):
        chunk_rows = conn.execute(
            "SELECT chunk_id, section_header, text FROM chunks "
            "WHERE doc_id = ? ORDER BY chunk_index",
            (paper["id"],),
        ).fetchall()
        try:
            spans = _paper_spans(conn, paper, chunk_rows, papers_dir)
        except Exception as e:  # missing/damaged PDF: record and keep going
            report["errors"].append({
                "doc_id": paper["id"], "filename": paper["filename"],
                "error": f"{type(e).__name__}: {e}"[:300],
            })
            report["chunks"] += len(chunk_rows)
            if progress_callback:
                progress_callback(i, len(papers))
            continue

        counts = {s: 0 for s in by_status}
        updates = []
        for c, span in zip(chunk_rows, spans):
            counts[span.status] += 1
            if span.page_start is not None:
                updates.append((span.page_start, span.page_end, c["chunk_id"]))
        for s, n in counts.items():
            by_status[s] += n
        report["chunks"] += len(chunk_rows)
        report["assigned"] += len(updates)
        doubtful = counts[UNLOCATED] + counts[INTERPOLATED] + counts[NO_TEXT]
        if doubtful:
            ambiguous.append({
                "doc_id": paper["id"], "filename": paper["filename"],
                "chunks": len(chunk_rows), "unlocated": counts[UNLOCATED],
                "interpolated": counts[INTERPOLATED], "no_text": counts[NO_TEXT],
            })
        if not dry_run:
            conn.executemany(
                "UPDATE chunks SET page_start = ?, page_end = ? WHERE chunk_id = ?",
                updates,
            )
            conn.commit()
            report["updated"] += len(updates)
        if progress_callback:
            progress_callback(i, len(papers))

    ambiguous.sort(key=lambda a: -(a["unlocated"] + a["interpolated"] + a["no_text"]) / a["chunks"])
    report["ambiguous_papers"] = len(ambiguous)
    report["ambiguous"] = ambiguous[:AMBIGUOUS_LIMIT]
    if report["chunks"]:
        report["coverage"] = round(report["assigned"] / report["chunks"], 4)
    return report
