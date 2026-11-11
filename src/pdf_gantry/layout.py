"""Titles read from page typography, not from linearized text.

``paper_text.raw_text`` has already lost the feature a reader uses to find a
title: it is the largest text near the top of the first page. This module
re-opens the PDF, so it belongs to the write path only (``process``,
``pipeline``, ``titles backfill``); the read path stays PDF-free and sees just
the stored ``papers.title_from_layout``.

The result is an unverified hint. No provider or person stands behind it, and
a bib-joined eval flatters it: those papers are the well-formed ones. So it
never reaches ``papers.title``, and ``title_layout_confidence`` says whether
font contrast backed the pick. Returning nothing beats returning garbage.
"""

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .fts import refresh_row
from .metadata import (
    _AFFILIATION_RE,
    _FRONT_MATTER_END_RE,
    _LINE_JUNK_RE,
    corpus_boilerplate,
    normalize_line,
)
from .queue import layout_pending_condition
from .utils import now_iso

logger = logging.getLogger(__name__)

TOP_FRACTION = 0.6  # a title sits in the top ~60% of its page
MAX_PAGES = 2  # page 2 only when page 1 yields nothing
MAX_COVER_PAGES = 2  # cover sheets skipped on top of that
MIN_TITLE_CHARS = 8
MAX_RUN_LINES = 6
MAX_TITLE_CHARS = 300
NO_CONTRAST_MAX_CHARS = 200
CONTRAST_RATIO = 1.15  # title size over body size for 'high' confidence
SIZE_TOLERANCE = 0.6  # points: same-size lines differ by rounding only
SMALLER_LINE_RATIO = 0.8  # a wrapped title may drop to this fraction of its size
LINE_GAP = 2.2  # same-size lines this many sizes apart are still adjacent
FAR_GAP = 4.5  # how far a dangling function word can reach
SHORT_WORDS = 2  # a run this short yields to a full title below it

_FUNCTION_WORD_END_RE = re.compile(
    r"(?:\b(?:a|an|and|as|at|by|for|from|in|into|of|on|or|the|to|under|versus|vs|with"
    r"|within|without|et|de|du|des|la|le|les|und|der|die|das)|[-:,–—])$",
    re.I,
)
_SECTION_MARKER_RE = re.compile(
    r"^\s*(title|chapter|article|section|part|book|annex|appendix)\s+"
    r"([0-9]+|[ivxlc]+)\b",
    re.I,
)
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|"
    "december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_RE = re.compile(
    rf"^\W*(?:[\w .'-]{{0,30}},\s*)?(?:\d{{1,2}}(?:st|nd|rd|th)?\s+)?(?:{_MONTHS})\.?"
    rf"(?:\s+\d{{1,2}}(?:st|nd|rd|th)?,?)?\s+\d{{4}}\W*$",
    re.I,
)
_LABEL_RE = re.compile(
    r"^\W*(?:(?:focus|review|research|original|feature|short|brief|invited|special|"
    r"policy|perspective|discussion|working|technical|conference|white|position|"
    r"opinion|commentary|editorial|regular|full)\s+"
    r"(?:article|paper|papers|report|essay|issue|communication|section|note|brief|"
    r"contribution|series)"
    r"|(?:(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|"
    r"twelfth|\d+(?:st|nd|rd|th))\s+edition)"
    r"|(?:online isbn|print isbn|e-?isbn|e-?issn)\b.*"
    r"|table of contents|contents|editorial|abstract|introduction|foreword|preface"
    r")\W*$",
    re.I,
)
_COPYRIGHT_RE = re.compile(
    r"©|\(c\)\s*\d{4}|\bcopyright\b|\ball rights reserved\b|\blicen[sc]ed under\b|"
    r"\bcreative commons\b",
    re.I,
)
_COVER_RE = re.compile(
    r"bluebook\s+\d+|heinonline|interlibrary\s+(?:services|loan)|\billiad\b|"
    r"document delivery|date downloaded:",
    re.I,
)
# A printed citation for the paper, not its title.
_CITATION_RE = re.compile(
    r"\b(?:volume|vol\.?)\s*\d|\bpp?\.\s*\d|,\s*(?:19|20)\d\d\W*$|\(\s*(?:19|20)\d\d\s*\)\s*\d",
    re.I,
)
_NOT_TITLE_RE = re.compile(
    r"\bauthor\S*\s*:|\b(?:recommended|suggested) citation\b|\bhow to cite\b|"
    r"\bworking paper series\b|\b(?:scholarly|digital) commons\b|\beditorial board\b",
    re.I,
)
# The whole line is a periodical's name: a provider may even hold a record under
# it, so the enrich rescue gate cannot catch this one downstream.
_PERIODICAL_RE = re.compile(
    r"^(?:the\s+)?(?:journal of\s+[\w&,' -]{3,60}"
    r"|(?:[\w&'-]+\s+){1,5}(?:law review|law journal|journal|quarterly)(?:\s+forum)?)$",
    re.I,
)
_MASTHEAD_END_RE = re.compile(r"journal homepage|homepage:\s*www", re.I)
_PRIVATE_USE_RE = re.compile("[\ue000-\uf8ff]")
_CHAPTER_PREFIX_RE = re.compile(r"^chapter title:\s*", re.I)  # JSTOR chapter covers
_NUMBERED_CLAUSE_RE = re.compile(r"^\d+\.\s")
_FEDERAL_REGISTER_RE = re.compile(r"Federal Register\s*/\s*Vol\.\s*\d+", re.I)
# Bold lines in a Federal Register document's opening block that are not title.
_FR_NOT_TITLE_RE = re.compile(r"^\[|^RIN\b|^\d+\s+CFR\b|^BILLING CODE\b", re.I)
_EDGE_MARKS = " *†‡§¶"
_AFFIL_MARK_RE = re.compile(r"\s+[a-z](?:,\s?[a-z*]+)*\*?(?=,|$)")
_INITIAL_RE = re.compile(r"(?:^|\s)[A-Z]\.(?:\s|$)")
_CID_RE = re.compile(r"\(cid:\d+\)")
_LETTERSPACED_RE = re.compile(r"(?:\S ){1,}\S")
_ASCII_WORD_RE = re.compile(r"^[A-Za-z]+$")
_VOWELS = set("aeiouyAEIOUY")
_CONSONANT_RUN_RE = re.compile(r"[b-df-hj-np-tv-xz]{6,}", re.I)


@dataclass
class LayoutTitle:
    title: str | None = None
    confidence: str | None = None  # 'high' | 'low', None without a title
    page: int | None = None


@dataclass
class _Line:
    y: float
    x: float
    size: float
    text: str


def clean_title(text: str, strip_marks: bool = True) -> str:
    """Normalize a title line: glyph debris, letterspacing, whitespace.

    ``strip_marks=False`` keeps trailing footnote marks, which on a raw line
    are evidence of a byline.
    """
    text = _PRIVATE_USE_RE.sub("", text.replace("\ufffd", "").replace("\u00ad", ""))
    # Letterspaced words are separated by a wider gap than their letters.
    words = [w.replace(" ", "") if _LETTERSPACED_RE.fullmatch(w) else w
             for w in re.split(r"\s{2,}", text.strip())]
    text = _CHAPTER_PREFIX_RE.sub("", " ".join(" ".join(words).split()))
    if strip_marks:
        text = text.rstrip(_EDGE_MARKS).lstrip(" :;,.-–—")
    tokens = text.split()
    half = len(tokens) // 2
    if half >= 2 and len(tokens) % 2 == 0 and tokens[:half] == tokens[half:]:
        text = " ".join(tokens[:half])  # repository covers print the name twice
    return text


def broken_encoding(page_text: str) -> bool:
    """True if a page's font encoding failed: many replacement characters."""
    return page_text.count("\ufffd") > max(5, 0.05 * len(page_text))


def _word_like(token: str) -> bool:
    if not _ASCII_WORD_RE.match(token):
        return token.isalpha()  # accented or non-Latin: nothing cheap to check
    if len(token) <= 3:
        return True
    return bool(_VOWELS & set(token)) and not _CONSONANT_RUN_RE.search(token)


def looks_garbled(text: str) -> bool:
    """True if text is a broken text layer rather than words.

    A wrong font encoding yields consonant soup, digits inside words, or
    ``(cid:N)`` placeholders. Checks shape, not vocabulary, so non-English
    titles pass.
    """
    if not text or _CID_RE.search(text) or text.count("\ufffd") > 1:
        return True
    tokens = [t.strip(".,:;()[]\"'“”‘’!?-–—") for t in text.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return True
    letters = sum(c.isalpha() for c in text)
    if letters < 0.5 * len(text.replace(" ", "")):
        return True
    wordish = sum(_word_like(t) for t in tokens)
    if wordish / len(tokens) < 0.6:
        return True
    ascii_letters = [c for c in text if c.isascii() and c.isalpha()]
    if len(ascii_letters) >= 12:
        vowels = sum(c in _VOWELS for c in ascii_letters) / len(ascii_letters)
        if vowels < 0.2:
            return True
    return False


def _is_byline(text: str) -> bool:
    """Positive evidence of people's names, strict enough for Title Case titles.

    ``metadata._looks_like_authors`` accepts any capitalised line with an
    "and", which is every other title here, so it can't be used on text that
    is already known to be large.
    """
    if text.rstrip().endswith(("†", "‡")):
        return True
    stripped = _AFFIL_MARK_RE.sub("", text.replace("*", ""))
    segments = [s.strip() for s in re.split(r",|;|&|\band\b", stripped) if s.strip()]
    if ":" in text or not segments:
        return False
    tokens_ok = all(
        2 <= len(seg.split()) <= 4
        and all(t[:1].isupper() or t in ("de", "van", "der", "von", "del") for t in seg.split())
        for seg in segments
    )
    if not tokens_ok:
        return False
    separated = len(segments) >= 2 and bool(re.search(r"[,;]", stripped))
    return separated or bool(_INITIAL_RE.search(stripped)) or stripped != text


def _span_join(spans: list[dict]) -> tuple[str, float]:
    """(text, size) for a line's spans, spacing them by geometry.

    Joining spans with a space turns small caps into "ECONO M ICS": each size
    change is a new span. A space goes in only where the page has a gap.
    Superscript footnote marks are dropped.
    """
    live = [s for s in spans if s["text"].strip()]
    if not live:
        return "", 0.0
    size = max(s["size"] for s in live)
    parts: list[str] = []
    previous = None
    for s in live:
        text = s["text"]
        if s["size"] < 0.7 * size and not any(c.isalpha() for c in text):
            continue
        if s["size"] < 0.7 * size and len(text.strip()) <= 2 and previous is not None:
            continue
        if previous is not None and not parts[-1].endswith(" ") and not text.startswith(" "):
            if s["bbox"][0] - previous["bbox"][2] > 0.2 * min(s["size"], previous["size"]):
                parts.append(" ")
        parts.append(text)
        previous = s
    return "".join(parts), size


def _page_lines(page) -> list[_Line]:
    lines: list[_Line] = []
    for block in page.get_text("dict", flags=0)["blocks"]:
        for line in block.get("lines", []):
            if abs(line.get("dir", (1, 0))[1]) > 0.1:  # arXiv stamps run up the margin
                continue
            text, size = _span_join(line["spans"])
            text = clean_title(text, strip_marks=False)
            if text:
                lines.append(_Line(line["bbox"][1], line["bbox"][0], size, text))
    lines.sort(key=lambda ln: (round(ln.y), ln.x))
    merged: list[_Line] = []
    for ln in lines:  # one visual line that the PDF split into pieces
        last = merged[-1] if merged else None
        if last and abs(ln.y - last.y) < 2 and abs(ln.size - last.size) <= SIZE_TOLERANCE:
            last.text = f"{last.text} {ln.text}"
        else:
            merged.append(ln)
    return merged


def _body_size(lines: list[_Line]) -> float:
    weights: dict[float, int] = {}
    for ln in lines:
        key = round(ln.size, 1)
        weights[key] = weights.get(key, 0) + len(ln.text)
    return max(weights, key=weights.get) if weights else 0.0


def _is_candidate(text: str, boilerplate) -> bool:
    if len(text) < 3 or "@" in text or normalize_line(text) in boilerplate:
        return False
    if _LINE_JUNK_RE.match(text) or _COPYRIGHT_RE.search(text):
        return False
    if _LABEL_RE.match(text) or _DATE_RE.match(text):
        return False
    if _CITATION_RE.search(text) or _NOT_TITLE_RE.search(text) or _PERIODICAL_RE.match(text):
        return False
    if text.count(";") >= 2 or (text[:1].islower() and text.count(",") >= 2):
        return False  # a keyword list
    return sum(c.isalpha() for c in text) >= 0.5 * len(text)


def _stops_run(line: _Line) -> bool:
    text = line.text
    return bool(
        _SECTION_MARKER_RE.match(text) or _FRONT_MATTER_END_RE.match(text)
        or _is_byline(text)
        or (_AFFILIATION_RE.search(text) and (text[0].isdigit() or "," in text))
    )


def _continues(run: list[_Line], nxt: _Line, size: float) -> bool:
    """Does ``nxt`` carry on the title that ``run`` has started?"""
    last = run[-1]
    gap = nxt.y - last.y
    if gap < 0 or _stops_run(nxt):
        return False
    dangling = bool(_FUNCTION_WORD_END_RE.search(last.text))
    if abs(nxt.size - size) <= SIZE_TOLERANCE:
        return gap < size * (FAR_GAP if dangling else LINE_GAP)
    if SMALLER_LINE_RATIO * size <= nxt.size < size and gap < size * LINE_GAP:
        return dangling or (last.text.isupper() and nxt.text.isupper())
    return False


def _run_from(lines: list[_Line], start: int) -> list[_Line]:
    size = lines[start].size
    run = [lines[start]]
    for nxt in lines[start + 1:]:
        if nxt.size < SMALLER_LINE_RATIO * size:
            continue  # a sidebar set beside the title; the gap test bounds this
        if len(run) >= MAX_RUN_LINES or not _continues(run, nxt, size):
            break
        run.append(nxt)
    return run


def _sentence_like(text: str) -> bool:
    words = text.split()
    return len(words) >= 8 and text.rstrip().endswith((".", ";"))


def _acceptable(text: str, size: float, body: float) -> bool:
    contrast = body > 0 and size >= body * CONTRAST_RATIO
    if len(text) < MIN_TITLE_CHARS or len(text) > MAX_TITLE_CHARS:
        return False
    if _sentence_like(text) or looks_garbled(text):
        return False
    if not contrast and (
        len(text) > NO_CONTRAST_MAX_CHARS or text[:1].islower()
        or _NUMBERED_CLAUSE_RE.match(text)
    ):
        return False
    return not _is_byline(text)


def _title_from_lines(lines: list[_Line], height: float, boilerplate) -> LayoutTitle | None:
    body = _body_size(lines)
    top = [ln for ln in lines if ln.y < height * TOP_FRACTION]
    ends = [ln.y for ln in top if _MASTHEAD_END_RE.search(ln.text)]
    if ends:  # publisher mastheads end at their homepage line
        top = [ln for ln in top if ln.y > max(ends)]
    cand = [ln for ln in top if _is_candidate(ln.text, boilerplate)]
    if not cand:
        return None

    runs: list[tuple[str, float]] = []
    for size in sorted({round(ln.size, 1) for ln in cand}, reverse=True):
        for i, ln in enumerate(cand):
            if abs(ln.size - size) > SIZE_TOLERANCE:
                continue
            if _SECTION_MARKER_RE.match(ln.text):
                continue
            text = " ".join(x.text for x in _run_from(cand, i))
            if _acceptable(text, ln.size, body):
                runs.append((text, ln.size))
                break
    if not runs:
        return None

    title, size = runs[0]
    if len(title.split()) <= SHORT_WORDS:
        # A series name or section label can outsize the title it sits above.
        for text, other in runs[1:]:
            if len(text.split()) >= 4 and other >= body * CONTRAST_RATIO:
                title, size = text, other
                break
    contrast = body and size >= body * CONTRAST_RATIO
    dangling = _FUNCTION_WORD_END_RE.search(title)
    confidence = "high" if contrast and not dangling else "low"
    return LayoutTitle(clean_title(title), confidence)




def _federal_register_title(page) -> str | None:
    """Title of the Federal Register document a PDF is for, or None.

    A govinfo PDF starts on the page where its document begins, so page 1
    often ends the previous document, and may hold a whole short one, before
    ours starts. Ours is the last to start there. Each document opens with
    one bold block: agency, docket or RIN line, title, then the AGENCY:
    label. Lines are read in content order, which in these PDFs runs column
    by column; sorting by height would interleave the columns.
    """
    lines: list[tuple[str, bool, float]] = []  # (text, bold, y)
    for block in page.get_text("dict", flags=0)["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = " ".join("".join(s["text"] for s in spans).split())
            bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
            lines.append((text, bold, line["bbox"][1]))
    top = 0.1 * (page.rect.height or 792)
    if not any(y < top and _FEDERAL_REGISTER_RE.search(t) for t, _, y in lines):
        return None
    starts = [i for i, (t, _, _) in enumerate(lines) if t.startswith("AGENCY:")]
    if not starts:
        return None
    parts: list[str] = []
    for text, bold, _ in reversed(lines[:starts[-1]]):
        if not bold or text.isupper() or _FR_NOT_TITLE_RE.match(text):
            break
        parts.append(text)
    title = ""
    for part in reversed(parts):  # "AI-" + "Assisted" keeps its hyphen
        title = f"{title}{part}" if title.endswith("-") or not title else f"{title} {part}"
    title = clean_title(title)
    return title if len(title) >= MIN_TITLE_CHARS else None


def extract_layout_title(pdf_path: Path, boilerplate=frozenset()) -> LayoutTitle:
    """Pick a title from the first pages of a PDF.

    A damaged or locked file gives an empty result, which callers may record
    as attempted. A file that isn't there raises ``FileNotFoundError``: that
    says nothing about the PDF, so the attempt must not be recorded.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        doc = pymupdf.open(str(path))
    except Exception:
        return LayoutTitle()
    try:
        if doc.needs_pass or doc.page_count == 0:
            return LayoutTitle()
        federal_register = _federal_register_title(doc[0])
        if federal_register:
            return LayoutTitle(federal_register, "high", 1)
        tried = 0
        covers = 0
        for number in range(min(doc.page_count, MAX_PAGES + MAX_COVER_PAGES)):
            page = doc[number]
            text = page.get_text()
            if covers < MAX_COVER_PAGES and _COVER_RE.search(text):
                covers += 1
                continue
            found = None
            if not broken_encoding(text):
                found = _title_from_lines(
                    _page_lines(page), page.rect.height or 792, boilerplate
                )
            if found:
                found.page = number + 1
                return found
            tried += 1
            if tried >= MAX_PAGES:
                break
        return LayoutTitle()
    except (RuntimeError, ValueError, IndexError, KeyError) as e:  # unparseable page
        logger.debug("layout title failed for %s: %s", path, e)
        return LayoutTitle()
    finally:
        doc.close()


def backfill_titles(
    conn: sqlite3.Connection,
    papers_dir: Path,
    paper_ids: list[int] | None = None,
    force: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    progress_callback=None,
) -> dict:
    """Compute layout titles for papers not yet attempted.

    Writes only the layout columns, and refreshes the keyword index for papers
    whose indexed title that changes. Corpus boilerplate is computed once.
    ``force`` recomputes papers already attempted. A missing or unreadable file
    is reported in ``errors`` and left pending; a PDF that simply yields no
    title is stamped so it isn't retried.
    """
    where: list[str] = []
    params: list = []
    if not force:
        where.append(layout_pending_condition())
    if paper_ids is not None:
        where.append(f"id IN ({','.join('?' * len(paper_ids))})")
        params.extend(paper_ids)
    sql = "SELECT id, path, filename, title, title_from_layout FROM papers"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()

    report = {
        "dry_run": dry_run, "selected": len(rows), "titled": 0, "high": 0, "low": 0,
        "empty": 0, "written": 0, "errors": [],
    }
    if not rows:
        return report

    pymupdf.TOOLS.mupdf_display_errors(False)
    boilerplate = corpus_boilerplate(conn)
    for i, row in enumerate(rows, 1):
        try:
            found = extract_layout_title(Path(papers_dir) / row["path"], boilerplate)
        except Exception as e:  # missing or unreadable: not evidence about the PDF
            report["errors"].append({
                "id": row["id"], "filename": row["filename"],
                "error": f"{type(e).__name__}: {e}"[:300],
            })
            found = None
        if found is not None:
            if found.title:
                report["titled"] += 1
                report[found.confidence] += 1
            else:
                report["empty"] += 1
            if not dry_run:
                conn.execute(
                    "UPDATE papers SET title_from_layout = ?, title_layout_confidence = ?, "
                    "title_layout_at = ? WHERE id = ?",
                    (found.title, found.confidence, now_iso(), row["id"]),
                )
                if not row["title"] and found.title != row["title_from_layout"]:
                    refresh_row(conn, row["id"])
                report["written"] += 1
                if report["written"] % 50 == 0:
                    conn.commit()
        if progress_callback:
            progress_callback(i, len(rows))
    conn.commit()
    return report
