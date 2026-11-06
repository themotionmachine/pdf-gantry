"""BibTeX reconciliation and bibliography generation."""

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import bibtexparser

_CITEKEY_CLEAN_RE = re.compile(r"[^a-z0-9]")
_YEAR_RE = re.compile(r"\d{4}")


@dataclass
class BibEntry:
    citekey: str
    entry_type: str
    title: str | None
    authors: str | None
    year: int | None
    doi: str | None
    file: str | None
    abstract: str | None = None


@dataclass
class LinkMatch:
    paper_id: int
    filename: str
    citekey: str
    source: str
    confidence: str
    score: float
    paper_title: str | None
    bib_title: str | None


@dataclass
class LinkStats:
    total_papers: int = 0
    total_bib_entries: int = 0
    matched_doi: int = 0
    matched_filename: int = 0
    matched_title_certain: int = 0
    matched_title_uncertain: int = 0
    unmatched_papers: int = 0
    unmatched_bib: int = 0
    skipped_existing: int = 0
    applied: int = 0


_LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+\s*')
_BRACES_RE = re.compile(r"[{}]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_DOI_PREFIX_RE = re.compile(r"^(https?://(www\.)?(doi\.org|dx\.doi\.org)/|doi:\s*)", re.I)

# Letter accents (\v, \c, ...) need the negative lookahead: without it, the
# "\t" of \textit reads as a tie accent and leaves "extit".
_LATEX_ACCENT_RE = re.compile(
    r"""\\(?:(["'^`~=.])|([uvHcdbkr])(?![a-zA-Z]))\s*\{?\s*\\?([a-zA-Z])\}?"""
)
_COMBINING = {
    "'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303",
    "=": "\u0304", ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b",
    "c": "\u0327", "d": "\u0323", "b": "\u0331", "k": "\u0328", "r": "\u030a",
}
_LATEX_LETTERS = {
    "ss": "ß", "ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "aa": "å", "AA": "Å",
    "o": "ø", "O": "Ø", "l": "ł", "L": "Ł",
}
_LATEX_LETTER_RE = re.compile(r"\\(" + "|".join(_LATEX_LETTERS) + r")(?![a-zA-Z])\s*")
_LATEX_ESCAPE_RE = re.compile(r"\\([&%$#_])")
_CITATION_TAG_RE = re.compile(r"^\s*\[[A-Z]\]\s*")  # "[C] Title" reference-manager tags


def _accent(m: re.Match) -> str:
    return m.group(3) + _COMBINING[m.group(1) or m.group(2)]


def clean_bib_text(text: str) -> str:
    """A BibTeX field value as plain text."""
    text = _LATEX_ACCENT_RE.sub(_accent, text)
    text = _LATEX_LETTER_RE.sub(lambda m: _LATEX_LETTERS[m.group(1)], text)
    text = _LATEX_ESCAPE_RE.sub(r"\1", text)
    text = _LATEX_CMD_RE.sub("", text)
    text = _BRACES_RE.sub("", text)
    text = text.replace("``", '"').replace("''", '"').replace("`", "'").replace("~", " ")
    text = _CITATION_TAG_RE.sub("", text)
    return unicodedata.normalize("NFC", _MULTI_SPACE_RE.sub(" ", text).strip())


_strip_latex = clean_bib_text


def normalize_title(title: str) -> str:
    return _strip_latex(title).lower()


def normalize_doi(doi: str) -> str:
    doi = _DOI_PREFIX_RE.sub("", doi)
    doi = doi.rstrip(".,;:)]}")
    return doi.lower()


def extract_bib_filename(file_field: str | None) -> str | None:
    if not file_field:
        return None
    # JabRef/Mendeley format: :path/file.pdf:TYPE
    if file_field.startswith(":") and ":" in file_field[1:]:
        inner = file_field[1:file_field.rindex(":")]
        return Path(inner).name if inner else None
    # Plain path or bare filename
    return Path(file_field).name


def parse_bib_file(bib_path: Path) -> list[BibEntry]:
    if not bib_path.exists():
        raise FileNotFoundError(f"Bibliography file not found: {bib_path}")

    with open(bib_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Keep biblatex types (@online, @report): they are entries like any other.
    parser = bibtexparser.bparser.BibTexParser(
        common_strings=True, ignore_nonstandard_types=False
    )
    db = bibtexparser.loads(text, parser)

    entries = []
    for e in db.entries:
        raw_title = e.get("title")
        display_title = _strip_latex(raw_title) if raw_title else None

        year_match = _YEAR_RE.search(e.get("year") or e.get("date") or "")
        year = int(year_match.group(0)) if year_match else None

        entries.append(BibEntry(
            citekey=e.get("ID", ""),
            entry_type=e.get("ENTRYTYPE", ""),
            title=display_title,
            authors=e.get("author"),
            year=year,
            doi=e.get("doi"),
            file=e.get("file"),
            abstract=clean_bib_text(e["abstract"]) if e.get("abstract") else None,
        ))
    return entries


def match_by_doi(
    papers: list[dict], bib_entries: list[BibEntry]
) -> list[LinkMatch]:
    doi_to_entry = {}
    for entry in bib_entries:
        if entry.doi:
            doi_to_entry[normalize_doi(entry.doi)] = entry

    matches = []
    for paper in papers:
        if not paper.get("doi"):
            continue
        norm = normalize_doi(paper["doi"])
        if norm in doi_to_entry:
            entry = doi_to_entry[norm]
            matches.append(LinkMatch(
                paper_id=paper["id"],
                filename=paper["filename"],
                citekey=entry.citekey,
                source="doi",
                confidence="certain",
                score=1.0,
                paper_title=paper.get("title"),
                bib_title=entry.title,
            ))
    return matches


def match_by_filename(
    papers: list[dict], bib_entries: list[BibEntry]
) -> list[LinkMatch]:
    fname_to_entry = {}
    for entry in bib_entries:
        bib_fname = extract_bib_filename(entry.file)
        if bib_fname:
            fname_to_entry[bib_fname.lower()] = entry

    matches = []
    for paper in papers:
        if paper["filename"].lower() in fname_to_entry:
            entry = fname_to_entry[paper["filename"].lower()]
            matches.append(LinkMatch(
                paper_id=paper["id"],
                filename=paper["filename"],
                citekey=entry.citekey,
                source="filename",
                confidence="likely",
                score=1.0,
                paper_title=paper.get("title"),
                bib_title=entry.title,
            ))
    return matches


def match_by_title(
    papers: list[dict],
    bib_entries: list[BibEntry],
    threshold: float = 0.85,
) -> list[LinkMatch]:
    matches = []
    for paper in papers:
        # The layout title is a fallback so an untitled paper can join at all.
        # It is unverified, so its matches stay "uncertain": they need
        # --include-uncertain, however exact the string match looks.
        hint_only = not paper.get("title")
        title = paper.get("title_from_layout") if hint_only else paper["title"]
        if not title:
            continue
        norm_paper = normalize_title(title)
        best_score = 0.0
        best_entry = None

        for entry in bib_entries:
            if not entry.title:
                continue
            norm_bib = normalize_title(entry.title)
            score = SequenceMatcher(None, norm_paper, norm_bib).ratio()
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < threshold:
            continue

        if best_score >= 0.95:
            confidence = "certain"
        elif best_score >= threshold:
            confidence = "likely"
        else:
            confidence = "uncertain"

        # Year boost: matching year bumps confidence one tier
        paper_year = paper.get("year")
        if paper_year and best_entry.year and paper_year == best_entry.year:
            if confidence == "uncertain":
                confidence = "likely"
            elif confidence == "likely":
                confidence = "certain"
        if hint_only:
            confidence = "uncertain"

        matches.append(LinkMatch(
            paper_id=paper["id"],
            filename=paper["filename"],
            citekey=best_entry.citekey,
            source="title",
            confidence=confidence,
            score=best_score,
            paper_title=title,
            bib_title=best_entry.title,
        ))
    return matches


def reconcile(
    conn: sqlite3.Connection,
    bib_entries: list[BibEntry],
    threshold: float = 0.85,
) -> tuple[list[LinkMatch], LinkStats]:
    papers = [dict(r) for r in conn.execute("SELECT * FROM papers").fetchall()]
    stats = LinkStats(total_papers=len(papers), total_bib_entries=len(bib_entries))

    all_matches: list[LinkMatch] = []
    matched_paper_ids: set[int] = set()
    matched_citekeys: set[str] = set()

    # Phase 1: DOI
    doi_matches = match_by_doi(papers, bib_entries)
    for m in doi_matches:
        if m.paper_id not in matched_paper_ids and m.citekey not in matched_citekeys:
            all_matches.append(m)
            matched_paper_ids.add(m.paper_id)
            matched_citekeys.add(m.citekey)
    stats.matched_doi = len([m for m in all_matches if m.source == "doi"])

    # Phase 2: Filename (exclude already matched)
    remaining_papers = [p for p in papers if p["id"] not in matched_paper_ids]
    remaining_entries = [e for e in bib_entries if e.citekey not in matched_citekeys]
    fname_matches = match_by_filename(remaining_papers, remaining_entries)
    for m in fname_matches:
        if m.paper_id not in matched_paper_ids and m.citekey not in matched_citekeys:
            all_matches.append(m)
            matched_paper_ids.add(m.paper_id)
            matched_citekeys.add(m.citekey)
    stats.matched_filename = len([m for m in all_matches if m.source == "filename"])

    # Phase 3: Title (exclude already matched)
    remaining_papers = [p for p in papers if p["id"] not in matched_paper_ids]
    remaining_entries = [e for e in bib_entries if e.citekey not in matched_citekeys]
    title_matches = match_by_title(remaining_papers, remaining_entries, threshold=threshold)
    for m in title_matches:
        if m.paper_id not in matched_paper_ids and m.citekey not in matched_citekeys:
            all_matches.append(m)
            matched_paper_ids.add(m.paper_id)
            matched_citekeys.add(m.citekey)

    title_certain = [
        m for m in all_matches
        if m.source == "title" and m.confidence in ("certain", "likely")
    ]
    title_uncertain = [
        m for m in all_matches
        if m.source == "title" and m.confidence == "uncertain"
    ]
    stats.matched_title_certain = len(title_certain)
    stats.matched_title_uncertain = len(title_uncertain)

    stats.unmatched_papers = len(papers) - len(matched_paper_ids)
    stats.unmatched_bib = len(bib_entries) - len(matched_citekeys)

    return all_matches, stats


def apply_matches(
    conn: sqlite3.Connection,
    matches: list[LinkMatch],
    include_uncertain: bool = False,
    force: bool = False,
) -> int:
    applied = 0
    for m in matches:
        if m.confidence == "uncertain" and not include_uncertain:
            continue

        if not force:
            row = conn.execute(
                "SELECT citekey FROM papers WHERE id = ?", (m.paper_id,)
            ).fetchone()
            if row and row["citekey"] is not None:
                continue

        conn.execute(
            "UPDATE papers SET citekey = ?, citekey_source = ? WHERE id = ?",
            (m.citekey, m.source, m.paper_id),
        )
        applied += 1

    conn.commit()
    return applied


# --- Bibliography generation ---


def generate_citekey(
    authors: str | None,
    year: int | None,
    title: str | None,
) -> str:
    parts = []

    if authors:
        first_author = authors.split(" and ")[0].strip()
        if "," in first_author:
            last_name = first_author.split(",")[0].strip()
        else:
            last_name = first_author.split()[-1] if first_author.split() else ""
        clean = _CITEKEY_CLEAN_RE.sub("", last_name.lower())
        if clean:
            parts.append(clean)

    if year is not None:
        parts.append(str(year))

    if title:
        stop_words = {"a", "an", "the", "of", "in", "on", "for", "and", "to"}
        words = re.findall(r"[a-z]+", title.lower())
        first_word = next(
            (w for w in words if w not in stop_words), None
        )
        if first_word:
            parts.append(first_word)

    return "".join(parts) if parts else "unknown"


_BIB_ESCAPES = {
    "\\": r"\textbackslash ",
    "{": r"\{",
    "}": r"\}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
}


def _bib_escape(text: str) -> str:
    """Escape LaTeX-special characters in a BibTeX field value."""
    return "".join(_BIB_ESCAPES.get(ch, ch) for ch in text)


def assign_citekeys(
    papers: list[dict],
    reserved_keys: set[str] | None = None,
) -> list[tuple[dict, str]]:
    """Pair each paper with a unique citekey, avoiding reserved + in-batch keys."""
    taken = {k.lower() for k in (reserved_keys or set())}
    assigned: list[tuple[dict, str]] = []

    for paper in papers:
        base = generate_citekey(
            paper.get("authors"), paper.get("year"), paper.get("title")
        )
        key = base
        suffix = 0
        while key.lower() in taken:
            key = f"{base}{chr(ord('a') + suffix)}"
            suffix += 1
        taken.add(key.lower())
        assigned.append((paper, key))

    return assigned


def _format_bib_entry(
    citekey: str,
    paper: dict,
    file_prefix: str | None = None,
) -> str:
    entry_type = "article" if paper.get("doi") else "misc"
    lines = [f"@{entry_type}{{{citekey},"]

    if paper.get("title"):
        lines.append(f"  title = {{{_bib_escape(str(paper['title']))}}},")
    if paper.get("authors"):
        lines.append(f"  author = {{{_bib_escape(str(paper['authors']))}}},")
    if paper.get("year"):
        lines.append(f"  year = {{{paper['year']}}},")
    if paper.get("doi"):
        lines.append(f"  doi = {{{_bib_escape(str(paper['doi']))}}},")

    filename = paper["filename"]
    file_value = (
        f"{file_prefix.rstrip('/')}/{filename}" if file_prefix else filename
    )
    lines.append(f"  file = {{{file_value}}},")
    lines.append("}")
    return "\n".join(lines)


def generate_bib_content(
    papers: list[dict],
    file_prefix: str | None = None,
) -> str:
    entries = [
        _format_bib_entry(key, paper, file_prefix=file_prefix)
        for paper, key in assign_citekeys(papers)
    ]
    return "\n\n".join(entries) + "\n"
