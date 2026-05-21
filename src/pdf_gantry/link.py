"""BibTeX reconciliation — match PDFs to bibliography entries."""

import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import bibtexparser


@dataclass
class BibEntry:
    citekey: str
    entry_type: str
    title: str | None
    authors: str | None
    year: int | None
    doi: str | None
    file: str | None


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
_LATEX_ACCENT_RE = re.compile(r"""\\["'^`~=.uvHtcdbkr]\s*""")
_LATEX_SYMBOL_RE = re.compile(r"\\&")
_BRACES_RE = re.compile(r"[{}]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_DOI_PREFIX_RE = re.compile(r"^https?://(doi\.org|dx\.doi\.org)/", re.IGNORECASE)


def _strip_latex(text: str) -> str:
    text = _LATEX_SYMBOL_RE.sub("&", text)
    text = _LATEX_ACCENT_RE.sub("", text)
    text = _LATEX_CMD_RE.sub("", text)
    text = _BRACES_RE.sub("", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


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

    with open(bib_path) as f:
        text = f.read()

    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    db = bibtexparser.loads(text, parser)

    entries = []
    for e in db.entries:
        raw_title = e.get("title")
        display_title = _strip_latex(raw_title) if raw_title else None

        year_str = e.get("year", "")
        try:
            year = int(year_str)
        except (ValueError, TypeError):
            year = None

        entries.append(BibEntry(
            citekey=e.get("ID", ""),
            entry_type=e.get("ENTRYTYPE", ""),
            title=display_title,
            authors=e.get("author"),
            year=year,
            doi=e.get("doi"),
            file=e.get("file"),
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
        if not paper.get("title"):
            continue
        norm_paper = normalize_title(paper["title"])
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

        matches.append(LinkMatch(
            paper_id=paper["id"],
            filename=paper["filename"],
            citekey=best_entry.citekey,
            source="title",
            confidence=confidence,
            score=best_score,
            paper_title=paper.get("title"),
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
