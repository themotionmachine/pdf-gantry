#!/usr/bin/env python
"""Score title extraction against a hand-curated .bib.

    python eval/title_harness.py DB BIB [--papers-dir DIR] [--threshold 0.55] [--misses N]

The eval set is every paper whose citekey joins a bib entry and whose bib title
is present in the document (containment >= 0.6). Scores the linearized-text
extractor and the layout title; with --papers-dir the layout title is
recomputed from the PDFs, otherwise the stored column is read. Opens DB
read-only and never writes it.

These papers are well-formed by selection, so this flatters any extractor: a
blind sample of untitled papers scored far lower. Use it to catch regressions,
not as an accuracy estimate for the corpus.
"""

import argparse
import sqlite3
from pathlib import Path

import pymupdf

from pdf_gantry.layout import extract_layout_title
from pdf_gantry.link import parse_bib_file
from pdf_gantry.metadata import (
    _extract_title_from_text,
    corpus_boilerplate,
    title_containment,
    title_similarity,
)


def load_pairs(conn, bib_path):
    entries = {e.citekey: e for e in parse_bib_file(Path(bib_path))}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)")}
    layout = "p.title_from_layout" if "title_from_layout" in cols else "NULL"
    rows = conn.execute(
        f"SELECT p.id, p.path, p.citekey, {layout} AS layout, pt.raw_text FROM papers p "
        "JOIN paper_text pt ON pt.paper_id = p.id WHERE p.citekey IS NOT NULL"
    ).fetchall()
    return [
        (r, entries[r["citekey"]]) for r in rows
        if r["citekey"] in entries and r["raw_text"] and entries[r["citekey"]].title
        and title_containment(entries[r["citekey"]].title, r["raw_text"]) >= 0.6
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("db")
    ap.add_argument("bib")
    ap.add_argument("--papers-dir", help="Recompute layout titles from these PDFs")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--misses", type=int, default=0, help="Print the first N layout misses")
    args = ap.parse_args()

    pymupdf.TOOLS.mupdf_display_errors(False)
    conn = sqlite3.connect(f"file:{Path(args.db).resolve()}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    boilerplate = corpus_boilerplate(conn)
    pairs = load_pairs(conn, args.bib)

    text_ok = layout_ok = produced = high = high_ok = 0
    misses = []
    for row, entry in pairs:
        guess = _extract_title_from_text(row["raw_text"], boilerplate)
        text_ok += title_similarity(guess, entry.title) >= args.threshold
        confidence = None
        if args.papers_dir:
            found = extract_layout_title(Path(args.papers_dir) / row["path"], boilerplate)
            title, confidence = found.title, found.confidence
        else:
            title = row["layout"]
        ok = title_similarity(title, entry.title) >= args.threshold
        layout_ok += ok
        produced += bool(title)
        high += confidence == "high"
        high_ok += ok and confidence == "high"
        if not ok:
            misses.append((row["id"], title, entry.title))

    n = len(pairs)
    pct = (lambda a, b: f"{a}/{b} = {a / b:.1%}" if b else "n/a")
    print(f"eval pairs: {n} (threshold {args.threshold})")
    print(f"text extractor: {pct(text_ok, n)}")
    print(f"layout title:   {pct(layout_ok, n)} (produced {produced})")
    if args.papers_dir:
        print(f"high confidence: {pct(high_ok, high)}")
    for pid, got, want in misses[:args.misses]:
        print(f"  [{pid}] got {str(got)[:70]!r}\n        bib {want[:70]!r}")


if __name__ == "__main__":
    main()
