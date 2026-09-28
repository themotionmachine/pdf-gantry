"""Relevant snippets (E10): the best-matching chunk, not the first 200 chars."""

import pytest

from pdf_gantry.search import (
    _trim_snippet,
    fts_search,
    hybrid_search,
    semantic_search,
)

from .search_helpers import add_paper, new_db, vec

MASTHEAD = "JOURNAL OF THINGS Vol.:(0123456789) ISSN 1234-5678 " * 5


@pytest.fixture
def db(tmp_path):
    conn, _ = new_db(tmp_path)
    add_paper(
        conn, 1, "a.pdf", title="Paper A",
        text=MASTHEAD + " intro ... the deepfakes elections section ...",
        chunks=[MASTHEAD, "Background on gardening and soil.",
                "Deepfakes threaten elections because voters cannot verify video."],
        chunk_vecs=[vec(10), vec(11), vec(12)],
        doc_vec=vec(1),
    )
    # No chunks at all: fallback to a window of raw_text around the term.
    add_paper(
        conn, 2, "b.pdf", title="Paper B",
        text=MASTHEAD + ("filler words " * 40) + "a rare zebrafish finding here. " + "tail " * 50,
        doc_vec=vec(2),
    )
    return conn


def test_fts_snippet_is_matching_chunk(db):
    [r] = fts_search(db, "deepfakes elections", restrict_ids=[1])
    assert "Deepfakes threaten elections" in r.snippet
    assert "ISSN" not in r.snippet


def test_fts_snippet_stemmed_term_still_found(db):
    [r] = fts_search(db, "election", restrict_ids=[1])
    assert "Deepfakes threaten elections" in r.snippet


def test_fts_snippet_falls_back_to_raw_text_window(db):
    [r] = fts_search(db, "zebrafish")
    assert "zebrafish" in r.snippet
    assert "ISSN" not in r.snippet
    assert len(r.snippet) <= 320


def test_semantic_snippet_is_best_vector_chunk(db):
    # Query vector points exactly at chunk 1 of paper 1 ("gardening").
    results = semantic_search(db, vec(11), restrict_ids=[1])
    assert "gardening" in results[0].snippet
    results = semantic_search(db, vec(11))
    assert "gardening" in {r.id: r for r in results}[1].snippet


def test_hybrid_snippet_prefers_best_vector_chunk(db):
    results = hybrid_search(db, "deepfakes", vec(11))
    by_id = {r.id: r for r in results}
    assert "gardening" in by_id[1].snippet


def test_hybrid_snippet_uses_fts_chunk_without_chunk_vectors(db):
    results = hybrid_search(db, "zebrafish", vec(2))
    assert "zebrafish" in {r.id: r for r in results}[2].snippet


def test_trim_snippet_centres_on_term_and_marks_truncation():
    text = "x " * 300 + "needle in the haystack " + "y " * 300
    s = _trim_snippet(text, ["needle"], width=120)
    assert "needle" in s
    assert s.startswith("…") and s.endswith("…")
    assert len(s) <= 125


def test_trim_snippet_short_text_unchanged():
    assert _trim_snippet("  short\n text ", ["zzz"]) == "short text"
