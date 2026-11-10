"""Hybrid search's vector half covers papers that only have chunk embeddings.

`pipeline` writes chunk embeddings only, so on a fresh install no paper has a
row in paper_embeddings. Hybrid read only paper_embeddings, so it was silently
keyword-only while still reporting mode "hybrid".
"""

import pytest

from pdf_gantry.search import hybrid_search

from .search_helpers import add_paper, new_db, vec


@pytest.fixture
def db(tmp_path):
    conn, _ = new_db(tmp_path)
    # Doc-level vector only (an older index).
    add_paper(conn, 1, "old.pdf", title="Old paper", text="gardening soil",
              doc_vec=vec(1))
    # Chunk vectors only (what pipeline writes today).
    add_paper(conn, 2, "new.pdf", title="New paper", text="valve seat",
              chunks=["Claims are rejected as obvious.", "Unrelated preamble."],
              chunk_vecs=[vec(20), vec(21)])
    # Both: the doc vector stays the vector signal.
    add_paper(conn, 3, "both.pdf", title="Both", text="lithium anode",
              chunks=["Dendrites grow."], chunk_vecs=[vec(30)], doc_vec=vec(3))
    return conn


def test_chunk_only_paper_found_without_keyword_match(db):
    results = hybrid_search(db, "obviousness", vec(20))
    by_id = {r.id: r for r in results}
    assert 2 in by_id
    assert by_id[2].score_vector == pytest.approx(1.0)
    assert by_id[2].rank_vector == 1
    assert by_id[2].score_fts is None


def test_chunk_only_paper_snippet_is_its_best_chunk(db):
    [top, *_] = hybrid_search(db, "obviousness", vec(20))
    assert top.id == 2
    assert "rejected as obvious" in top.snippet


def test_doc_vector_still_used_when_present(db):
    results = hybrid_search(db, "nothingmatches", vec(3))
    by_id = {r.id: r for r in results}
    assert by_id[3].score_vector == pytest.approx(1.0)
    assert by_id[3].rank_vector == 1


def test_each_paper_appears_once(db):
    results = hybrid_search(db, "nothingmatches", vec(30))
    ids = [r.id for r in results]
    assert len(ids) == len(set(ids))
    assert set(ids) == {1, 2, 3}


def test_restrict_ids_covers_chunk_only_papers(db):
    results = hybrid_search(db, "nothingmatches", vec(21), restrict_ids=[2])
    assert [r.id for r in results] == [2]
    assert results[0].score_vector == pytest.approx(1.0)


def test_restrict_ids_excludes_other_chunk_only_papers(db):
    results = hybrid_search(db, "nothingmatches", vec(20), restrict_ids=[1])
    assert [r.id for r in results] == [1]
