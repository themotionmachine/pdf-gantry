"""works.py: fold copies of the same paper into one work (Ramelli PR 1).

A wrong merge hides a real paper, so most of these tests pin what must NOT
fold: a chapter with its book, generic titles, suspect DOIs.
"""

import sqlite3

import pytest

from pdf_gantry import works
from pdf_gantry.db import get_connection
from pdf_gantry.models import SearchResult


def _add(conn, pid, *, file_hash=None, doi=None, title=None, layout=None, year=None,
         pages=20, suspect=0, citekey=None, source=None, has_text=1,
         indexed_at="2026-01-01T00:00:00+00:00", filename=None):
    conn.execute(
        "INSERT INTO papers (id, path, filename, file_hash, file_size, file_modified, "
        "page_count, has_text, title, year, doi, metadata_source, metadata_suspect, "
        "citekey, title_from_layout, indexed_at, updated_at) "
        "VALUES (?, ?, ?, ?, 1, '2026-01-01', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (pid, f"/p/{pid}.pdf", filename or f"{pid}.pdf", file_hash or f"h{pid}", pages,
         has_text, title, year, doi, source, suspect, citekey, layout, indexed_at,
         indexed_at),
    )
    conn.commit()


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "index.db"
    conn = get_connection(str(path))
    works.clear_cache()
    yield conn, path
    conn.close()
    works.clear_cache()


def _ro(path):
    """A plain read-only connection, no row factory: how Ramelli and tests read."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _members(conn, pid):
    return works.works_for(conn, [pid])[pid].member_ids


TITLE = "Free-energy minimization in joint agent-environment systems"


# --- edges -------------------------------------------------------------------

def test_singleton_is_its_own_work(db):
    conn, _ = db
    _add(conn, 5, title="A lonely paper")
    w = works.works_for(conn, [5])[5]
    assert w == works.Work(key="work:5", canonical_id=5, member_ids=(5,))


def test_same_file_hash_folds(db):
    conn, _ = db
    _add(conn, 1, file_hash="same")
    _add(conn, 2, file_hash="same", pages=300)
    assert _members(conn, 1) == (1, 2)


def test_agreeing_doi_and_titles_fold_case_and_prefix_insensitive(db):
    conn, _ = db
    _add(conn, 3, doi="https://doi.org/10.1016/J.JTBI.2018.07.002", title=TITLE, pages=18)
    _add(conn, 7, doi="10.1016/j.jtbi.2018.07.002", title=TITLE + ".", pages=19)
    w = works.works_for(conn, [7])[7]
    assert w.member_ids == (3, 7)
    assert w.key == "work:3"


def test_doi_edge_needs_page_counts_within_two(db):
    """A chapter carrying its book's DOI is not the book."""
    conn, _ = db
    _add(conn, 1, doi="10.4135/9781446249413", title="The SAGE Handbook of Case-Based Methods",
         pages=21)
    _add(conn, 2, doi="10.4135/9781446249413", title="The SAGE Handbook of Case-Based Methods",
         pages=24)
    assert _members(conn, 1) == (1,)


def test_doi_edge_needs_known_page_counts(db):
    conn, _ = db
    _add(conn, 1, doi="10.1/x", title=TITLE, pages=None)
    _add(conn, 2, doi="10.1/x", title=TITLE, pages=18)
    assert _members(conn, 1) == (1,)


def test_doi_edge_needs_agreeing_titles(db):
    conn, _ = db
    _add(conn, 1, doi="10.37016/mr-2020-002", title="Cross-Platform Disinformation Campaigns")
    _add(conn, 2, doi="10.37016/mr-2020-002", title="Disinformation as Collaborative Work")
    assert _members(conn, 1) == (1,)


def test_doi_edge_accepts_one_title_containing_the_other(db):
    conn, _ = db
    _add(conn, 1, doi="10.1177/0275074014533142", title="From Policy “Frames” to “Framing”")
    _add(conn, 2, doi="10.1177/0275074014533142",
         title='From Policy "Frames" to "Framing". Theorizing a More Dynamic, Political '
               "Approach")
    assert _members(conn, 1) == (1, 2)


def test_doi_edge_compares_display_titles(db):
    """No metadata title: the layout title stands in, then the filename."""
    conn, _ = db
    _add(conn, 1, doi="10.1/x", title=TITLE)
    _add(conn, 2, doi="10.1/x", title=None, layout=TITLE)
    _add(conn, 3, doi="10.1/x", title=None, layout=None, filename="Bruineberg et al 2018.pdf")
    assert _members(conn, 1) == (1, 2)


def test_suspect_row_never_takes_a_doi_edge(db):
    conn, _ = db
    _add(conn, 1, doi="10.1/x", title=TITLE)
    _add(conn, 2, doi="10.1/x", title=TITLE, suspect=1)
    assert _members(conn, 1) == (1,)
    assert _members(conn, 2) == (2,)


def test_doi_edge_refused_when_printed_titles_disagree(db):
    """Two chapters both stamped with the book's DOI and title: the titles read
    off each PDF's first page say they are different documents."""
    conn, _ = db
    book = "The SAGE Handbook of Case-Based Methods"
    _add(conn, 1, doi="10.4135/9781446249413", title=book, year=2009,
         layout="Making the Most of an Historical Case Study")
    _add(conn, 2, doi="10.4135/9781446249413", title=book, year=2009,
         layout="Complexity and Case: Network Approaches")
    assert _members(conn, 1) == (1,)


def test_same_title_and_year_fold(db):
    conn, _ = db
    _add(conn, 1, title="Is the Free-Energy Principle a Formal Theory of Semantics?", year=2020)
    _add(conn, 2, title="Is the free energy principle a formal theory of semantics", year=2020)
    assert _members(conn, 2) == (1, 2)


@pytest.mark.parametrize("kw", [
    {"year": 2021},                     # different year
    {"year": None},                     # unknown year
    {"pages": 300},                     # a book with its chapter's title
    {"suspect": 1},                     # the title came from a suspect match
])
def test_title_year_edge_refusals(db, kw):
    conn, _ = db
    title = "Is the Free-Energy Principle a Formal Theory of Semantics"
    _add(conn, 1, title=title, year=2020)
    _add(conn, 2, **{"title": title, "year": 2020, **kw})
    assert _members(conn, 1) == (1,)


def test_title_year_edge_needs_five_words(db):
    conn, _ = db
    _add(conn, 1, title="Disinformation in International Politics", year=2019)
    _add(conn, 2, title="Disinformation in international politics", year=2019)
    assert _members(conn, 1) == (1,)


def test_title_year_edge_ignores_layout_title(db):
    conn, _ = db
    t = "Is the Free-Energy Principle a Formal Theory of Semantics"
    _add(conn, 1, title=None, layout=t, year=2020)
    _add(conn, 2, title=None, layout=t, year=2020)
    assert _members(conn, 1) == (1,)


@pytest.mark.parametrize("title", [
    "Introduction to the Special Issue on Platform Governance",
    "Book review of the new institutionalism reader",
    "Editorial: notes from the editors this year",
    "Reply to my critics on the free energy principle",
    "Preface to the second edition of this volume",
    "Table of contents for volume twelve issue three",
])
def test_generic_titles_never_fold_on_title(db, title):
    conn, _ = db
    _add(conn, 1, title=title, year=2020)
    _add(conn, 2, title=title, year=2020)
    assert _members(conn, 1) == (1,)


def test_union_is_transitive_and_key_is_smallest_member(db):
    conn, _ = db
    _add(conn, 9, file_hash="hh", doi="10.1/x", title=TITLE)
    _add(conn, 4, file_hash="hh")
    _add(conn, 6, doi="10.1/X", title=TITLE)
    ws = works.works_for(conn, [9, 6, 4])
    assert {w.key for w in ws.values()} == {"work:4"}
    assert ws[6].member_ids == (4, 6, 9)


def test_unknown_ids_are_left_out(db):
    conn, _ = db
    _add(conn, 1)
    assert set(works.works_for(conn, [1, 999])) == {1}


# --- canonical copy ------------------------------------------------------------

def _canonical(conn, *rows):
    for pid, kw in rows:
        _add(conn, pid, file_hash="dup", **kw)
    return works.works_for(conn, [rows[0][0]])[rows[0][0]].canonical_id


def test_canonical_prefers_a_citekey(db):
    conn, _ = db
    assert _canonical(conn, (1, {"source": "manual:me"}), (2, {"citekey": "k2020"})) == 2


@pytest.mark.parametrize("winner,loser", [
    ("manual:steward", "bib:prime"),
    ("brev-manual-from-text", "bib:prime"),
    ("bib:prime", "openalex"),
    ("openalex_title", "none:openalex"),
    ("openalex", None),
])
def test_canonical_source_order(db, winner, loser):
    conn, _ = db
    assert _canonical(conn, (1, {"source": loser}), (2, {"source": winner})) == 2


def test_canonical_then_not_suspect_then_text_then_newest(db):
    conn, _ = db
    assert _canonical(conn, (1, {"suspect": 1}), (2, {})) == 2
    conn.execute("DELETE FROM papers")
    conn.commit()
    assert _canonical(conn, (1, {"has_text": 0}), (2, {})) == 2
    conn.execute("DELETE FROM papers")
    conn.commit()
    assert _canonical(conn, (1, {"indexed_at": "2026-05-01T00:00:00+00:00"}),
                      (2, {"indexed_at": "2026-02-01T00:00:00+00:00"})) == 1


# --- collapse --------------------------------------------------------------------

def _r(pid, score):
    return SearchResult(id=pid, filename=f"{pid}.pdf", path=f"/p/{pid}.pdf", score=score)


def test_search_result_work_fields_default_empty():
    r = _r(1, 1.0)
    assert r.work_key is None and r.canonical_id is None and r.copies == []


def test_collapse_keeps_best_ranked_copy_with_its_own_id(db):
    conn, _ = db
    _add(conn, 1, file_hash="dup", citekey="canon")
    _add(conn, 2, file_hash="dup")
    _add(conn, 3, file_hash="dup")
    _add(conn, 4)
    out = works.collapse(conn, [_r(3, 0.9), _r(4, 0.8), _r(1, 0.7)], limit=10)
    assert [r.id for r in out] == [3, 4]
    top = out[0]
    assert (top.work_key, top.canonical_id, top.copies) == ("work:1", 1, [1, 2])
    assert (out[1].work_key, out[1].canonical_id, out[1].copies) == ("work:4", 4, [])


def test_collapse_truncates_after_folding(db):
    conn, _ = db
    _add(conn, 1, file_hash="dup")
    _add(conn, 2, file_hash="dup")
    _add(conn, 3)
    _add(conn, 4)
    out = works.collapse(conn, [_r(1, 4), _r(2, 3), _r(3, 2), _r(4, 1)], limit=2)
    assert [r.id for r in out] == [1, 3]


def test_collapse_does_not_mutate_its_input(db):
    conn, _ = db
    _add(conn, 1)
    given = [_r(1, 1.0)]
    works.collapse(conn, given, limit=5)
    assert given[0].work_key is None


# --- read-only, cache -------------------------------------------------------------

def test_reads_never_write(db):
    conn, path = db
    _add(conn, 1, file_hash="dup")
    _add(conn, 2, file_hash="dup")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    files = [p for p in (path, path.with_name(path.name + "-wal")) if p.exists()]
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    ro = _ro(path)
    assert ro.row_factory is None
    ws = works.works_for(ro, [1, 2])
    works.collapse(ro, [_r(2, 1.0)], limit=5)
    assert ws[1].member_ids == (1, 2)
    assert ro.total_changes == 0
    ro.close()
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    assert after == before


def test_cache_hits_on_unchanged_data(db, monkeypatch):
    conn, path = db
    _add(conn, 1)
    calls = []
    real = works._load_rows
    monkeypatch.setattr(works, "_load_rows", lambda c: calls.append(1) or real(c))
    works.works_for(conn, [1])
    works.works_for(conn, [1])
    ro = _ro(path)
    works.works_for(ro, [1])  # another connection, same unchanged file
    ro.close()
    assert len(calls) == 1


def test_cache_sees_own_writes(db):
    conn, _ = db
    _add(conn, 1, file_hash="a")
    _add(conn, 2, file_hash="b")
    assert _members(conn, 1) == (1,)
    conn.execute("UPDATE papers SET file_hash = 'a' WHERE id = 2")
    conn.commit()
    assert _members(conn, 1) == (1, 2)


def test_cache_sees_other_connections_writes(db):
    conn, path = db
    _add(conn, 1, file_hash="a")
    _add(conn, 2, file_hash="b")
    ro = _ro(path)
    assert _members(ro, 1) == (1,)
    conn.execute("UPDATE papers SET file_hash = 'a' WHERE id = 2")
    conn.commit()
    assert _members(ro, 1) == (1, 2)  # same reader connection
    fresh = _ro(path)
    assert _members(fresh, 1) == (1, 2)  # a new connection
    ro.close()
    fresh.close()


def test_cache_is_per_database(tmp_path):
    works.clear_cache()
    a = get_connection(str(tmp_path / "a.db"))
    b = get_connection(str(tmp_path / "b.db"))
    _add(a, 1, file_hash="x")
    _add(a, 2, file_hash="x")
    _add(b, 1, file_hash="x")
    _add(b, 2, file_hash="y")
    assert _members(a, 1) == (1, 2)
    assert _members(b, 1) == (1,)
    a.close()
    b.close()


def test_in_memory_database_works():
    works.clear_cache()
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE papers (id INTEGER PRIMARY KEY, filename TEXT, file_hash TEXT, "
        "doi TEXT, title TEXT, title_from_layout TEXT, year INTEGER, page_count INTEGER, "
        "metadata_suspect INTEGER, citekey TEXT, metadata_source TEXT, has_text INTEGER, "
        "indexed_at TEXT)"
    )
    conn.executemany(
        "INSERT INTO papers VALUES (?, 'f.pdf', ?, NULL, NULL, NULL, NULL, 1, 0, NULL, "
        "NULL, 1, '2026')", [(1, "x"), (2, "x")])
    assert _members(conn, 2) == (1, 2)
    conn.close()
