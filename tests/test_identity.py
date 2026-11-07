"""identity.py: matching a stored file_hash + path key back onto `papers`.

collections.py and reading.py both key their rows this way; the matcher's
behaviour is pinned through collections' own tests too.
"""

import sqlite3

import pytest

from pdf_gantry import identity
from pdf_gantry.db import get_connection


@pytest.fixture
def index(tmp_path):
    conn = get_connection(str(tmp_path / "index.db"))
    yield conn
    conn.close()


def add_paper(index, path, file_hash, title=None):
    cur = index.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified,"
        " indexed_at, updated_at, title) VALUES (?, ?, ?, 1, '2024', '2024', '2024', ?)",
        (path, path.rsplit("/", 1)[-1], file_hash, title),
    )
    index.commit()
    return cur.lastrowid


def test_public_names():
    for name in ("Paper", "Match", "Index", "load_index", "decide", "match"):
        assert hasattr(identity, name)


def test_exact_key_resolves(index):
    a = add_paper(index, "a.pdf", "H")
    add_paper(index, "b.pdf", "H")
    found = identity.match(identity.load_index(index), "H", "a.pdf")
    assert found.paper.id == a and not found.ambiguous


def test_hash_only_then_path_only(index):
    a = add_paper(index, "renamed.pdf", "H")
    b = add_paper(index, "x.pdf", "OTHER")
    idx = identity.load_index(index)
    assert identity.match(idx, "H", "gone.pdf").paper.id == a
    assert identity.match(idx, "CHANGED", "x.pdf").paper.id == b
    assert identity.match(idx, "NONE", "none.pdf").paper is None


def test_ties_are_ambiguous(index):
    a = add_paper(index, "a.pdf", "H")
    b = add_paper(index, "b.pdf", "H")
    found = identity.match(identity.load_index(index), "H", "c.pdf")
    assert found.ambiguous and set(found.candidates) == {a, b}


def test_liveness_breaks_ties_and_without_liveness_ignores_it(index, tmp_path):
    pdir = tmp_path / "papers"
    pdir.mkdir()
    add_paper(index, "a.pdf", "H")
    b = add_paper(index, "b.pdf", "H")
    (pdir / "b.pdf").write_bytes(b"%PDF")
    idx = identity.load_index(index, pdir)
    assert idx.is_live(idx.by_id[b])
    assert identity.match(idx, "H", "c.pdf").paper.id == b
    assert identity.match(idx.without_liveness(), "H", "c.pdf").ambiguous


def test_works_on_a_plain_read_only_connection(index, tmp_path):
    a = add_paper(index, "a.pdf", "H", title="T")
    ro = sqlite3.connect(f"{(tmp_path / 'index.db').resolve().as_uri()}?mode=ro", uri=True)
    try:
        idx = identity.load_index(ro)
    finally:
        ro.close()
    assert idx.by_id[a].title == "T"
