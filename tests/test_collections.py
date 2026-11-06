"""collections.py: hand-curated paper sets in a sidecar database.

papers.id is not stable, so memberships are keyed on file_hash + path and
matched back onto `papers` at read time. These tests pin that matching, the
sidecar's isolation from index.db, and the undo rules.
"""

import os
import sqlite3
import stat

import pytest

from pdf_gantry import collections as coll
from pdf_gantry.config import Config
from pdf_gantry.db import get_connection


def add_paper(index, path, file_hash, title=None, citekey=None):
    cur = index.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified,"
        " indexed_at, updated_at, title, citekey)"
        " VALUES (?, ?, ?, 1, '2024', '2024', '2024', ?, ?)",
        (path, path.rsplit("/", 1)[-1], file_hash, title, citekey),
    )
    index.commit()
    return cur.lastrowid


def touch(papers_dir, *names):
    for name in names:
        (papers_dir / name).write_bytes(b"%PDF")


@pytest.fixture
def cfg(tmp_path):
    return Config(index_dir=tmp_path / "gantry")


@pytest.fixture
def index(cfg):
    cfg.index_dir.mkdir()
    conn = get_connection(str(cfg.db_path))
    yield conn
    conn.close()


@pytest.fixture
def conn(cfg, index):
    c = coll.connect(cfg)
    yield c
    c.close()


@pytest.fixture
def pdir(tmp_path):
    d = tmp_path / "papers"
    d.mkdir()
    return d


# --- storage ---


def test_connect_creates_private_sidecar_in_wal_mode(cfg):
    c = coll.connect(cfg)
    path = cfg.index_dir / "collections.db"
    assert path.exists()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert c.execute("PRAGMA user_version").fetchone()[0] == coll.SCHEMA_VERSION
    c.close()


def test_connect_does_not_touch_index_db(cfg):
    cfg.index_dir.mkdir()
    coll.connect(cfg).close()
    assert not cfg.db_path.exists()


def test_connect_is_repeatable_without_writing(cfg):
    coll.connect(cfg).close()
    c = coll.connect(cfg)
    assert c.total_changes == 0
    c.close()


def test_connect_refuses_a_newer_schema(cfg):
    c = coll.connect(cfg)
    c.execute(f"PRAGMA user_version = {coll.SCHEMA_VERSION + 1}")
    c.close()
    with pytest.raises(coll.CollectionError):
        coll.connect(cfg)


def test_connect_readonly_never_creates_the_file(cfg):
    cfg.index_dir.mkdir()
    assert coll.connect_readonly(cfg) is None
    assert not (cfg.index_dir / "collections.db").exists()


def test_connect_readonly_reads_and_cannot_write(cfg):
    w = coll.connect(cfg)
    coll.create(w, "reading", by="ryan")
    w.close()
    r = coll.connect_readonly(cfg)
    assert [c["name"] for c in coll.list_collections(r)] == ["reading"]
    with pytest.raises(sqlite3.OperationalError):
        coll.create(r, "other", by="ryan")
    r.close()


# --- matching ---


def members(conn, index, name, papers_dir=None):
    return coll.resolve(conn, index, name, papers_dir=papers_dir)


def seeded(conn, index, *ids):
    coll.create(conn, "c", by="t")
    return coll.add(conn, index, "c", list(ids), "t")


def test_resolve_returns_members_with_provenance(conn, index):
    pid = add_paper(index, "a.pdf", "H", title="A", citekey="a2020")
    out = coll.create(conn, "c", by="ryan")
    assert out["name"] == "c"
    added = coll.add(conn, index, "c", [pid], "agent", why="relevant")
    r = members(conn, index, "c")
    assert r["ids"] == [pid]
    assert r["missing"] == [] and r["ambiguous"] == []
    m = r["members"][0]
    assert (m["id"], m["filename"], m["title"], m["citekey"]) == (pid, "a.pdf", "A", "a2020")
    assert m["added_by"] == "agent"
    assert m["why"] == "relevant"
    assert m["batch_id"] == added["batch_id"]


def test_rename_on_disk_follows_the_hash_once_the_old_row_is_pruned(conn, index):
    pid = add_paper(index, "a.pdf", "H")
    seeded(conn, index, pid)
    index.execute("DELETE FROM papers WHERE id = ?", (pid,))
    new = add_paper(index, "b.pdf", "H")
    assert members(conn, index, "c")["ids"] == [new]


def test_annotation_at_the_same_path_follows_the_path(conn, index):
    pid = add_paper(index, "a.pdf", "H")
    seeded(conn, index, pid)
    index.execute("UPDATE papers SET file_hash = 'H2' WHERE id = ?", (pid,))
    assert members(conn, index, "c")["ids"] == [pid]


def test_exact_key_beats_a_duplicate_hash(conn, index):
    """Two live rows can share a hash: each membership keeps its own row."""
    a = add_paper(index, "a.pdf", "H")
    b = add_paper(index, "b.pdf", "H")
    seeded(conn, index, a, b)
    r = members(conn, index, "c")
    assert sorted(r["ids"]) == sorted([a, b])
    assert r["ambiguous"] == []


def test_stale_row_yields_to_the_live_row(conn, index, pdir):
    """A renamed file leaves its old row behind until a manual prune."""
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    new = add_paper(index, "b.pdf", "H")
    touch(pdir, "b.pdf")
    assert members(conn, index, "c")["ids"] == [old]  # no liveness: exact key
    assert members(conn, index, "c", pdir)["ids"] == [new]


def test_live_path_candidate_beats_a_dead_hash_candidate(conn, index, pdir):
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    index.execute("UPDATE papers SET path = 'gone.pdf' WHERE id = ?", (a,))
    live = add_paper(index, "a.pdf", "OTHER")
    touch(pdir, "a.pdf")
    assert members(conn, index, "c", pdir)["ids"] == [live]


def test_all_candidates_dead_still_resolves_the_row(conn, index, pdir):
    """An evicted or moved file is not a lost paper: its row still serves reads."""
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    r = members(conn, index, "c", pdir)
    assert r["ids"] == [a] and r["missing"] == []


def test_absent_papers_dir_skips_the_liveness_check(conn, index, tmp_path):
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    add_paper(index, "b.pdf", "H")
    assert members(conn, index, "c", tmp_path / "nope")["ids"] == [old]


def test_equally_good_candidates_are_ambiguous_not_guessed(conn, index):
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    index.execute("DELETE FROM papers WHERE id = ?", (a,))
    b = add_paper(index, "b.pdf", "H")
    c = add_paper(index, "c.pdf", "H")
    r = members(conn, index, "c")
    assert r["ids"] == []
    assert r["ambiguous"][0]["candidates"] == [b, c]
    assert r["ambiguous"][0]["path"] == "a.pdf"


def test_live_candidate_breaks_the_tie(conn, index, pdir):
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    index.execute("DELETE FROM papers WHERE id = ?", (a,))
    add_paper(index, "b.pdf", "H")
    c = add_paper(index, "c.pdf", "H")
    touch(pdir, "c.pdf")
    assert members(conn, index, "c", pdir)["ids"] == [c]


def test_no_candidate_is_missing_and_keeps_its_snapshot(conn, index):
    a = add_paper(index, "a.pdf", "H", title="Lost", citekey="lost2020")
    seeded(conn, index, a)
    index.execute("DELETE FROM papers WHERE id = ?", (a,))
    r = members(conn, index, "c")
    assert r["ids"] == []
    m = r["missing"][0]
    assert (m["filename"], m["title"], m["citekey"]) == ("a.pdf", "Lost", "lost2020")
    assert m["file_hash"] == "H"
    # Never deleted: the paper returns, the member returns.
    index.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified,"
        " indexed_at, updated_at) VALUES ('z.pdf', 'z.pdf', 'H', 1, '', '', '')"
    )
    assert len(members(conn, index, "c")["ids"]) == 1


def test_two_memberships_on_one_row_resolve_to_one_member(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    new = add_paper(index, "b.pdf", "H")
    coll.add(conn, index, "c", [new], "t")
    touch(pdir, "b.pdf")
    r = members(conn, index, "c", pdir)
    assert r["ids"] == [new] and len(r["members"]) == 1


def test_resolve_works_on_a_plain_read_only_index_connection(cfg, conn, index):
    """Ramelli opens index.db mode=ro with no sqlite-vec and no row factory."""
    a = add_paper(index, "a.pdf", "H", title="A")
    ro = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    coll.create(conn, "c", by="t")
    assert coll.add(conn, ro, "c", [a], "t")["added"] == [a]
    assert coll.resolve(conn, ro, "c")["ids"] == [a]
    assert coll.collections_for(conn, ro, a)[0]["name"] == "c"
    assert coll.refresh(conn, ro, None)["updated"] == 0
    assert coll.remove(conn, "c", [a], "t", ro)["removed"] == [a]
    ro.close()


def test_collections_for_uses_the_same_matching(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    other = add_paper(index, "other.pdf", "O")
    seeded(conn, index, old)
    assert [c["name"] for c in coll.collections_for(conn, index, old)] == ["c"]
    assert coll.collections_for(conn, index, other) == []
    assert coll.collections_for(conn, index, 9999) == []
    new = add_paper(index, "b.pdf", "H")
    touch(pdir, "b.pdf")
    assert coll.collections_for(conn, index, new) == []
    assert coll.collections_for(conn, index, new, papers_dir=pdir)[0]["name"] == "c"


def test_collections_for_ignores_deleted_and_removed(conn, index):
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    coll.create(conn, "d", by="t")
    coll.add(conn, index, "d", [a], "t")
    coll.delete(conn, "d", "t")
    assert [c["name"] for c in coll.collections_for(conn, index, a)] == ["c"]
    coll.remove(conn, "c", [a], "t", index)
    assert coll.collections_for(conn, index, a) == []


# --- reads never write ---


def test_reads_never_write(cfg, conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    add_paper(index, "b.pdf", "H")
    touch(pdir, "b.pdf")
    before = conn.total_changes
    coll.resolve(conn, index, "c", papers_dir=pdir)
    coll.collections_for(conn, index, old, papers_dir=pdir)
    coll.list_collections(conn, index)
    coll.history(conn)
    coll.history(conn, "c")
    assert conn.total_changes == before
    row = conn.execute("SELECT path FROM memberships").fetchone()
    assert row["path"] == "a.pdf"


# --- refresh ---


def test_refresh_persists_drifted_keys(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    new = add_paper(index, "b.pdf", "H", title="New")
    touch(pdir, "b.pdf")
    out = coll.refresh(conn, index, pdir)
    assert out["updated"] == 1
    row = conn.execute("SELECT * FROM memberships").fetchone()
    assert (row["path"], row["file_hash"], row["filename"]) == ("b.pdf", "H", "b.pdf")
    index.execute("DELETE FROM papers WHERE id = ?", (old,))
    assert members(conn, index, "c")["ids"] == [new]


def test_refresh_persists_a_changed_hash(conn, index):
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    index.execute("UPDATE papers SET file_hash = 'H2' WHERE id = ?", (a,))
    assert coll.refresh(conn, index, None)["updated"] == 1
    assert conn.execute("SELECT file_hash FROM memberships").fetchone()[0] == "H2"


def test_refresh_with_nothing_drifted_writes_nothing(conn, index):
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    before = conn.total_changes
    out = coll.refresh(conn, index, None)
    assert out["updated"] == 0 and out["checked"] == 1
    assert conn.total_changes == before


def test_refresh_reports_and_leaves_missing_and_ambiguous(conn, index):
    a = add_paper(index, "a.pdf", "H")
    b = add_paper(index, "b.pdf", "G")
    seeded(conn, index, a, b)
    index.execute("DELETE FROM papers")
    add_paper(index, "x.pdf", "G")
    add_paper(index, "y.pdf", "G")
    out = coll.refresh(conn, index, None)
    assert (out["missing"], out["ambiguous"], out["updated"]) == (1, 1, 0)
    assert conn.execute(
        "SELECT COUNT(*) FROM memberships WHERE removed_at IS NULL"
    ).fetchone()[0] == 2


def test_refresh_survives_a_key_collision(conn, index, pdir):
    """Drifting onto a key the collection already holds must not violate the
    live-membership unique index: leave it, count it, resolve still dedupes."""
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    new = add_paper(index, "b.pdf", "H")
    coll.add(conn, index, "c", [new], "t")
    touch(pdir, "b.pdf")
    out = coll.refresh(conn, index, pdir)
    assert out["collided"] == 1 and out["updated"] == 0
    assert members(conn, index, "c", pdir)["ids"] == [new]


def test_refresh_is_recorded_only_when_it_changes_something(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    coll.refresh(conn, index, pdir)
    assert "refresh" not in [h["op"] for h in coll.history(conn)]
    add_paper(index, "b.pdf", "H")
    touch(pdir, "b.pdf")
    batch = coll.refresh(conn, index, pdir)["batch_id"]
    assert coll.history(conn)[0]["batch_id"] == batch
    with pytest.raises(coll.CollectionError):
        coll.undo(conn, batch, "t")


# --- write rules ---


def test_every_write_gets_a_batch_and_a_history_row(conn, index):
    a = add_paper(index, "a.pdf", "H")
    ids = [
        coll.create(conn, "c", by="u1")["batch_id"],
        coll.add(conn, index, "c", [a], "u2", why="because")["batch_id"],
        coll.remove(conn, "c", [a], "u3", index)["batch_id"],
        coll.rename(conn, "c", "d", "u4")["batch_id"],
        coll.delete(conn, "d", "u5")["batch_id"],
    ]
    rows = coll.history(conn)
    assert [r["batch_id"] for r in rows] == ids[::-1]
    assert [r["op"] for r in rows] == ["delete", "rename", "remove", "add", "create"]
    assert [r["by"] for r in rows] == ["u5", "u4", "u3", "u2", "u1"]
    add_row = rows[3]
    assert add_row["why"] == "because" and add_row["count"] == 1
    assert add_row["undone"] is False and add_row["at"]


def test_writes_require_by(conn, index):
    with pytest.raises(coll.CollectionError):
        coll.create(conn, "c")
    with pytest.raises(coll.CollectionError):
        coll.create(conn, "c", by="  ")


def test_add_is_idempotent_and_reports_each_bucket(conn, index):
    a = add_paper(index, "a.pdf", "H")
    b = add_paper(index, "b.pdf", "I")
    coll.create(conn, "c", by="t")
    assert coll.add(conn, index, "c", [a, a], "t")["added"] == [a]
    out = coll.add(conn, index, "c", [a, b, 999], "t")
    assert out["added"] == [b]
    assert out["already_present"] == [a]
    assert out["not_found"] == [999]
    assert coll.resolve(conn, index, "c")["ids"] == [a, b]


def test_add_sees_a_paper_held_under_a_drifted_key_and_refreshes_it(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    new = add_paper(index, "b.pdf", "H")
    touch(pdir, "b.pdf")
    out = coll.add(conn, index, "c", [new], "t", papers_dir=pdir)
    assert out["already_present"] == [new]
    row = conn.execute("SELECT path FROM memberships").fetchone()
    assert row["path"] == "b.pdf"


def test_add_to_unknown_collection_fails(conn, index):
    with pytest.raises(coll.UnknownCollection):
        coll.add(conn, index, "nope", [1], "t")
    assert coll.history(conn) == []


def test_remove_tombstones_and_keeps_the_row(conn, index):
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    out = coll.remove(conn, "c", [a, 999], "ryan", index)
    assert out["removed"] == [a] and out["not_found"] == [999]
    row = conn.execute("SELECT removed_at, removed_by FROM memberships").fetchone()
    assert row["removed_at"] and row["removed_by"] == "ryan"
    assert coll.resolve(conn, index, "c")["ids"] == []


def test_remove_goes_through_matching(conn, index, pdir):
    """The id `show` printed for a renamed file must be removable."""
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    new = add_paper(index, "b.pdf", "H")
    touch(pdir, "b.pdf")
    assert coll.resolve(conn, index, "c", pdir)["ids"] == [new]
    assert coll.remove(conn, "c", [new], "t", index, papers_dir=pdir)["removed"] == [new]
    assert coll.resolve(conn, index, "c", pdir)["ids"] == []


def test_remove_accepts_the_stale_id_the_member_was_added_under(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    add_paper(index, "b.pdf", "H")
    touch(pdir, "b.pdf")
    assert coll.remove(conn, "c", [old], "t", index, papers_dir=pdir)["removed"] == [old]


def test_remove_does_not_guess_at_an_ambiguous_membership(conn, index):
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    index.execute("DELETE FROM papers WHERE id = ?", (a,))
    b = add_paper(index, "b.pdf", "H")
    add_paper(index, "c.pdf", "H")
    out = coll.remove(conn, "c", [b], "t", index)
    assert out["removed"] == [] and out["ambiguous"] == [b]
    assert len(coll.resolve(conn, index, "c")["ambiguous"]) == 1


def test_remove_takes_every_membership_on_that_paper(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    seeded(conn, index, old)
    new = add_paper(index, "b.pdf", "H")
    coll.add(conn, index, "c", [new], "t")
    touch(pdir, "b.pdf")
    coll.remove(conn, "c", [new], "t", index, papers_dir=pdir)
    assert coll.resolve(conn, index, "c", pdir)["ids"] == []


def test_names_are_unique_case_insensitively_among_live_only(conn):
    coll.create(conn, "Reading", by="t")
    with pytest.raises(coll.NameConflict):
        coll.create(conn, "rEADING", by="t")
    coll.delete(conn, "reading", "t")
    coll.create(conn, "READING", by="t")
    assert [c["name"] for c in coll.list_collections(conn)] == ["READING"]


def test_name_uniqueness_folds_beyond_ascii(conn):
    coll.create(conn, "Étude", by="t")
    with pytest.raises(coll.NameConflict):
        coll.create(conn, "éTUDE", by="t")
    with pytest.raises(coll.NameConflict):
        coll.create(conn, "Étude", by="t")  # decomposed


def test_blank_names_are_refused(conn):
    with pytest.raises(coll.CollectionError):
        coll.create(conn, "   ", by="t")


def test_lookups_ignore_case(conn, index):
    coll.create(conn, "Reading", by="t")
    assert coll.resolve(conn, index, "READING")["ids"] == []


def test_rename_keeps_the_id_and_members(conn, index):
    a = add_paper(index, "a.pdf", "H")
    made = coll.create(conn, "c", by="t")
    coll.add(conn, index, "c", [a], "t")
    out = coll.rename(conn, "c", "d", "t")
    assert out["id"] == made["id"] and out["name"] == "d" and out["old_name"] == "c"
    assert coll.resolve(conn, index, "d")["ids"] == [a]
    with pytest.raises(coll.UnknownCollection):
        coll.resolve(conn, index, "c")


def test_rename_refuses_a_taken_name_but_allows_a_case_change(conn):
    coll.create(conn, "a", by="t")
    coll.create(conn, "b", by="t")
    with pytest.raises(coll.NameConflict):
        coll.rename(conn, "a", "B", "t")
    assert coll.rename(conn, "a", "A", "t")["name"] == "A"


def test_list_counts(conn, index):
    a = add_paper(index, "a.pdf", "H")
    b = add_paper(index, "b.pdf", "I")
    seeded(conn, index, a, b)
    index.execute("DELETE FROM papers WHERE id = ?", (b,))
    assert coll.list_collections(conn)[0]["count"] == 2
    entry = coll.list_collections(conn, index)[0]
    assert (entry["count"], entry["resolved"], entry["missing"]) == (2, 1, 1)


def test_history_filters_by_name_and_limits(conn):
    coll.create(conn, "a", by="t")
    coll.create(conn, "b", by="t")
    coll.rename(conn, "b", "b2", "t")
    assert [h["op"] for h in coll.history(conn, "a")] == ["create"]
    assert [h["op"] for h in coll.history(conn, "b2")] == ["rename", "create"]
    assert len(coll.history(conn, limit=1)) == 1
    with pytest.raises(coll.UnknownCollection):
        coll.history(conn, "nope")


def test_history_reaches_a_deleted_collection(conn):
    coll.create(conn, "a", by="t")
    coll.delete(conn, "a", "t")
    assert [h["op"] for h in coll.history(conn, "a")] == ["delete", "create"]


# --- undo ---


def test_undo_add_removes_only_what_the_batch_created(conn, index):
    a = add_paper(index, "a.pdf", "H")
    b = add_paper(index, "b.pdf", "I")
    seeded(conn, index, a)
    batch = coll.add(conn, index, "c", [a, b], "t")["batch_id"]
    out = coll.undo(conn, batch, "ryan")
    assert out["reversed"] == 1 and out["undone"] == batch and out["op"] == "add"
    assert coll.resolve(conn, index, "c")["ids"] == [a]


def test_undo_is_a_history_row_and_marks_the_original(conn, index):
    a = add_paper(index, "a.pdf", "H")
    batch = seeded(conn, index, a)["batch_id"]
    out = coll.undo(conn, batch, "ryan")
    rows = coll.history(conn)
    assert rows[0]["batch_id"] == out["batch_id"]
    assert rows[0]["op"] == "undo" and rows[0]["by"] == "ryan" and rows[0]["count"] == 1
    assert rows[0]["undoes"] == batch
    original = next(r for r in rows if r["batch_id"] == batch)
    assert original["undone"] is True


def test_undoing_twice_is_a_clean_error(conn, index):
    a = add_paper(index, "a.pdf", "H")
    batch = seeded(conn, index, a)["batch_id"]
    coll.undo(conn, batch, "t")
    before = len(coll.history(conn))
    with pytest.raises(coll.AlreadyUndone):
        coll.undo(conn, batch, "t")
    assert len(coll.history(conn)) == before


def test_undo_unknown_batch_is_a_clean_error(conn):
    with pytest.raises(coll.UnknownBatch):
        coll.undo(conn, "nope", "t")


def test_an_undo_cannot_itself_be_undone(conn):
    batch = coll.create(conn, "c", by="t")["batch_id"]
    undo_batch = coll.undo(conn, batch, "t")["batch_id"]
    with pytest.raises(coll.CollectionError):
        coll.undo(conn, undo_batch, "t")


def test_undo_add_leaves_a_later_readd_alone(conn, index):
    a = add_paper(index, "a.pdf", "H")
    first = seeded(conn, index, a)["batch_id"]
    coll.remove(conn, "c", [a], "t", index)
    coll.add(conn, index, "c", [a], "t")
    out = coll.undo(conn, first, "t")
    assert out["reversed"] == 0
    assert out["skipped"][0]["reason"] == "already_removed"
    assert coll.resolve(conn, index, "c")["ids"] == [a]


def test_undo_remove_restores(conn, index):
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    batch = coll.remove(conn, "c", [a], "t", index)["batch_id"]
    assert coll.undo(conn, batch, "t")["reversed"] == 1
    r = coll.resolve(conn, index, "c")
    assert r["ids"] == [a] and r["members"][0]["added_by"] == "t"
    row = conn.execute("SELECT removed_at, removed_by FROM memberships").fetchone()
    assert row["removed_at"] is None and row["removed_by"] is None


def test_undo_remove_reports_a_membership_readded_since(conn, index):
    a = add_paper(index, "a.pdf", "H")
    b = add_paper(index, "b.pdf", "I")
    seeded(conn, index, a, b)
    batch = coll.remove(conn, "c", [a, b], "t", index)["batch_id"]
    readd = coll.add(conn, index, "c", [a], "someone", why="back")["batch_id"]
    out = coll.undo(conn, batch, "t")
    assert out["reversed"] == 1
    assert [s["reason"] for s in out["skipped"]] == ["re_added"]
    r = coll.resolve(conn, index, "c")
    assert sorted(r["ids"]) == sorted([a, b])
    held = next(m for m in r["members"] if m["id"] == a)
    assert held["batch_id"] == readd and held["why"] == "back"


def test_undo_remove_only_touches_that_batch(conn, index):
    a = add_paper(index, "a.pdf", "H")
    b = add_paper(index, "b.pdf", "I")
    seeded(conn, index, a, b)
    first = coll.remove(conn, "c", [a], "t", index)["batch_id"]
    coll.remove(conn, "c", [b], "t", index)
    coll.undo(conn, first, "t")
    assert coll.resolve(conn, index, "c")["ids"] == [a]


def test_undo_delete_brings_back_the_collection_and_members(conn, index):
    a = add_paper(index, "a.pdf", "H")
    seeded(conn, index, a)
    batch = coll.delete(conn, "c", "t")["batch_id"]
    with pytest.raises(coll.UnknownCollection):
        coll.resolve(conn, index, "c")
    coll.undo(conn, batch, "t")
    assert coll.resolve(conn, index, "c")["ids"] == [a]


def test_undo_delete_refuses_when_the_name_is_taken(conn, index):
    coll.create(conn, "c", by="t")
    batch = coll.delete(conn, "c", "t")["batch_id"]
    coll.create(conn, "C", by="t")
    with pytest.raises(coll.UndoConflict):
        coll.undo(conn, batch, "t")
    assert [h["undone"] for h in coll.history(conn) if h["batch_id"] == batch] == [False]
    assert len(coll.list_collections(conn)) == 1


def test_undo_rename_restores_the_old_name(conn, index):
    coll.create(conn, "a", by="t")
    batch = coll.rename(conn, "a", "b", "t")["batch_id"]
    coll.undo(conn, batch, "t")
    assert [c["name"] for c in coll.list_collections(conn)] == ["a"]


def test_undo_rename_refuses_if_the_old_name_was_taken(conn):
    coll.create(conn, "a", by="t")
    batch = coll.rename(conn, "a", "b", "t")["batch_id"]
    coll.create(conn, "a", by="t")
    with pytest.raises(coll.UndoConflict):
        coll.undo(conn, batch, "t")


def test_undo_rename_refuses_if_renamed_again(conn):
    coll.create(conn, "a", by="t")
    batch = coll.rename(conn, "a", "b", "t")["batch_id"]
    coll.rename(conn, "b", "c", "t")
    with pytest.raises(coll.UndoConflict):
        coll.undo(conn, batch, "t")
    assert [c["name"] for c in coll.list_collections(conn)] == ["c"]


def test_undo_create_tombstones_an_empty_collection(conn):
    batch = coll.create(conn, "c", by="t")["batch_id"]
    coll.undo(conn, batch, "t")
    assert coll.list_collections(conn) == []
    coll.create(conn, "c", by="t")


def test_undo_create_refuses_to_take_later_members_with_it(conn, index):
    a = add_paper(index, "a.pdf", "H")
    batch = coll.create(conn, "c", by="t")["batch_id"]
    coll.add(conn, index, "c", [a], "t")
    with pytest.raises(coll.UndoConflict):
        coll.undo(conn, batch, "t")
    assert coll.resolve(conn, index, "c")["ids"] == [a]


def test_undo_inside_a_deleted_collection_is_refused(conn, index):
    a = add_paper(index, "a.pdf", "H")
    batch = seeded(conn, index, a)["batch_id"]
    coll.delete(conn, "c", "t")
    with pytest.raises(coll.UndoConflict):
        coll.undo(conn, batch, "t")


def test_undo_requires_by(conn):
    batch = coll.create(conn, "c", by="t")["batch_id"]
    with pytest.raises(coll.CollectionError):
        coll.undo(conn, batch, "")


def test_undo_still_finds_a_membership_whose_key_was_refreshed(conn, index):
    a = add_paper(index, "a.pdf", "H")
    batch = seeded(conn, index, a)["batch_id"]
    index.execute("UPDATE papers SET file_hash = 'H2' WHERE id = ?", (a,))
    coll.refresh(conn, index, None)
    assert coll.undo(conn, batch, "t")["reversed"] == 1
    assert coll.resolve(conn, index, "c")["ids"] == []
