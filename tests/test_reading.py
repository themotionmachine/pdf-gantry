"""reading.py: a spaced-repetition reading inbox and a reading log in reading.db.

Built like collections.db: items are keyed on file_hash + path and matched at
read time, every write is a batch with by and why, tombstones, undo, reads
never write. Times are UTC ISO strings; every clock-reading function takes now=.
"""

import os
import sqlite3
import stat
import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from pdf_gantry import reading as rd
from pdf_gantry.config import Config
from pdf_gantry.db import get_connection

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)


def at(days=0.0):
    return NOW + timedelta(days=days)


def iso(dt):
    return dt.isoformat(timespec="milliseconds")


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
    c = rd.connect(cfg)
    yield c
    c.close()


@pytest.fixture
def pdir(tmp_path):
    d = tmp_path / "papers"
    d.mkdir()
    return d


def enqueue(conn, index, *ids, **kw):
    kw.setdefault("now", NOW)
    return rd.enqueue(conn, index, list(ids), kw.pop("by", "t"), **kw)


def item_of(out, pid):
    return next(i["item_id"] for i in out["items"] if i["paper_id"] == pid)


# --- storage ---


def test_connect_creates_private_sidecar_in_wal_mode(cfg):
    conn = rd.connect(cfg)
    path = cfg.index_dir / "reading.db"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA user_version").fetchone()[0] == rd.SCHEMA_VERSION
    conn.close()


def test_connect_does_not_touch_index_db(cfg):
    rd.connect(cfg).close()
    assert not cfg.db_path.exists()


def test_connect_refuses_a_newer_schema(cfg):
    conn = rd.connect(cfg)
    conn.execute(f"PRAGMA user_version = {rd.SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(rd.ReadingError):
        rd.connect(cfg)
    with pytest.raises(rd.ReadingError):
        rd.connect_readonly(cfg)


def test_connect_readonly_never_creates_the_file(cfg):
    assert rd.connect_readonly(cfg) is None
    assert not (cfg.index_dir / "reading.db").exists()


def test_connect_readonly_reads_and_cannot_write(cfg, conn, index):
    a = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, a)
    ro = rd.connect_readonly(cfg)
    try:
        assert [i["paper_id"] for i in rd.due(ro, index, now=at(1))] == [a]
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("DELETE FROM items")
    finally:
        ro.close()


# --- scheduler ---


@pytest.mark.parametrize("prev, verdict, reviews, expected", [
    (None, "good", 0, 1.0),       # first interval is a day
    (None, "later", 0, 1.0),
    (None, "doubt", 0, 1.0),
    (None, "key", 0, 7.0),        # key never schedules sooner than a week
    (1.0, "key", 1, 7.0),
    (10.0, "key", 3, 30.0),
    (4.0, "good", 2, 8.0),
    (4.0, "later", 2, 6.0),       # no penalty: still grows
    (10.0, "doubt", 3, 2.0),      # doubt drops back to 1-2 days
    (3.0, "doubt", 3, 1.5),
    (1.0, "doubt", 1, 1.0),
    (8.0, "done", 4, None),       # retired
])
def test_next_interval(prev, verdict, reviews, expected):
    assert rd.next_interval(prev, verdict, reviews) == expected


def test_scheduler_constants_are_module_level():
    assert rd.FIRST_INTERVAL_DAYS == 1.0
    assert rd.KEY_MULTIPLIER == 3.0 and rd.KEY_MIN_DAYS == 7.0
    assert rd.GOOD_MULTIPLIER == 2.0 and rd.LATER_MULTIPLIER == 1.5
    assert (rd.DOUBT_MIN_DAYS, rd.DOUBT_MAX_DAYS) == (1.0, 2.0)
    assert rd.VERDICTS == ("key", "good", "doubt", "later", "done")


def test_next_interval_refuses_an_unknown_verdict():
    with pytest.raises(rd.InvalidArgumentError):
        rd.next_interval(1.0, "meh", 1)


# --- enqueue ---


def test_enqueue_makes_a_due_item_with_provenance(conn, index):
    a = add_paper(index, "a.pdf", "H", title="Alpha", citekey="alpha2020")
    out = enqueue(conn, index, a, by="ryan", why="for ch3", source="agent")
    assert out["batch_id"] and out["added"] == [a]
    assert out["already_present"] == [] and out["not_found"] == []
    [item] = rd.due(conn, index, now=NOW)
    assert item["item_id"] == item_of(out, a)
    assert item["paper_id"] == a and item["file_hash"] == "H" and item["title"] == "Alpha"
    assert (item["source"], item["by"], item["why"]) == ("agent", "ryan", "for ch3")
    assert item["added_at"] == iso(NOW) and item["due_at"] == iso(NOW)
    assert item["status"] == "due" and item["passage"] is None
    assert item["interval_days"] is None and item["reviews"] == 0
    assert item["last_verdict"] is None


def test_enqueueing_a_whole_paper_again_is_a_reported_no_op(conn, index):
    a = add_paper(index, "a.pdf", "H")
    first = enqueue(conn, index, a)
    again = enqueue(conn, index, a, a)
    assert again["added"] == [] and again["already_present"] == [a]
    assert item_of(again, a) == item_of(first, a)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1
    assert rd.history(conn)[0]["count"] == 0


def test_already_present_sees_an_item_held_under_a_drifted_key(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, old)
    new = add_paper(index, "renamed.pdf", "H")
    touch(pdir, "renamed.pdf")
    out = enqueue(conn, index, new, papers_dir=pdir)
    assert out["already_present"] == [new]
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1


def test_a_paper_holds_several_passages_but_one_whole_paper_item(conn, index):
    a = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, a)
    p1 = enqueue(conn, index, a, passage={"page": 3, "quote": "q", "note": None})
    p2 = enqueue(conn, index, a, passage={"page": 7})
    same = enqueue(conn, index, a, passage={"page": 3, "quote": "q", "note": None})
    assert p1["added"] == [a] and p2["added"] == [a]
    assert same["already_present"] == [a] and item_of(same, a) == item_of(p1, a)
    items = rd.items(conn, index, now=NOW)
    passages = sorted((i["passage"] or {}).get("page", 0) for i in items)
    assert passages == [0, 3, 7]
    assert next(i for i in items if i["item_id"] == item_of(p2, a))["passage"] == {
        "page": 7, "quote": None, "note": None}


@pytest.mark.parametrize("passage", [{"page": 0}, {"quote": "q"}, {"page": "x"},
                                     {"page": 2, "quote": 5}])
def test_bad_passages_are_refused(conn, index, passage):
    a = add_paper(index, "a.pdf", "H")
    with pytest.raises(rd.InvalidArgumentError):
        enqueue(conn, index, a, passage=passage)
    assert rd.history(conn) == []


def test_enqueue_reports_ids_not_in_the_index(conn, index):
    a = add_paper(index, "a.pdf", "H")
    out = enqueue(conn, index, a, 99)
    assert out["added"] == [a] and out["not_found"] == [99]


def test_source_must_be_known(conn, index):
    a = add_paper(index, "a.pdf", "H")
    for source in ("manual", "agent", "vault"):
        enqueue(conn, index, a, source=source, passage={"page": 1, "note": source})
    with pytest.raises(rd.InvalidArgumentError):
        enqueue(conn, index, a, source="cron")


def test_writes_require_by(conn, index):
    a = add_paper(index, "a.pdf", "H")
    with pytest.raises(rd.ReadingError):
        rd.enqueue(conn, index, [a], "", now=NOW)
    item = item_of(enqueue(conn, index, a), a)
    with pytest.raises(rd.ReadingError):
        rd.review(conn, item, "good", " ", now=NOW)
    with pytest.raises(rd.ReadingError):
        rd.record_open(conn, index, a, 1, 5, "", now=NOW)
    with pytest.raises(rd.ReadingError):
        rd.undo(conn, rd.history(conn)[0]["batch_id"], None)


def test_a_retired_paper_can_be_enqueued_afresh(conn, index):
    a = add_paper(index, "a.pdf", "H")
    first = item_of(enqueue(conn, index, a), a)
    rd.review(conn, first, "done", "t", now=NOW)
    out = enqueue(conn, index, a, now=at(30))
    assert out["added"] == [a] and item_of(out, a) != first


# --- review and due ---


def test_review_schedules_and_returns_the_item(conn, index):
    a = add_paper(index, "a.pdf", "H")
    item = item_of(enqueue(conn, index, a), a)
    out = rd.review(conn, item, "good", "ryan", now=NOW)
    assert out["item_id"] == item and out["batch_id"]
    assert out["interval_days"] == 1.0 and out["due_at"] == iso(at(1))
    assert out["reviews"] == 1 and out["last_verdict"] == "good"
    assert out["status"] == "scheduled"
    out = rd.review(conn, item, "good", "ryan", now=at(1))
    assert out["interval_days"] == 2.0 and out["due_at"] == iso(at(3))
    out = rd.review(conn, item, "key", "ryan", now=at(3))
    assert out["interval_days"] == 7.0 and out["due_at"] == iso(at(10))


def test_review_can_resolve_the_paper_when_given_the_index(conn, index):
    a = add_paper(index, "a.pdf", "H")
    item = item_of(enqueue(conn, index, a), a)
    assert rd.review(conn, item, "good", "t", now=NOW, index_conn=index)["paper_id"] == a


def test_done_retires(conn, index):
    a = add_paper(index, "a.pdf", "H")
    item = item_of(enqueue(conn, index, a), a)
    out = rd.review(conn, item, "done", "t", now=NOW)
    assert out["status"] == "done" and out["due_at"] is None and out["interval_days"] is None
    assert rd.due(conn, index, now=at(1000)) == []
    assert [i["item_id"] for i in rd.items(conn, index, status="done")] == [item]
    with pytest.raises(rd.ItemRetiredError):
        rd.review(conn, item, "good", "t", now=NOW)


def test_review_refuses_unknown_items_and_verdicts(conn, index):
    a = add_paper(index, "a.pdf", "H")
    item = item_of(enqueue(conn, index, a), a)
    with pytest.raises(rd.UnknownItemError):
        rd.review(conn, 999, "good", "t", now=NOW)
    with pytest.raises(rd.InvalidArgumentError):
        rd.review(conn, item, "meh", "t", now=NOW)
    assert len(rd.history(conn)) == 1


def test_due_is_oldest_first_and_respects_now_and_limit(conn, index):
    a, b, c = (add_paper(index, f"{n}.pdf", n) for n in "abc")
    ia = item_of(enqueue(conn, index, a), a)
    ib = item_of(enqueue(conn, index, b, now=at(-2)), b)
    ic = item_of(enqueue(conn, index, c, now=at(-1)), c)
    rd.review(conn, ia, "key", "t", now=NOW)  # due in a week
    assert [i["item_id"] for i in rd.due(conn, index, now=NOW)] == [ib, ic]
    assert [i["item_id"] for i in rd.due(conn, index, now=NOW, limit=1)] == [ib]
    assert [i["item_id"] for i in rd.due(conn, index, now=at(8))] == [ib, ic, ia]


def test_items_by_status(conn, index):
    a, b, c = (add_paper(index, f"{n}.pdf", n) for n in "abc")
    ia = item_of(enqueue(conn, index, a), a)
    ib = item_of(enqueue(conn, index, b), b)
    ic = item_of(enqueue(conn, index, c), c)
    rd.review(conn, ib, "good", "t", now=NOW)
    rd.review(conn, ic, "done", "t", now=NOW)

    def ids(status):
        return sorted(i["item_id"] for i in rd.items(conn, index, status=status, now=NOW))

    assert ids("active") == [ia, ib]
    assert ids("due") == [ia]
    assert ids("scheduled") == [ib]
    assert ids("done") == [ic]
    assert ids("all") == [ia, ib, ic]
    with pytest.raises(rd.InvalidArgumentError):
        rd.items(conn, index, status="nope")


def test_unresolvable_items_are_reported_missing_or_ambiguous(conn, index):
    a = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, a)
    index.execute("DELETE FROM papers")
    add_paper(index, "x.pdf", "H")
    add_paper(index, "y.pdf", "H")
    b = add_paper(index, "b.pdf", "G")
    enqueue(conn, index, b)
    index.execute("DELETE FROM papers WHERE id = ?", (b,))
    index.commit()
    got = {i["file_hash"]: i for i in rd.due(conn, index, now=NOW)}
    assert got["H"]["paper_id"] is None and got["H"]["resolution"] == "ambiguous"
    assert len(got["H"]["candidates"]) == 2
    assert got["G"]["paper_id"] is None and got["G"]["resolution"] == "missing"
    assert got["G"]["title"] == "b.pdf"  # the stored snapshot


def test_an_item_follows_its_paper_through_a_rename(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, old)
    new = add_paper(index, "renamed.pdf", "H")
    touch(pdir, "renamed.pdf")
    assert [i["paper_id"] for i in rd.due(conn, index, now=NOW, papers_dir=pdir)] == [new]


# --- times ---


@pytest.mark.parametrize("now", [
    datetime(2026, 11, 7, 7, 0, tzinfo=timezone(timedelta(hours=-5))),
    datetime(2026, 11, 7, 12, 0),        # naive is taken as UTC
    "2026-11-07T12:00:00+00:00",
    "2026-11-07T12:00:00Z",
])
def test_times_are_stored_as_utc_iso(conn, index, now):
    a = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, a, now=now)
    row = conn.execute("SELECT added_at, due_at FROM items").fetchone()
    assert row["added_at"] == row["due_at"] == "2026-11-07T12:00:00.000+00:00"


# --- the reading log ---


def test_record_open_appends_and_is_not_a_batch(conn, index):
    a = add_paper(index, "a.pdf", "H")
    out = rd.record_open(conn, index, a, 4, 12.5, "ramelli", now=NOW)
    assert out == {"paper_id": a, "file_hash": "H", "path": "a.pdf", "page": 4,
                   "seconds": 12.5, "by": "ramelli", "at": iso(NOW)}
    rd.record_open(conn, index, a, 5, 3, "ramelli", now=at(1))
    assert conn.execute("SELECT COUNT(*) FROM opens").fetchone()[0] == 2
    assert rd.history(conn) == []


def test_record_open_refuses_an_unknown_paper(conn, index):
    with pytest.raises(rd.UnknownPaperError):
        rd.record_open(conn, index, 42, 1, 1, "t", now=NOW)


def test_opens_are_indexed_on_hash_and_time(conn):
    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT MAX(at) FROM opens WHERE file_hash = 'H'"
    ).fetchall()
    assert any("opens_hash_at" in r[-1] for r in plan)


# --- state_for ---


def test_state_for_summarises_items_and_opens(conn, index):
    a, b, c, d = (add_paper(index, f"{n}.pdf", n) for n in "abcd")
    ia = item_of(enqueue(conn, index, a), a)
    ib = item_of(enqueue(conn, index, b), b)
    rd.review(conn, ib, "good", "t", now=NOW)
    rd.record_open(conn, index, a, 1, 5, "t", now=at(-1))
    rd.record_open(conn, index, a, 2, 5, "t", now=NOW)
    rd.record_open(conn, index, c, 1, 5, "t", now=NOW)
    out = rd.state_for(conn, index, [a, b, c, d], now=NOW)
    assert set(out) == {a, b, c}
    assert out[a] == {"item_id": ia, "status": "due", "due_at": iso(NOW),
                      "interval_days": None, "last_verdict": None,
                      "last_opened": iso(NOW), "opens": 2, "passages": 0}
    assert out[b]["status"] == "scheduled" and out[b]["last_verdict"] == "good"
    assert out[b]["opens"] == 0 and out[b]["last_opened"] is None
    assert out[c]["item_id"] is None and out[c]["status"] is None and out[c]["opens"] == 1


def test_state_for_prefers_the_whole_paper_item_and_counts_passages(conn, index):
    a = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, a, passage={"page": 2}, now=at(-5))
    whole = item_of(enqueue(conn, index, a), a)
    out = rd.state_for(conn, index, [a], now=NOW)[a]
    assert out["item_id"] == whole and out["passages"] == 1


def test_state_for_reports_a_retired_paper_as_done(conn, index):
    a = add_paper(index, "a.pdf", "H")
    rd.review(conn, item_of(enqueue(conn, index, a), a), "done", "t", now=NOW)
    assert rd.state_for(conn, index, [a], now=NOW)[a]["status"] == "done"


def test_state_for_matches_like_every_other_read(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, old)
    rd.record_open(conn, index, old, 1, 1, "t", now=NOW)
    new = add_paper(index, "renamed.pdf", "H")
    touch(pdir, "renamed.pdf")
    out = rd.state_for(conn, index, [old, new], papers_dir=pdir, now=NOW)
    assert set(out) == {new} and out[new]["opens"] == 1


def test_state_for_without_a_sidecar_is_empty(index):
    assert rd.state_for(None, index, [1, 2]) == {}


def test_state_for_is_fast(conn, index):
    for n in range(2000):
        index.execute(
            "INSERT INTO papers (path, filename, file_hash, file_size, file_modified,"
            " indexed_at, updated_at) VALUES (?, ?, ?, 1, '2024', '2024', '2024')",
            (f"p{n}.pdf", f"p{n}.pdf", f"h{n}"),
        )
    index.commit()
    ids = [r[0] for r in index.execute("SELECT id FROM papers ORDER BY id LIMIT 200")]
    enqueue(conn, index, *ids[:100])
    conn.executemany(
        "INSERT INTO opens (file_hash, path, page, seconds, by, at) VALUES (?, ?, 1, 1, 't', ?)",
        [(f"h{n % 300}", f"p{n % 300}.pdf", iso(at(-n / 100))) for n in range(5000)],
    )
    batch = ids[60:100]
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        out = rd.state_for(conn, index, batch, now=NOW)
        best = min(best, time.perf_counter() - t0)
    assert len(out) == 40
    assert best < 0.005, f"state_for took {best * 1000:.1f} ms"


# --- reads never write ---


def test_reads_never_write(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, old)
    add_paper(index, "b.pdf", "H")
    touch(pdir, "b.pdf")
    before = conn.total_changes
    rd.due(conn, index, now=NOW, papers_dir=pdir)
    rd.items(conn, index, status="all", papers_dir=pdir)
    rd.state_for(conn, index, [old], papers_dir=pdir, now=NOW)
    rd.history(conn)
    assert conn.total_changes == before
    assert conn.execute("SELECT path FROM items").fetchone()["path"] == "a.pdf"


# --- history and undo ---


def test_every_write_is_a_batch_with_by_and_why(conn, index):
    a = add_paper(index, "a.pdf", "H")
    item = item_of(enqueue(conn, index, a, by="claude", why="suggested"), a)
    rd.review(conn, item, "good", "ryan", now=NOW)
    h = rd.history(conn)
    assert [r["op"] for r in h] == ["review", "enqueue"]
    assert h[1]["by"] == "claude" and h[1]["why"] == "suggested" and h[1]["count"] == 1
    assert h[0]["by"] == "ryan" and h[0]["item_id"] == item and h[0]["verdict"] == "good"
    assert h[0]["undone"] is False
    assert len(rd.history(conn, limit=1)) == 1


def test_undo_enqueue_tombstones(conn, index):
    a, b = add_paper(index, "a.pdf", "H"), add_paper(index, "b.pdf", "G")
    out = enqueue(conn, index, a, b)
    u = rd.undo(conn, out["batch_id"], "t")
    assert u["op"] == "enqueue" and u["reversed"] == 2 and u["skipped"] == []
    assert rd.items(conn, index, status="all") == []
    assert conn.execute("SELECT COUNT(*) FROM items WHERE removed_at IS NOT NULL"
                        ).fetchone()[0] == 2
    assert rd.history(conn)[0]["op"] == "undo" and rd.history(conn)[1]["undone"] is True
    assert enqueue(conn, index, a)["added"] == [a]  # the tombstone frees the slot


def test_undo_review_restores_the_scheduling_state(conn, index):
    a = add_paper(index, "a.pdf", "H")
    item = item_of(enqueue(conn, index, a), a)
    rd.review(conn, item, "good", "t", now=NOW)
    second = rd.review(conn, item, "key", "t", now=at(1))
    rd.undo(conn, second["batch_id"], "t")
    [got] = rd.items(conn, index, now=at(1))
    assert (got["interval_days"], got["due_at"], got["reviews"], got["last_verdict"]) == (
        1.0, iso(at(1)), 1, "good")


def test_undo_done_unretires(conn, index):
    a = add_paper(index, "a.pdf", "H")
    item = item_of(enqueue(conn, index, a), a)
    done = rd.review(conn, item, "done", "t", now=NOW)
    rd.undo(conn, done["batch_id"], "t")
    [got] = rd.due(conn, index, now=NOW)
    assert got["item_id"] == item and got["status"] == "due" and got["reviews"] == 0


def test_undo_review_refuses_when_reviewed_again_since(conn, index):
    a = add_paper(index, "a.pdf", "H")
    item = item_of(enqueue(conn, index, a), a)
    first = rd.review(conn, item, "good", "t", now=NOW)
    rd.review(conn, item, "good", "t", now=at(1))
    with pytest.raises(rd.UndoConflictError):
        rd.undo(conn, first["batch_id"], "t")
    assert rd.items(conn, index)[0]["reviews"] == 2
    assert rd.history(conn)[0]["op"] == "review"  # the refused undo left no batch


def test_undo_done_refuses_when_the_paper_was_enqueued_afresh(conn, index):
    a = add_paper(index, "a.pdf", "H")
    done = rd.review(conn, item_of(enqueue(conn, index, a), a), "done", "t", now=NOW)
    enqueue(conn, index, a, now=at(1))
    with pytest.raises(rd.UndoConflictError):
        rd.undo(conn, done["batch_id"], "t")


def test_undo_review_refuses_on_a_removed_item(conn, index):
    a = add_paper(index, "a.pdf", "H")
    added = enqueue(conn, index, a)
    review = rd.review(conn, item_of(added, a), "good", "t", now=NOW)
    rd.undo(conn, added["batch_id"], "t")
    with pytest.raises(rd.UndoConflictError):
        rd.undo(conn, review["batch_id"], "t")


def test_undo_errors(conn, index):
    a = add_paper(index, "a.pdf", "H")
    out = enqueue(conn, index, a)
    u = rd.undo(conn, out["batch_id"], "t")
    with pytest.raises(rd.AlreadyUndoneError):
        rd.undo(conn, out["batch_id"], "t")
    with pytest.raises(rd.UnknownBatchError):
        rd.undo(conn, "nope", "t")
    with pytest.raises(rd.ReadingError):
        rd.undo(conn, u["batch_id"], "t")
    assert {e.code for e in (rd.AlreadyUndoneError(), rd.UnknownBatchError(),
                             rd.UndoConflictError(), rd.UnknownItemError(),
                             rd.ItemRetiredError(), rd.InvalidArgumentError(),
                             rd.UnknownPaperError())} == {
        "already_undone", "unknown_batch", "undo_conflict", "unknown_item",
        "item_retired", "invalid_argument", "not_found"}


# --- refresh ---


def test_refresh_persists_drifted_keys(conn, index, pdir):
    old = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, old)
    add_paper(index, "renamed.pdf", "H")
    touch(pdir, "renamed.pdf")
    out = rd.refresh(conn, index, pdir)
    assert out["updated"] == 1 and out["batch_id"]
    assert conn.execute("SELECT path FROM items").fetchone()["path"] == "renamed.pdf"


def test_refresh_with_nothing_drifted_writes_nothing(conn, index):
    a = add_paper(index, "a.pdf", "H")
    enqueue(conn, index, a)
    before = conn.total_changes
    out = rd.refresh(conn, index, None)
    assert out["updated"] == 0 and out["batch_id"] is None and out["checked"] == 1
    assert conn.total_changes == before


def test_refresh_survives_two_whole_paper_items_drifting_onto_one_key(conn, index, pdir):
    a = add_paper(index, "a.pdf", "H")
    b = add_paper(index, "b.pdf", "H")
    enqueue(conn, index, a, b)
    add_paper(index, "c.pdf", "H")
    touch(pdir, "c.pdf")
    out = rd.refresh(conn, index, pdir)
    assert out["updated"] == 1 and out["collided"] == 1


def test_refresh_if_present_never_creates_the_file(cfg, index):
    assert rd.refresh_if_present(cfg.index_dir, index, None) is None
    assert not (cfg.index_dir / "reading.db").exists()
