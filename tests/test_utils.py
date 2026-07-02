"""Tests for shared utility helpers in pdf_gantry.utils."""

from pdf_gantry.db import get_connection
from pdf_gantry.utils import missing_ids, resolve_ids


def test_missing_ids_all_found():
    """When every requested ID resolved, nothing is missing."""
    assert missing_ids([1, 2, 3], {1, 2, 3}) == []


def test_missing_ids_some_missing():
    """Requested IDs absent from ``found`` are reported."""
    assert missing_ids([1, 2, 3], {1, 3}) == [2]


def test_missing_ids_preserves_request_order():
    """Output order follows the order IDs were requested, not sorted order."""
    assert missing_ids([5, 2, 9], {2}) == [5, 9]


def test_missing_ids_dedupes_requested():
    """A requested ID repeated in the input is only reported once."""
    assert missing_ids([4, 4, 4], set()) == [4]


def test_missing_ids_empty_requested():
    """Nothing was requested, so nothing can be missing."""
    assert missing_ids([], {1, 2}) == []


def test_missing_ids_empty_found():
    """Nothing resolved: every requested ID is missing."""
    assert missing_ids([1, 2], []) == [1, 2]


def _seed_papers(tmp_path, n):
    """Insert ``n`` bare papers; return (conn, ids) in insertion order."""
    conn = get_connection(str(tmp_path / "index.db"))
    ids = []
    for i in range(n):
        cur = conn.execute(
            "INSERT INTO papers "
            "(path, filename, file_hash, file_size, file_modified, indexed_at, updated_at) "
            "VALUES (?, ?, 'hash', 100, '2024-01-01', '2024-01-01', '2024-01-01')",
            (f"/papers/test{i}.pdf", f"test{i}.pdf"),
        )
        ids.append(cur.lastrowid)
    conn.commit()
    return conn, ids


def test_resolve_ids_all_found(tmp_path):
    conn, ids = _seed_papers(tmp_path, 2)
    found, not_found = resolve_ids(conn, ids)
    assert found == ids
    assert not_found == []


def test_resolve_ids_some_missing(tmp_path):
    conn, ids = _seed_papers(tmp_path, 2)
    bogus = 99999
    found, not_found = resolve_ids(conn, [ids[0], bogus])
    assert found == [ids[0]]
    assert not_found == [bogus]


def test_resolve_ids_preserves_request_order(tmp_path):
    """Order of ``found`` follows request order, not insertion/id order."""
    conn, ids = _seed_papers(tmp_path, 3)
    found, _ = resolve_ids(conn, [ids[2], ids[0]])
    assert found == [ids[2], ids[0]]


def test_resolve_ids_dedupes_requested(tmp_path):
    conn, ids = _seed_papers(tmp_path, 1)
    found, not_found = resolve_ids(conn, [ids[0], ids[0]])
    assert found == [ids[0]]
    assert not_found == []


def test_resolve_ids_empty_requested(tmp_path):
    conn, _ids = _seed_papers(tmp_path, 1)
    assert resolve_ids(conn, []) == ([], [])
