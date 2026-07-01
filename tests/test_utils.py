"""Tests for shared utility helpers in pdf_gantry.utils."""

from pdf_gantry.utils import missing_ids


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
