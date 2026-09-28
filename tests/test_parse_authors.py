"""utils.parse_authors: tolerant normalisation of stored author strings (B7)."""

import pytest

from pdf_gantry.utils import parse_authors


@pytest.mark.parametrize("value,expected", [
    (None, []),
    ("", []),
    ("   ", []),
    ("[]", []),
    ('["Soyoung Park", "Ann Lee"]', ["Soyoung Park", "Ann Lee"]),
    ("Erol Yayboke; Sam Brannen", ["Erol Yayboke", "Sam Brannen"]),
    ("Kai Xiang Teo", ["Kai Xiang Teo"]),
    ("Ann Smith and Bob Jones", ["Ann Smith", "Bob Jones"]),
    ("Grimmelmann, James; Zhang, Pengfei;", ["Grimmelmann, James", "Zhang, Pengfei"]),
    ('["Grimmelmann, James; Zhang, Pengfei;"]', ["Grimmelmann, James", "Zhang, Pengfei"]),
    ('["  Ann  ", "", null]', ["Ann"]),
    ("Anderson, Sandra", ["Anderson, Sandra"]),
    ('"Solo Author"', ["Solo Author"]),
    ("[not json", ["[not json"]),
    (["Already", "A List"], ["Already", "A List"]),
])
def test_parse_authors(value, expected):
    assert parse_authors(value) == expected
