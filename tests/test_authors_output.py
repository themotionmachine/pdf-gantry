"""authors is a real JSON list in info/find/read output (B7, authorised shape change)."""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli

from .search_helpers import add_paper, new_db, point_cli_at


@pytest.fixture
def env(tmp_path, monkeypatch):
    conn, db_path = new_db(tmp_path)
    add_paper(conn, 1, "smith_deepfakes.pdf", title="Deepfakes",
              authors='["Ann Smith", "Bob Jones"]', text="deepfakes text",
              chunks=["deepfakes chunk"])
    add_paper(conn, 2, "doe_climate.pdf", title="Climate", authors="Cara Doe; Dan Roe",
              text="climate text")
    add_paper(conn, 3, "anon.pdf", title=None, authors=None, text="anon text")
    conn.close()
    point_cli_at(tmp_path, monkeypatch, db_path)


def _json(args):
    r = CliRunner().invoke(cli, args)
    assert r.exit_code == 0, r.output
    return json.loads(r.stdout)


def test_info_authors_list(env):
    papers = {p["id"]: p for p in _json(["info", "--ids", "1,2,3", "--json"])["papers"]}
    assert papers[1]["authors"] == ["Ann Smith", "Bob Jones"]
    assert papers[2]["authors"] == ["Cara Doe", "Dan Roe"]
    assert papers[3]["authors"] == []


def test_info_text_joins_authors(env):
    r = CliRunner().invoke(cli, ["info", "--ids", "1"])
    assert "Authors: Ann Smith; Bob Jones" in r.stdout


def test_find_authors_list(env):
    res = _json(["find", "doe", "--json"])["results"]
    assert res[0]["authors"] == ["Cara Doe", "Dan Roe"]


def test_read_authors_list(env):
    assert _json(["read", "1", "--json"])["authors"] == ["Ann Smith", "Bob Jones"]
    assert _json(["read", "3", "--json"])["authors"] == []
