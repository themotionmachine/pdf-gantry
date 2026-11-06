"""display_title / title_source: metadata, then layout hint, then filename."""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.search import find_papers
from pdf_gantry.utils import resolve_title

from .search_helpers import add_paper, new_db, point_cli_at


def test_resolve_title_precedence():
    assert resolve_title("Meta", "Layout", "f.pdf") == ("Meta", "metadata")
    assert resolve_title(None, "Layout", "f.pdf") == ("Layout", "layout")
    assert resolve_title("  ", "Layout", "f.pdf") == ("Layout", "layout")
    assert resolve_title("", "", "f.pdf") == ("f.pdf", "filename")
    assert resolve_title(None, None, "f.pdf") == ("f.pdf", "filename")


@pytest.fixture
def lib(tmp_path, monkeypatch):
    conn, db_path = new_db(tmp_path)
    add_paper(conn, 1, "meta.pdf", title="Walrus Metadata Title", text="walrus body")
    add_paper(conn, 2, "layout.pdf", text="walrus body")
    add_paper(conn, 3, "bare.pdf", text="walrus body")
    conn.execute("UPDATE papers SET title_from_layout = 'Walrus Layout Hint', "
                 "title_layout_confidence = 'low' WHERE id = 2")
    conn.commit()
    point_cli_at(tmp_path, monkeypatch, db_path)
    return conn


def _by_id(rows):
    return {r["id"]: r for r in rows}


def _assert_shapes(rows):
    assert rows[1]["title"] == "Walrus Metadata Title"
    assert rows[1]["display_title"] == "Walrus Metadata Title"
    assert rows[1]["title_source"] == "metadata"
    assert rows[2]["title"] is None  # `title` keeps meaning verified metadata
    assert rows[2]["display_title"] == "Walrus Layout Hint"
    assert rows[2]["title_source"] == "layout"
    assert rows[2]["title_layout_confidence"] == "low"
    assert rows[3]["display_title"] == "bare.pdf"
    assert rows[3]["title_source"] == "filename"


def test_search_json(lib):
    out = CliRunner().invoke(cli, ["search", "walrus", "--fts", "--json"]).output
    _assert_shapes(_by_id(json.loads(out)["results"]))


def test_info_json(lib):
    out = CliRunner().invoke(cli, ["info", "1", "2", "3", "--json"]).output
    _assert_shapes(_by_id(json.loads(out)["papers"]))


def test_find_json(lib):
    out = CliRunner().invoke(cli, ["find", "pdf", "--json"]).output
    _assert_shapes(_by_id(json.loads(out)["results"]))


def test_semantic_json_carries_the_keys(tmp_path, monkeypatch):
    from .search_helpers import vec
    conn, db_path = new_db(tmp_path)
    add_paper(conn, 1, "a.pdf", text="x", doc_vec=vec(1))
    conn.execute("UPDATE papers SET title_from_layout = 'Hint' WHERE id = 1")
    conn.commit()
    point_cli_at(tmp_path, monkeypatch, db_path)
    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", lambda *a, **k: vec(1))
    out = CliRunner().invoke(cli, ["semantic", "x", "--doc-only", "--json"]).output
    r = json.loads(out)["results"][0]
    assert (r["display_title"], r["title_source"]) == ("Hint", "layout")


def test_fields_can_select_display_title(lib):
    out = CliRunner().invoke(
        cli, ["search", "walrus", "--fts", "--json", "--fields", "id,display_title"]
    ).output
    assert all(set(r) == {"id", "display_title"} for r in json.loads(out)["results"])
    out = CliRunner().invoke(
        cli, ["info", "2", "--json", "--fields", "id,display_title,title_source"]
    ).output
    assert json.loads(out)["papers"][0]["title_source"] == "layout"


def test_oneline_and_plain_fall_back(lib):
    runner = CliRunner()
    one = runner.invoke(cli, ["search", "walrus", "--fts", "--format", "oneline"]).output
    lines = {ln.split("\t")[0]: ln.split("\t")[-1] for ln in one.strip().splitlines()}
    assert lines == {"1": "Walrus Metadata Title", "2": "Walrus Layout Hint", "3": "bare.pdf"}
    plain = runner.invoke(cli, ["search", "walrus", "--fts"]).output
    assert "Walrus Layout Hint" in plain


def test_plain_info_marks_the_hint_as_unverified(lib):
    plain = CliRunner().invoke(cli, ["info", "2"]).output
    assert "Walrus Layout Hint" in plain and "unverified" in plain
    assert "Title:" not in CliRunner().invoke(cli, ["info", "3"]).output


def test_plain_find_falls_back(lib):
    assert "Walrus Layout Hint" in CliRunner().invoke(cli, ["find", "layout"]).output


def test_find_matches_layout_title(lib):
    hits = find_papers(lib, "layout hint")
    assert [h["id"] for h in hits] == [2]
    assert "title_from_layout" in hits[0]["matched_fields"]


def test_find_ranks_hint_below_verified_title(tmp_path):
    conn, _ = new_db(tmp_path)
    add_paper(conn, 1, "z.pdf", title="Otter Economics", text="")
    add_paper(conn, 2, "a.pdf", text="")
    conn.execute("UPDATE papers SET title_from_layout = 'Otter Economics' WHERE id = 2")
    conn.commit()
    assert [h["id"] for h in find_papers(conn, "Otter Economics")] == [1, 2]


