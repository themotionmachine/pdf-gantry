"""Search/semantic output: result metadata fields, --fields validation,
--format oneline, total/returned (B5, E2, B7)."""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli

from .search_helpers import add_paper, new_db, point_cli_at, vec


@pytest.fixture
def env(tmp_path, monkeypatch):
    conn, db_path = new_db(tmp_path)
    add_paper(conn, 1, "a.pdf", title="Deepfakes and Social Media", year=2020,
              authors='["Ann Smith", "Bob Jones"]', citekey="smith2020deepfakes",
              text="deepfakes social media election", doc_vec=vec(1))
    add_paper(conn, 2, "b.pdf", title=None, year=None, authors="Cara Doe; Dan Roe",
              text="deepfakes detection methods", doc_vec=vec(2))
    add_paper(conn, 3, "c.pdf", title="Unrelated\tTitle\nWith breaks", year=1999,
              text="gardening tips", doc_vec=vec(3))
    conn.close()
    point_cli_at(tmp_path, monkeypatch, db_path)
    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", lambda m, q: vec(1))
    return db_path


def _run(args):
    return CliRunner().invoke(cli, args)


def test_search_json_has_title_year_authors_citekey(env):
    r = _run(["search", "deepfakes", "--fts", "--json"])
    assert r.exit_code == 0, r.output
    res = {d["id"]: d for d in json.loads(r.stdout)["results"]}
    assert res[1]["title"] == "Deepfakes and Social Media"
    assert res[1]["year"] == 2020
    assert res[1]["authors"] == ["Ann Smith", "Bob Jones"]
    assert res[1]["citekey"] == "smith2020deepfakes"
    assert res[2]["title"] is None
    assert res[2]["authors"] == ["Cara Doe", "Dan Roe"]
    # existing keys preserved
    for key in ("filename", "path", "score", "snippet", "has_markdown", "has_embeddings"):
        assert key in res[1]


def test_semantic_json_has_title_year_authors_citekey(env):
    r = _run(["semantic", "deepfakes", "--doc-only", "--json"])
    assert r.exit_code == 0, r.output
    res = {d["id"]: d for d in json.loads(r.stdout)["results"]}
    assert res[1]["title"] == "Deepfakes and Social Media"
    assert res[1]["year"] == 2020
    assert res[1]["authors"] == ["Ann Smith", "Bob Jones"]
    assert res[1]["citekey"] == "smith2020deepfakes"


def test_search_fields_accepts_metadata_fields(env):
    r = _run(["search", "deepfakes", "--fts", "--json",
              "--fields", "id,title,year,authors,citekey"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.stdout)["results"][0]
    assert set(d) == {"id", "title", "year", "authors", "citekey"}
    assert r.stderr == ""


@pytest.mark.parametrize("cmd", [
    ["search", "deepfakes", "--fts"],
    ["semantic", "deepfakes", "--doc-only"],
    ["info", "--ids", "1"],
])
def test_unknown_fields_warn_on_stderr(env, cmd):
    r = _run([*cmd, "--json", "--fields", "id,bogus"])
    assert r.exit_code == 0, r.output
    assert "bogus" in r.stderr
    assert "id" in r.stderr  # the valid set is listed
    json.loads(r.stdout)  # stdout still clean JSON


def test_info_known_fields_no_warning(env):
    r = _run(["info", "--ids", "1", "--json", "--fields", "id,title"])
    assert r.exit_code == 0
    assert r.stderr == ""


def test_search_plain_text_shows_id_and_title(env):
    r = _run(["search", "deepfakes", "--fts"])
    assert r.exit_code == 0, r.output
    assert "[1]" in r.stdout
    assert "Deepfakes and Social Media" in r.stdout
    # No title: fall back to filename
    assert "[2]" in r.stdout and "b.pdf" in r.stdout


def test_semantic_plain_text_shows_id_and_title(env):
    r = _run(["semantic", "deepfakes", "--doc-only"])
    assert r.exit_code == 0, r.output
    assert "[1]" in r.stdout
    assert "Deepfakes and Social Media" in r.stdout


def test_search_format_oneline(env):
    r = _run(["search", "deepfakes", "--fts", "--format", "oneline"])
    assert r.exit_code == 0, r.output
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 2
    by_id = {ln.split("\t")[0]: ln.split("\t") for ln in lines}
    one = by_id["1"]
    assert len(one) == 5
    float(one[1])
    assert one[2:] == ["2020", "smith2020deepfakes", "Deepfakes and Social Media"]
    two = by_id["2"]
    assert two[2:] == ["", "", "b.pdf"]


def test_oneline_sanitises_tabs_and_newlines(env):
    r = _run(["semantic", "gardening", "--doc-only", "--format", "oneline"])
    assert r.exit_code == 0, r.output
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("3\t")][0]
    assert line.split("\t")[4] == "Unrelated Title With breaks"


def test_semantic_format_oneline(env):
    r = _run(["semantic", "deepfakes", "--doc-only", "--format", "oneline"])
    assert r.exit_code == 0, r.output
    for ln in r.stdout.strip().splitlines():
        assert len(ln.split("\t")) == 5


def test_format_oneline_beats_global_json(env):
    r = _run(["--json", "search", "deepfakes", "--fts", "--format", "oneline"])
    assert r.exit_code == 0, r.output
    assert not r.stdout.lstrip().startswith("{")


def test_ids_only_still_works(env):
    r = _run(["search", "deepfakes", "--fts", "--ids-only"])
    assert r.exit_code == 0
    assert sorted(r.stdout.split()) == ["1", "2"]


def test_total_and_returned_fts(env):
    r = _run(["search", "deepfakes", "--fts", "--json", "-n", "1"])
    payload = json.loads(r.stdout)
    assert payload["total"] == 2  # global match count
    assert payload["returned"] == 1


def test_total_and_returned_hybrid(env, monkeypatch):
    r = _run(["search", "deepfakes", "--json", "-n", "1"])
    payload = json.loads(r.stdout)
    assert payload["returned"] == 1
    assert payload["total"] == payload["returned"]


def test_total_and_returned_semantic(env):
    r = _run(["semantic", "deepfakes", "--doc-only", "--json", "-n", "2"])
    payload = json.loads(r.stdout)
    assert payload["returned"] == 2
    assert payload["total"] == 2


def test_help_documents_total(env):
    r = _run(["search", "--help"])
    assert "returned" in r.stdout and "total" in r.stdout
