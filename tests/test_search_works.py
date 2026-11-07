"""gantry search --works: opt-in folding of copies; default output unchanged."""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry import works
from pdf_gantry.cli import cli

from .search_helpers import add_paper, new_db, point_cli_at, vec

WORK_KEYS = ("work_key", "canonical_id", "copies")


@pytest.fixture
def env(tmp_path, monkeypatch):
    works.clear_cache()
    conn, db_path = new_db(tmp_path)
    add_paper(conn, 1, "a.pdf", title="Deepfakes", text="deepfakes deepfakes deepfakes",
              doc_vec=vec(1))
    add_paper(conn, 2, "a copy.pdf", title="Deepfakes", citekey="smith2020",
              text="deepfakes deepfakes", doc_vec=vec(1))
    add_paper(conn, 3, "c.pdf", title="Other", text="deepfakes", doc_vec=vec(3))
    conn.execute("UPDATE papers SET file_hash = 'dup' WHERE id IN (1, 2)")
    conn.commit()
    conn.close()
    point_cli_at(tmp_path, monkeypatch, db_path)
    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", lambda m, q: vec(1))
    yield db_path
    works.clear_cache()


def _run(args):
    return CliRunner().invoke(cli, args)


@pytest.mark.parametrize("mode", [["--fts"], []])
def test_default_json_is_unchanged(env, mode):
    r = _run(["search", "deepfakes", "--json", *mode])
    assert r.exit_code == 0, r.output
    results = json.loads(r.stdout)["results"]
    assert sorted(d["id"] for d in results) == [1, 2, 3]
    for d in results:
        assert not set(WORK_KEYS) & set(d)


@pytest.mark.parametrize("mode", [["--fts"], []])
def test_works_folds_copies_and_adds_fields(env, mode):
    r = _run(["search", "deepfakes", "--json", "--works", *mode])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    by_id = {d["id"]: d for d in payload["results"]}
    assert len(by_id) == 2 and 3 in by_id
    kept = (by_id.keys() - {3}).pop()  # whichever copy ranked first, its own id
    assert by_id[kept]["work_key"] == "work:1"
    assert by_id[kept]["canonical_id"] == 2  # the copy with a citekey
    assert by_id[kept]["copies"] == [m for m in (1, 2) if m != kept]
    assert by_id[3]["copies"] == [] and by_id[3]["work_key"] == "work:3"
    assert payload["returned"] == 2


def test_works_overfetches_so_limit_counts_works(env):
    r = _run(["search", "deepfakes", "--fts", "--json", "--works", "-n", "2"])
    ids = [d["id"] for d in json.loads(r.stdout)["results"]]
    assert len(ids) == 2 and 3 in ids


def test_works_ids_only_and_fields(env):
    r = _run(["search", "deepfakes", "--fts", "--works", "--ids-only"])
    assert r.exit_code == 0
    assert len(r.stdout.split()) == 2
    r = _run(["search", "deepfakes", "--fts", "--works", "--json",
              "--fields", "id,work_key,copies"])
    assert r.stderr == ""
    d = json.loads(r.stdout)["results"][0]
    assert set(d) == {"id", "work_key", "copies"}


def test_works_text_output_names_copies(env):
    r = _run(["search", "deepfakes", "--fts", "--works"])
    assert r.exit_code == 0
    assert "1 other copy" in r.stdout


def test_works_queries_file(env):
    r = CliRunner().invoke(cli, ["search", "--queries-file", "-", "--fts", "--works",
                                 "--json"], input="deepfakes\nother\n")
    assert r.exit_code == 0, r.output
    qs = json.loads(r.stdout)["queries"]
    assert qs[0]["returned"] == 2
    assert all("work_key" in d for q in qs for d in q["results"])


def test_works_no_results_exits_2(env):
    r = _run(["search", "zzzqqq", "--fts", "--works", "--json"])
    assert r.exit_code == 2


def test_help_documents_works(env):
    r = _run(["search", "--help"])
    assert "--works" in r.stdout and "copies" in r.stdout
