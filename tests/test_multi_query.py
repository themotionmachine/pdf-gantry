"""--queries-file: many queries, one process, one model load (E5)."""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli

from .search_helpers import add_paper, new_db, point_cli_at, vec


@pytest.fixture
def env(tmp_path, monkeypatch):
    conn, db_path = new_db(tmp_path)
    add_paper(conn, 1, "a.pdf", title="Deepfakes", year=2020, citekey="a2020",
              text="deepfakes election", doc_vec=vec(1))
    add_paper(conn, 2, "b.pdf", title="Climate", year=2021, citekey="b2021",
              text="climate adaptation", doc_vec=vec(2))
    conn.close()
    point_cli_at(tmp_path, monkeypatch, db_path)
    calls = []

    def fake_embed(model, q):
        calls.append(q)
        return vec(1) if "deep" in q else vec(2)

    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", fake_embed)
    qfile = tmp_path / "queries.txt"
    qfile.write_text("deepfakes\n\nclimate\nzzznomatch\n")
    return qfile, calls


def _run(args, input=None):
    return CliRunner().invoke(cli, args, input=input)


def test_search_queries_file_json(env):
    qfile, _ = env
    r = _run(["search", "--queries-file", str(qfile), "--fts", "--json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    qs = payload["queries"]
    assert [q["query"] for q in qs] == ["deepfakes", "climate", "zzznomatch"]
    assert [q["mode"] for q in qs] == ["fts_only"] * 3
    assert [d["id"] for d in qs[0]["results"]] == [1]
    assert [d["id"] for d in qs[1]["results"]] == [2]
    assert qs[2]["results"] == []
    assert qs[0]["returned"] == 1 and "total" in qs[0]


def test_search_queries_from_stdin(env):
    r = _run(["search", "--queries-file", "-", "--fts", "--json"], input="deepfakes\nclimate\n")
    assert r.exit_code == 0, r.output
    assert len(json.loads(r.stdout)["queries"]) == 2


def test_semantic_queries_file_json_embeds_each_query(env):
    qfile, calls = env
    r = _run(["semantic", "--queries-file", str(qfile), "--doc-only", "--json", "-n", "1"])
    assert r.exit_code == 0, r.output
    qs = json.loads(r.stdout)["queries"]
    assert [q["results"][0]["id"] for q in qs] == [1, 2, 2]
    assert calls == ["deepfakes", "climate", "zzznomatch"]


def test_hybrid_queries_file(env):
    qfile, _ = env
    r = _run(["search", "--queries-file", str(qfile), "--json"])
    assert r.exit_code == 0, r.output
    qs = json.loads(r.stdout)["queries"]
    assert qs[0]["mode"] == "hybrid"


def test_oneline_prefixes_query_index(env):
    qfile, _ = env
    r = _run(["search", "--queries-file", str(qfile), "--fts", "--format", "oneline"])
    assert r.exit_code == 0, r.output
    lines = [ln.split("\t") for ln in r.stdout.strip().splitlines()]
    assert [(ln[0], ln[1]) for ln in lines] == [("0", "1"), ("1", "2")]
    assert all(len(ln) == 6 for ln in lines)


def test_ids_only_is_deduplicated_union(env, tmp_path):
    q = tmp_path / "q2.txt"
    q.write_text("deepfakes\ndeepfakes OR climate\nclimate\n")
    r = _run(["search", "--queries-file", str(q), "--fts", "--ids-only"])
    assert r.exit_code == 0, r.output
    assert r.stdout.split() == ["1", "2"]


def test_all_queries_empty_exits_2(env, tmp_path):
    q = tmp_path / "q3.txt"
    q.write_text("zzz\nyyy\n")
    r = _run(["search", "--queries-file", str(q), "--fts", "--json"])
    assert r.exit_code == 2
    assert len(json.loads(r.stdout)["queries"]) == 2


def test_query_and_queries_file_are_exclusive(env):
    qfile, _ = env
    r = _run(["search", "x", "--queries-file", str(qfile), "--json"])
    assert r.exit_code == 1
    assert "error" in json.loads(r.stdout)


def test_missing_query_is_error(env):
    r = _run(["search", "--json"])
    assert r.exit_code == 1
    assert "error" in json.loads(r.stdout)


def test_per_query_fts_syntax_error_is_partial(env, tmp_path):
    q = tmp_path / "q4.txt"
    q.write_text("deepfakes\nSocial: Media\n")
    r = _run(["search", "--queries-file", str(q), "--fts", "--fts-syntax", "--json"])
    assert r.exit_code == 3, r.output
    qs = json.loads(r.stdout)["queries"]
    assert "error" in qs[1]
    assert [d["id"] for d in qs[0]["results"]] == [1]


def test_query_model_loaded_once(monkeypatch):
    """embed_query reuses one loaded model across calls in a process."""
    from pdf_gantry import embeddings

    loads = []

    class M:
        def encode(self, text, show_progress_bar=False):
            return [0.0] * 4

    def loader(name):
        loads.append(name)
        return M()

    monkeypatch.setattr(embeddings, "_get_embedding_model", loader)
    embeddings._QUERY_MODEL_CACHE.clear()
    embeddings.embed_query("m", "a")
    embeddings.embed_query("m", "b")
    embeddings.embed_query("m", "c")
    assert loads == ["m"]
    embeddings._QUERY_MODEL_CACHE.clear()
