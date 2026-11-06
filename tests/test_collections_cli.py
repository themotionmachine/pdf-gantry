"""`gantry collection ...` and `gantry search --collection`."""

import json
import os
import stat

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli

from .search_helpers import add_paper, new_db


@pytest.fixture
def env(tmp_path, monkeypatch):
    conn, db_path = new_db(tmp_path)
    add_paper(conn, 1, "one.pdf", title="Alpha", citekey="alpha2020", text="glacier ice")
    add_paper(conn, 2, "two.pdf", title="Beta", text="glacier melt")
    add_paper(conn, 3, "three.pdf", title="Gamma", text="glacier retreat")
    papers = tmp_path / "papers"
    papers.mkdir()
    conn.execute("UPDATE papers SET path = filename")
    conn.commit()
    for name in ("one.pdf", "two.pdf", "three.pdf"):
        (papers / name).write_bytes(b"%PDF")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"index_dir": str(tmp_path), "papers_dir": str(papers)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg)
    for var in ("GANTRY_INDEX_DIR", "GANTRY_PAPERS_DIR"):
        monkeypatch.delenv(var, raising=False)
    yield conn, tmp_path
    conn.close()


def run(*args, stdin=None):
    return CliRunner().invoke(cli, list(args), input=stdin)


def gantry_json(*args, stdin=None):
    result = run(*args, "--json", stdin=stdin)
    return result, json.loads(result.stdout)


def make(name="c", ids="1,2"):
    assert run("collection", "create", name, "--by", "t").exit_code == 0
    if ids:
        assert run("collection", "add", name, "--ids", ids, "--by", "t").exit_code == 0


def test_create_add_show_round_trip(env):
    r, out = gantry_json("collection", "create", "c", "--note", "n", "--by", "ryan")
    assert r.exit_code == 0 and out["name"] == "c" and out["note"] == "n" and out["batch_id"]
    r, out = gantry_json("collection", "add", "c", "--ids", "1,2", "--by", "a", "--why", "w")
    assert r.exit_code == 0 and out["added"] == [1, 2] and out["batch_id"]
    r, out = gantry_json("collection", "show", "c")
    assert r.exit_code == 0
    assert out["ids"] == [1, 2]
    assert out["members"][0]["added_by"] == "a" and out["members"][0]["why"] == "w"
    assert out["collection"]["note"] == "n"


def test_sidecar_is_private(env):
    make()
    path = env[1] / "collections.db"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


@pytest.mark.parametrize("args", [
    ("create", "c"),
    ("add", "c", "--ids", "1"),
    ("remove", "c", "--ids", "1"),
    ("rename", "c", "d"),
    ("delete", "c"),
    ("undo", "abc"),
])
def test_by_is_required_on_writes(env, args):
    assert run("collection", *args).exit_code == 64


def test_add_from_stdin_takes_ids_only_output(env):
    make(ids=None)
    ids = run("search", "glacier", "--fts", "--ids-only").stdout
    assert len(ids.split()) == 3
    r, out = gantry_json("collection", "add", "c", "--ids", "-", "--by", "t", stdin=ids)
    assert r.exit_code == 0 and sorted(out["added"]) == [1, 2, 3]


def test_remove_from_stdin_with_spaces(env):
    make()
    r, out = gantry_json("collection", "remove", "c", "--ids", "-", "--by", "t", stdin="1 2\n")
    assert r.exit_code == 0 and out["removed"] == [1, 2]


def test_add_with_no_ids_is_no_results_and_writes_nothing(env):
    make(ids=None)
    r = run("collection", "add", "c", "--ids", "-", "--by", "t", stdin="")
    assert r.exit_code == 2
    assert len(gantry_json("collection", "history")[1]["history"]) == 1


def test_add_bad_ids_is_an_error(env):
    make(ids=None)
    assert run("collection", "add", "c", "--ids", "1,x", "--by", "t").exit_code == 1


def test_add_exit_codes_report_not_found(env):
    make(ids=None)
    r, out = gantry_json("collection", "add", "c", "--ids", "1,99", "--by", "t")
    assert r.exit_code == 3 and out["not_found"] == [99]
    assert run("collection", "add", "c", "--ids", "98", "--by", "t").exit_code == 2


def test_add_already_present_is_success(env):
    make()
    r, out = gantry_json("collection", "add", "c", "--ids", "1", "--by", "t")
    assert r.exit_code == 0 and out["already_present"] == [1]


def test_show_empty_exits_2(env):
    make(ids=None)
    assert run("collection", "show", "c").exit_code == 2


def test_show_with_missing_exits_3_and_ids_only_keeps_stdout_bare(env):
    conn, _ = env
    make()
    conn.execute("DELETE FROM papers WHERE id = 2")
    conn.commit()
    r = run("collection", "show", "c", "--ids-only")
    assert r.exit_code == 3
    assert r.stdout == "1\n"
    assert "1 missing" in r.stderr


def test_show_with_ambiguous_exits_3(env):
    conn, _ = env
    make(ids="1")
    conn.execute("DELETE FROM papers WHERE id = 1")
    conn.execute("UPDATE papers SET file_hash = 'h1'")
    conn.commit()
    r, out = gantry_json("collection", "show", "c")
    assert r.exit_code == 3
    assert out["ambiguous"][0]["candidates"] == [2, 3]


def test_show_text_names_members_and_problems(env):
    conn, _ = env
    make()
    conn.execute("DELETE FROM papers WHERE id = 2")
    conn.commit()
    out = run("collection", "show", "c").stdout
    assert "[1]" in out and "Alpha" in out and "@alpha2020" in out
    assert "two.pdf" in out


def test_show_uses_papers_dir_for_liveness(env):
    conn, tmp = env
    make(ids="1")
    conn.execute(
        "INSERT INTO papers (id, path, filename, file_hash, file_size, file_modified,"
        " indexed_at, updated_at) VALUES (9, 'moved.pdf', 'moved.pdf', 'h1', 1, '', '', '')"
    )
    conn.commit()
    (tmp / "papers" / "one.pdf").rename(tmp / "papers" / "moved.pdf")
    assert gantry_json("collection", "show", "c")[1]["ids"] == [9]


def test_show_skips_liveness_when_papers_dir_is_absent(env):
    _, tmp = env
    make()
    for f in (tmp / "papers").iterdir():
        f.unlink()
    (tmp / "papers").rmdir()
    r, out = gantry_json("collection", "show", "c")
    assert r.exit_code == 0 and out["ids"] == [1, 2] and out["missing"] == []


def test_remove_undo_via_cli(env):
    make()
    r, out = gantry_json("collection", "remove", "c", "--ids", "1,99", "--by", "t")
    assert r.exit_code == 3 and out["removed"] == [1] and out["not_found"] == [99]
    r, undone = gantry_json("collection", "undo", out["batch_id"], "--by", "t")
    assert r.exit_code == 0 and undone["reversed"] == 1
    assert gantry_json("collection", "show", "c")[1]["ids"] == [1, 2]


def test_undo_twice_exits_1_with_a_code(env):
    make(ids=None)
    batch = gantry_json("collection", "history")[1]["history"][0]["batch_id"]
    assert run("collection", "undo", batch, "--by", "t").exit_code == 0
    r, out = gantry_json("collection", "undo", batch, "--by", "t")
    assert r.exit_code == 1 and out["code"] == "already_undone" and out["error"]


def test_undo_with_skipped_rows_exits_3(env):
    make()
    batch = gantry_json("collection", "remove", "c", "--ids", "1", "--by", "t")[1]["batch_id"]
    run("collection", "add", "c", "--ids", "1", "--by", "t")
    r, out = gantry_json("collection", "undo", batch, "--by", "t")
    assert r.exit_code == 3 and out["skipped"][0]["reason"] == "re_added"


def test_rename_delete_list_history(env):
    make()
    make("other", ids=None)
    r, out = gantry_json("collection", "rename", "c", "d", "--by", "t")
    assert r.exit_code == 0 and out["name"] == "d"
    r, out = gantry_json("collection", "list")
    assert [(c["name"], c["count"]) for c in out["collections"]] == [("d", 2), ("other", 0)]
    assert out["collections"][0]["resolved"] == 2
    r, out = gantry_json("collection", "delete", "other", "--by", "t")
    assert r.exit_code == 0
    r, out = gantry_json("collection", "history", "d", "--limit", "2")
    assert [h["op"] for h in out["history"]] == ["rename", "add"]
    assert "rename" in run("collection", "history").stdout


def test_name_conflict_exits_1(env):
    make(ids=None)
    r, out = gantry_json("collection", "create", "C", "--by", "t")
    assert r.exit_code == 1 and out["code"] == "name_conflict"


def test_unknown_collection_is_a_usage_error_on_show_and_writes(env):
    make(ids=None)
    r, out = gantry_json("collection", "show", "nope")
    assert r.exit_code == 64 and "nope" in out["error"]
    assert run("collection", "add", "nope", "--ids", "1", "--by", "t").exit_code == 64


def test_reads_with_no_sidecar_never_create_it(env):
    _, tmp = env
    path = tmp / "collections.db"
    r, out = gantry_json("collection", "list")
    assert r.exit_code == 0 and out == {"count": 0, "collections": []}
    assert run("collection", "list").exit_code == 0
    assert run("collection", "show", "c").exit_code == 64
    r, out = gantry_json("collection", "history")
    assert r.exit_code == 0 and out["history"] == []
    assert run("collection", "history", "c").exit_code == 64
    assert run("search", "glacier", "--fts", "--collection", "c").exit_code == 64
    assert not path.exists()


def _snapshot(tmp):
    path = tmp / "collections.db"
    return path.stat().st_mtime_ns, path.read_bytes()


@pytest.mark.parametrize("args", [
    ("collection", "show", "c"),
    ("collection", "show", "c", "--ids-only"),
    ("collection", "list"),
    ("collection", "history"),
    ("collection", "history", "c"),
    ("search", "glacier", "--fts", "--collection", "c"),
])
def test_read_commands_do_not_write_the_sidecar(env, args):
    conn, tmp = env
    make()
    conn.execute(
        "INSERT INTO papers (id, path, filename, file_hash, file_size, file_modified,"
        " indexed_at, updated_at) VALUES (9, 'moved.pdf', 'moved.pdf', 'h1', 1, '', '', '')"
    )
    conn.commit()
    (tmp / "papers" / "one.pdf").rename(tmp / "papers" / "moved.pdf")
    before = _snapshot(tmp)
    assert run(*args).exit_code in (0, 3)
    assert _snapshot(tmp) == before


def test_show_does_not_write_index_db(env):
    conn, tmp = env
    make()
    conn.commit()
    before = (tmp / "index.db").stat().st_mtime_ns
    run("collection", "show", "c")
    run("collection", "list")
    assert (tmp / "index.db").stat().st_mtime_ns == before


def test_show_without_index_db_errors_cleanly(env):
    conn, tmp = env
    make()
    conn.close()
    for f in tmp.glob("index.db*"):
        f.unlink()
    assert run("collection", "show", "c").exit_code == 1


def test_search_collection_restricts_results(env):
    make()
    r, out = gantry_json("search", "glacier", "--fts", "--collection", "c")
    assert r.exit_code == 0
    assert sorted(x["id"] for x in out["results"]) == [1, 2]


def test_search_collection_intersects_restrict_to_ids(env):
    make()
    r, out = gantry_json("search", "glacier", "--fts", "--collection", "c",
                         "--restrict-to-ids", "2,3")
    assert [x["id"] for x in out["results"]] == [2]


def test_search_empty_collection_is_no_results_not_everything(env):
    make(ids=None)
    r, out = gantry_json("search", "glacier", "--fts", "--collection", "c")
    assert r.exit_code == 2 and out["results"] == []


def test_search_empty_collection_on_the_default_path(env, monkeypatch):
    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", lambda model, q: b"\0" * 3072)
    make(ids=None)
    r, out = gantry_json("search", "glacier", "--collection", "c")
    assert r.exit_code == 2 and out["results"] == []


def test_search_unknown_collection_is_usage_error(env):
    make(ids=None)
    r, out = gantry_json("search", "glacier", "--fts", "--collection", "nope")
    assert r.exit_code == 64 and "nope" in out["error"]


def test_search_warns_about_missing_members_on_stderr(env):
    conn, _ = env
    make()
    conn.execute("DELETE FROM papers WHERE id = 2")
    conn.commit()
    r = run("search", "glacier", "--fts", "--collection", "c", "--ids-only")
    assert r.stdout == "1\n"
    assert "1 missing" in r.stderr


def test_help_documents_the_group(env):
    out = run("collection", "--help").stdout
    for word in ("create", "add", "remove", "show", "list", "rename", "delete", "undo",
                 "history"):
        assert word in out
    assert "--collection" in run("search", "--help").stdout


def test_pipeline_reports_the_refresh_for_several_files(env, monkeypatch):
    make()
    stats = {"ingested": 0, "processed": 0, "enriched": 0, "embedded": 0, "errors": 0,
             "layout_titled": 0, "new_files": [], "ids": [],
             "collections": {"checked": 2, "updated": 0}}
    monkeypatch.setattr("pdf_gantry.pipeline.run_pipeline", lambda *a, **k: dict(stats))
    r = run("pipeline", "one.pdf", "two.pdf", "--json")
    assert json.loads(r.stdout)["collections"]["checked"] == 2
