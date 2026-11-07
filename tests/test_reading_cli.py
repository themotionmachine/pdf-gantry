"""`gantry inbox ...`: the reading inbox from the command line."""

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


def add(ids="1,2", *extra):
    r, out = gantry_json("inbox", "add", "--ids", ids, "--by", "t", *extra)
    assert r.exit_code == 0, r.output
    return out


def test_add_list_due_round_trip(env):
    out = add("1,2", "--why", "chapter 3", "--source", "agent")
    assert out["added"] == [1, 2] and out["batch_id"]
    r, listed = gantry_json("inbox", "list")
    assert r.exit_code == 0 and [i["paper_id"] for i in listed["items"]] == [1, 2]
    item = listed["items"][0]
    assert (item["why"], item["source"], item["by"], item["title"]) == (
        "chapter 3", "agent", "t", "Alpha")
    r, due = gantry_json("inbox", "due")
    assert r.exit_code == 0 and due["count"] == 2
    assert run("inbox", "due", "--ids-only").stdout == "1\n2\n"


def test_sidecar_is_private(env):
    add()
    assert stat.S_IMODE(os.stat(env[1] / "reading.db").st_mode) == 0o600


@pytest.mark.parametrize("args", [
    ("add", "--ids", "1"),
    ("review", "1", "good"),
    ("undo", "abc"),
])
def test_by_is_required_on_writes(env, args):
    assert run("inbox", *args).exit_code == 64


def test_add_from_stdin_takes_ids_only_output(env):
    ids = run("search", "glacier", "--fts", "--ids-only").stdout
    r, out = gantry_json("inbox", "add", "--ids", "-", "--by", "t", stdin=ids)
    assert r.exit_code == 0 and sorted(out["added"]) == [1, 2, 3]


def test_add_again_is_success_and_reported(env):
    add("1")
    r, out = gantry_json("inbox", "add", "--ids", "1", "--by", "t")
    assert r.exit_code == 0 and out["already_present"] == [1] and out["added"] == []


def test_add_exit_codes(env):
    r, out = gantry_json("inbox", "add", "--ids", "1,99", "--by", "t")
    assert r.exit_code == 3 and out["not_found"] == [99]
    assert run("inbox", "add", "--ids", "98", "--by", "t").exit_code == 2
    assert run("inbox", "add", "--ids", "-", "--by", "t", stdin="").exit_code == 2
    assert run("inbox", "add", "--ids", "1,x", "--by", "t").exit_code == 1


def test_add_a_passage(env):
    out = add("1", "--page", "4", "--quote", "the ice", "--note", "check this")
    r, listed = gantry_json("inbox", "list")
    assert listed["items"][0]["passage"] == {"page": 4, "quote": "the ice",
                                             "note": "check this"}
    assert out["items"][0]["item_id"] == listed["items"][0]["item_id"]


@pytest.mark.parametrize("args", [
    ("--quote", "q"),
    ("--note", "n"),
    ("--page", "0"),
    ("--source", "vault"),
])
def test_bad_add_options_are_usage_errors(env, args):
    assert run("inbox", "add", "--ids", "1", "--by", "t", *args).exit_code == 64


def test_review_and_undo(env):
    item = add("1")["items"][0]["item_id"]
    r, out = gantry_json("inbox", "review", str(item), "good", "--by", "ryan")
    assert r.exit_code == 0 and out["interval_days"] == 1.0 and out["paper_id"] == 1
    assert out["status"] == "scheduled"
    assert run("inbox", "due").exit_code == 2
    r, undone = gantry_json("inbox", "undo", out["batch_id"], "--by", "ryan")
    assert r.exit_code == 0 and undone["op"] == "review" and undone["reversed"] == 1
    assert gantry_json("inbox", "due")[1]["count"] == 1


def test_review_errors(env):
    item = add("1")["items"][0]["item_id"]
    r, out = gantry_json("inbox", "review", "999", "good", "--by", "t")
    assert r.exit_code == 64 and "999" in out["error"]
    assert run("inbox", "review", str(item), "meh", "--by", "t").exit_code == 64
    assert run("inbox", "review", str(item), "done", "--by", "t").exit_code == 0
    r, out = gantry_json("inbox", "review", str(item), "good", "--by", "t")
    assert r.exit_code == 1 and out["code"] == "item_retired"


def test_undo_errors_carry_stable_codes(env):
    batch = add("1")["batch_id"]
    assert run("inbox", "undo", batch, "--by", "t").exit_code == 0
    r, out = gantry_json("inbox", "undo", batch, "--by", "t")
    assert r.exit_code == 1 and out["code"] == "already_undone"
    r, out = gantry_json("inbox", "undo", "nope", "--by", "t")
    assert r.exit_code == 1 and out["code"] == "unknown_batch"


def test_list_statuses_and_history(env):
    item = add("1,2")["items"][0]["item_id"]
    run("inbox", "review", str(item), "done", "--by", "t")
    assert [i["paper_id"] for i in gantry_json("inbox", "list")[1]["items"]] == [2]
    assert [i["paper_id"] for i in gantry_json(
        "inbox", "list", "--status", "done")[1]["items"]] == [1]
    assert run("inbox", "list", "--status", "nope").exit_code == 64
    r, h = gantry_json("inbox", "history", "--limit", "5")
    assert r.exit_code == 0 and [b["op"] for b in h["history"]] == ["review", "enqueue"]
    assert h["history"][0]["verdict"] == "done"


def test_missing_items_exit_3(env):
    conn, _ = env
    add("1,2")
    conn.execute("DELETE FROM papers WHERE id = 2")
    conn.commit()
    r, out = gantry_json("inbox", "due")
    assert r.exit_code == 3 and [i["resolution"] for i in out["items"]] == [
        "resolved", "missing"]
    r = run("inbox", "due", "--ids-only")
    assert r.stdout == "1\n" and "1 missing" in r.stderr


def test_reads_with_no_sidecar_never_create_it(env):
    _, tmp = env
    r, out = gantry_json("inbox", "list")
    assert r.exit_code == 2 and out["items"] == []
    assert run("inbox", "due").exit_code == 2
    r, out = gantry_json("inbox", "history")
    assert r.exit_code == 0 and out["history"] == []
    assert not (tmp / "reading.db").exists()


@pytest.mark.parametrize("args", [
    ("inbox", "list"), ("inbox", "due"), ("inbox", "due", "--ids-only"),
    ("inbox", "history"), ("inbox", "list", "--status", "all"),
])
def test_read_commands_do_not_write_the_sidecar(env, args):
    add()
    path = env[1] / "reading.db"
    before = path.read_bytes()
    run(*args)
    assert path.read_bytes() == before


def test_text_output(env):
    add("1")
    r = run("inbox", "due")
    assert r.exit_code == 0 and "Alpha" in r.stdout and "[1]" in r.stdout


def test_help_documents_the_group(env):
    out = run("inbox", "--help").stdout
    for word in ("add", "list", "due", "review", "history", "undo", "reading.db"):
        assert word in out
    assert "!!" in run("inbox", "review", "--help").stdout


def test_pipeline_reports_the_reading_refresh_for_several_files(env, monkeypatch):
    stats = {"ingested": 0, "processed": 0, "enriched": 0, "embedded": 0, "errors": 0,
             "layout_titled": 0, "new_files": [], "ids": [],
             "reading": {"checked": 2, "updated": 0}}
    monkeypatch.setattr("pdf_gantry.pipeline.run_pipeline", lambda *a, **k: dict(stats))
    r = run("pipeline", "one.pdf", "two.pdf", "--json")
    assert json.loads(r.stdout)["reading"]["checked"] == 2
