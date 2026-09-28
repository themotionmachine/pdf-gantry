"""Tests for `gantry grep`: literal (quote) search over chunk text, with pages."""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection


def _add_paper(conn, filename, raw_text=None):
    cur = conn.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at, has_text) VALUES (?, ?, 'h', 1, 't', 't', 't', 1)",
        (filename, filename),
    )
    pid = cur.lastrowid
    if raw_text is not None:
        conn.execute(
            "INSERT INTO paper_text (paper_id, raw_text, markdown, text_length, "
            "markdown_length) VALUES (?, ?, ?, ?, ?)",
            (pid, raw_text, raw_text, len(raw_text), len(raw_text)),
        )
    return pid


def _add_chunk(conn, doc_id, index, text, pages=(None, None)):
    cur = conn.execute(
        "INSERT INTO chunks (doc_id, chunk_index, text, page_start, page_end) "
        "VALUES (?, ?, ?, ?, ?)",
        (doc_id, index, text, pages[0], pages[1]),
    )
    return cur.lastrowid


@pytest.fixture
def db(tmp_path):
    conn = get_connection(str(tmp_path / "index.db"))
    a = _add_paper(conn, "alpha.pdf")
    _add_chunk(conn, a, 0, "Opening remarks on method and scope.", (1, 1))
    a1 = _add_chunk(conn, a, 1,
                    "Citizens assemblies select members by lottery so the room "
                    "resembles the population.", (2, 3))
    b = _add_paper(conn, "beta.pdf")
    b0 = _add_chunk(conn, b, 0,
                    "Recommendation systems optimise engage-\nment, which can "
                    "amplify **outrage** and “sensational” claims. The ﬁrst\n\n"
                    "finding   holds.", (7, 7))
    c = _add_paper(conn, "gamma.pdf",
                   raw_text="Unchunked paper: the lottery selects members at random.")
    conn.commit()
    return conn, {"a": a, "b": b, "c": c, "a1": a1, "b0": b0}


def test_exact_hit_fields(db):
    from pdf_gantry.grep import grep

    conn, ids = db
    res = grep(conn, "select members by lottery")
    assert res["count"] == 1
    h = res["hits"][0]
    assert h["doc_id"] == ids["a"]
    assert h["chunk_id"] == ids["a1"]
    assert h["chunk_index"] == 1
    assert (h["page_start"], h["page_end"]) == (2, 3)
    assert h["match"] == "exact"
    assert h["source"] == "chunks"
    text = conn.execute("SELECT text FROM chunks WHERE chunk_id=?", (ids["a1"],)).fetchone()[0]
    assert text[h["offset"]:h["offset"] + h["length"]] == "select members by lottery"
    assert h["matched_text"] == "select members by lottery"
    assert "select members by lottery" in h["context"]


def test_case_sensitive_by_default(db):
    from pdf_gantry.grep import grep

    conn, _ = db
    assert grep(conn, "SELECT MEMBERS BY LOTTERY")["count"] == 0
    res = grep(conn, "SELECT MEMBERS BY LOTTERY", ignore_case=True)
    assert res["count"] == 1
    assert res["hits"][0]["match"] == "exact"


def test_normalised_match_across_hyphenation_markdown_and_quotes(db):
    """Line-break hyphen, markdown emphasis and curly quotes are tolerated."""
    from pdf_gantry.grep import grep

    conn, ids = db
    q = 'optimise engagement, which can amplify outrage and "sensational" claims'
    res = grep(conn, q)
    assert res["count"] == 1
    h = res["hits"][0]
    assert h["match"] == "normalized"
    assert h["chunk_id"] == ids["b0"]
    text = conn.execute("SELECT text FROM chunks WHERE chunk_id=?", (ids["b0"],)).fetchone()[0]
    assert text[h["offset"]:h["offset"] + h["length"]] == h["matched_text"]
    assert h["matched_text"].startswith("optimise engage-\nment")
    assert h["matched_text"].endswith("” claims")


def test_normalised_match_ligature_and_whitespace(db):
    from pdf_gantry.grep import grep

    conn, _ = db
    res = grep(conn, "The first finding holds.")
    assert res["count"] == 1
    assert res["hits"][0]["match"] == "normalized"
    assert res["hits"][0]["page_start"] == 7


def test_ids_scope(db):
    from pdf_gantry.grep import grep

    conn, ids = db
    assert grep(conn, "lottery", doc_ids=[ids["b"]])["count"] == 0
    res = grep(conn, "lottery", doc_ids=[ids["a"], ids["b"]])
    assert {h["doc_id"] for h in res["hits"]} == {ids["a"]}


def test_raw_text_fallback_for_unchunked_paper(db):
    from pdf_gantry.grep import grep

    conn, ids = db
    res = grep(conn, "the lottery selects members")
    assert res["count"] == 1
    h = res["hits"][0]
    assert h["doc_id"] == ids["c"]
    assert h["source"] == "raw_text"
    assert h["chunk_id"] is None and h["page_start"] is None


def test_no_raw_text_fallback_when_paper_has_chunks(tmp_path):
    from pdf_gantry.grep import grep

    conn = get_connection(str(tmp_path / "i.db"))
    p = _add_paper(conn, "x.pdf", raw_text="only in raw text: zebra crossing")
    _add_chunk(conn, p, 0, "chunk text without the phrase")
    conn.commit()
    assert grep(conn, "zebra crossing")["count"] == 0


def test_limit_and_truncated(tmp_path):
    from pdf_gantry.grep import grep

    conn = get_connection(str(tmp_path / "i.db"))
    p = _add_paper(conn, "x.pdf")
    for i in range(5):
        _add_chunk(conn, p, i, f"part {i}: needle here")
    conn.commit()
    res = grep(conn, "needle", limit=3)
    assert res["count"] == 3
    assert res["truncated"] is True
    assert grep(conn, "needle", limit=10)["truncated"] is False


def test_multiple_hits_in_one_chunk(tmp_path):
    from pdf_gantry.grep import grep

    conn = get_connection(str(tmp_path / "i.db"))
    p = _add_paper(conn, "x.pdf")
    _add_chunk(conn, p, 0, "echo one, echo two")
    conn.commit()
    res = grep(conn, "echo")
    assert [h["offset"] for h in res["hits"]] == [0, 10]


def test_overlap_duplicate_is_collapsed(tmp_path):
    """Consecutive chunks share ~200 chars of overlap; a quote inside the
    overlap is reported once (in the earlier chunk), not twice."""
    from pdf_gantry.grep import grep

    conn = get_connection(str(tmp_path / "i.db"))
    p = _add_paper(conn, "x.pdf")
    tail = "the quoted passage sits in the overlap region."
    c0 = _add_chunk(conn, p, 0, "Earlier material. " + tail, (1, 1))
    _add_chunk(conn, p, 1, tail + "\n\nLater material continues here.", (1, 2))
    conn.commit()
    res = grep(conn, "quoted passage sits")
    assert res["count"] == 1
    assert res["hits"][0]["chunk_id"] == c0


def test_grep_never_opens_pdfs(db, monkeypatch):
    import pymupdf

    def boom(*a, **k):
        raise AssertionError("grep must not open PDFs")

    monkeypatch.setattr(pymupdf, "open", boom)
    from pdf_gantry.grep import grep

    conn, _ = db
    assert grep(conn, "lottery")["count"] >= 1


def test_normalize_with_map_offsets():
    from pdf_gantry.grep import normalize_with_map

    norm, idx = normalize_with_map("A  ﬁne-\n tuned **bold**")
    assert norm == "A finetuned bold"
    assert len(idx) == len(norm)
    # every normalised char maps back to a source position
    assert all(0 <= i < len("A  ﬁne-\n tuned **bold**") for i in idx)


# --- parse_ids composition -------------------------------------------------


def test_parse_ids_accepts_ids_only_output():
    """`search --ids-only` prints one id per line; --ids must take that as is."""
    from pdf_gantry.utils import parse_ids

    assert parse_ids("12\n7\n31\n") == [12, 7, 31]
    assert parse_ids("1, 2\n3") == [1, 2, 3]


# --- CLI -------------------------------------------------------------------


@pytest.fixture
def cli_env(db, tmp_path, monkeypatch):
    conn, ids = db
    conn.close()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))
    return ids


def test_cli_grep_json(cli_env):
    ids = cli_env
    r = CliRunner().invoke(cli, ["grep", "select members by lottery", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["query"] == "select members by lottery"
    assert data["count"] == 1
    h = data["hits"][0]
    for key in ("doc_id", "chunk_id", "chunk_index", "page_start", "page_end",
                "offset", "context", "match", "source"):
        assert key in h
    assert h["doc_id"] == ids["a"]


def test_cli_grep_no_hits_exit_2(cli_env):
    r = CliRunner().invoke(cli, ["grep", "no such phrase anywhere", "--json"])
    assert r.exit_code == 2
    assert json.loads(r.output)["count"] == 0


def test_cli_grep_ids_from_ids_only_output(cli_env):
    ids = cli_env
    piped = f"{ids['b']}\n{ids['a']}\n"
    r = CliRunner().invoke(cli, ["grep", "lottery", "--ids", piped, "--json"])
    assert r.exit_code == 0, r.output
    assert {h["doc_id"] for h in json.loads(r.output)["hits"]} == {ids["a"]}


def test_cli_grep_ids_from_stdin(cli_env):
    ids = cli_env
    r = CliRunner().invoke(cli, ["grep", "lottery", "--ids", "-", "--json"],
                           input=f"{ids['a']}\n")
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["count"] == 1


def test_cli_grep_unknown_ids_partial(cli_env):
    ids = cli_env
    r = CliRunner().invoke(cli, ["grep", "lottery", "--ids", f"{ids['a']},9999", "--json"])
    assert r.exit_code == 3
    assert json.loads(r.output)["not_found"] == [9999]


def test_cli_grep_ignore_case_and_limit(cli_env):
    r = CliRunner().invoke(cli, ["grep", "LOTTERY", "-i", "--limit", "1", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["count"] == 1 and data["truncated"] is True


def test_cli_grep_text_output(cli_env):
    r = CliRunner().invoke(cli, ["grep", "select members by lottery"])
    assert r.exit_code == 0, r.output
    assert "alpha.pdf" in r.output
    assert "p.2-3" in r.output
