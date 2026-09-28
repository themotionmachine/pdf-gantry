"""Tests for `find`: filename lookup plus title/author/citekey/DOI/year (E3/L1)."""

import json

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.search import find_papers


def test_find_exact_match(populated_db):
    """Exact filename match returns the paper."""
    results = find_papers(populated_db, "test_climate.pdf")
    assert len(results) == 1
    assert results[0]["filename"] == "test_climate.pdf"


def test_find_partial_match(populated_db):
    """Partial filename fragment matches."""
    results = find_papers(populated_db, "climate")
    assert len(results) >= 1
    assert any("climate" in r["filename"].lower() for r in results)


def test_find_case_insensitive(populated_db):
    """Search is case-insensitive."""
    results = find_papers(populated_db, "CLIMATE")
    assert len(results) >= 1


def test_find_multiple_results(populated_db):
    """Fragment matching multiple files returns all."""
    results = find_papers(populated_db, ".pdf")
    assert len(results) == 2  # Both test PDFs


def test_find_no_match(populated_db):
    """No match returns empty list."""
    results = find_papers(populated_db, "nonexistent_xyz")
    assert results == []


def test_find_respects_limit(populated_db):
    """Limit parameter caps results."""
    results = find_papers(populated_db, ".pdf", limit=1)
    assert len(results) == 1


# --- find over metadata (E3/L1) ----------------------------------------------

def _add(conn, filename, title=None, authors=None, year=None, citekey=None, doi=None):
    cur = conn.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified,"
        " indexed_at, updated_at, title, authors, year, citekey, doi)"
        " VALUES (?, ?, 'h', 1, '2024', '2024', '2024', ?, ?, ?, ?, ?)",
        (f"/p/{filename}", filename, title, authors, year, citekey, doi),
    )
    return cur.lastrowid


@pytest.fixture
def meta_db(tmp_db):
    ids = {
        "mind": _add(tmp_db, "14614448211014355.pdf",
                     title="Mind games: A temporal sentiment analysis of the IRA",
                     authors='["Dan Hiaeshutter-Rice", "Brian Weeks"]', year=2021),
        "flew": _add(tmp_db, "flew_platform.pdf", title="Digital Platform Regulation",
                     authors='["Terry Flew", "Fiona Martin"]', year=2022,
                     citekey="flew2022digital", doi="10.1007/978-3-030-95220-4"),
        "zhang22": _add(tmp_db, "a1.pdf", title="Something about platforms",
                        authors='["Yini Zhang", "Chris Wells"]', year=2022),
        "zhang19": _add(tmp_db, "a2.pdf", title="Disinformation, performed",
                        authors='["Yiping Xia", "Yini Zhang"]', year=2019),
        "accent": _add(tmp_db, "a3.pdf", title="Populism in Europe",
                       authors='["Ond\\u0159ej Filipec"]', year=2019),
        "mention": _add(tmp_db, "mind_games_notes.pdf", title="Notes on games of the mind"),
    }
    tmp_db.commit()
    return tmp_db, ids


def test_find_matches_title(meta_db):
    conn, ids = meta_db
    results = find_papers(conn, "Mind games")
    assert results[0]["id"] == ids["mind"]
    assert "title" in results[0]["matched_fields"]


def test_find_title_match_ranks_before_filename_match(meta_db):
    conn, ids = meta_db
    results = find_papers(conn, "mind games")
    found = [r["id"] for r in results]
    assert found.index(ids["mind"]) < found.index(ids["mention"])


def test_find_multi_token_all_must_match(meta_db):
    conn, ids = meta_db
    found = {r["id"] for r in find_papers(conn, "Zhang 2022")}
    assert found == {ids["zhang22"]}


def test_find_author_ampersand_query(meta_db):
    conn, ids = meta_db
    results = find_papers(conn, "Flew & Martin")
    assert [r["id"] for r in results] == [ids["flew"]]
    assert "authors" in results[0]["matched_fields"]


def test_find_et_al_query(meta_db):
    conn, ids = meta_db
    found = [r["id"] for r in find_papers(conn, "Zhang et al. (2019)")]
    assert found == [ids["zhang19"]]


def test_find_citekey_and_doi(meta_db):
    conn, ids = meta_db
    r = find_papers(conn, "@flew2022digital")
    assert r[0]["id"] == ids["flew"] and "citekey" in r[0]["matched_fields"]
    r = find_papers(conn, "https://doi.org/10.1007/978-3-030-95220-4")
    assert r[0]["id"] == ids["flew"] and "doi" in r[0]["matched_fields"]


def test_find_accent_and_case_insensitive(meta_db):
    conn, ids = meta_db
    assert [r["id"] for r in find_papers(conn, "ondrej FILIPEC")] == [ids["accent"]]


def test_find_filename_still_works(meta_db):
    conn, ids = meta_db
    r = find_papers(conn, "14614448211014355")
    assert r[0]["id"] == ids["mind"] and r[0]["matched_fields"] == ["filename"]


def test_find_result_has_year(meta_db):
    conn, ids = meta_db
    assert find_papers(conn, "Flew")[0]["year"] == 2022


@pytest.fixture
def find_cli(tmp_path, monkeypatch):
    from pdf_gantry.db import get_connection
    conn = get_connection(str(tmp_path / "index.db"))
    a = _add(conn, "x.pdf", title="Mind games", authors='["A B"]', year=2021)
    b = _add(conn, "y.pdf", title="Mind the hype", authors='["C D"]', year=2020)
    conn.commit()
    conn.close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg)
    return a, b


def test_find_cli_json_reports_matched_fields(find_cli):
    a, _ = find_cli
    result = CliRunner().invoke(cli, ["find", "Mind games", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["results"][0]["id"] == a
    assert payload["results"][0]["matched_fields"] == ["title"]
    assert payload["fragment"] == "Mind games"


def test_find_cli_ids_only(find_cli):
    a, b = find_cli
    result = CliRunner().invoke(cli, ["find", "mind", "--ids-only"])
    assert result.exit_code == 0
    assert sorted(int(x) for x in result.output.split()) == sorted([a, b])


def test_find_cli_limit(find_cli):
    result = CliRunner().invoke(cli, ["find", "mind", "--ids-only", "--limit", "1"])
    assert len(result.output.split()) == 1


def test_find_cli_ids_only_no_results(find_cli):
    result = CliRunner().invoke(cli, ["find", "zzzz", "--ids-only"])
    assert result.exit_code == 2
    assert result.stdout == ""
