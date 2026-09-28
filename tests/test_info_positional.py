"""`gantry info` accepts positional identifiers, not only --ids.

Agents tried `info <id>` six times in transcripts; each was a usage error.
Positional identifiers may be paper IDs (space- or comma-separated),
filenames, or citekeys (optionally with a leading @) -- the same forms
`gantry read` resolves.
"""

import json

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.utils import resolve_identifier, resolve_identifiers


def _insert(conn, filename, citekey=None, title=None):
    cur = conn.execute(
        "INSERT INTO papers "
        "(path, filename, file_hash, file_size, file_modified, indexed_at, updated_at,"
        " citekey, title) "
        "VALUES (?, ?, 'h', 1, '2024', '2024', '2024', ?, ?)",
        (f"/papers/{filename}", filename, citekey, title),
    )
    return cur.lastrowid


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    a = _insert(conn, "alpha.pdf", citekey="smith2020", title="Alpha")
    b = _insert(conn, "beta.pdf", citekey="Jones2021Beta", title="Beta")
    c = _insert(conn, "gamma.pdf", title="Gamma")
    conn.commit()
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg_file)
    yield conn, (a, b, c)
    conn.close()


# --- resolver ---------------------------------------------------------------

def test_resolve_identifier_by_id(seeded):
    conn, (a, _, _) = seeded
    assert resolve_identifier(conn, str(a)) == a


def test_resolve_identifier_by_filename(seeded):
    conn, (_, b, _) = seeded
    assert resolve_identifier(conn, "beta.pdf") == b


def test_resolve_identifier_by_citekey_with_or_without_at(seeded):
    conn, (a, b, _) = seeded
    assert resolve_identifier(conn, "smith2020") == a
    assert resolve_identifier(conn, "@smith2020") == a
    assert resolve_identifier(conn, "jones2021beta") == b  # case-insensitive fallback


def test_resolve_identifier_unknown(seeded):
    conn, _ = seeded
    assert resolve_identifier(conn, "9999") is None
    assert resolve_identifier(conn, "nope.pdf") is None


def test_resolve_identifiers_splits_commas_and_reports_missing(seeded):
    conn, (a, b, c) = seeded
    found, missing = resolve_identifiers(conn, [f"{a},{b}", "gamma.pdf", "999", "ghost"])
    assert found == [a, b, c]
    assert missing == [999, "ghost"]


# --- CLI --------------------------------------------------------------------

def test_info_positional_single_id(seeded):
    _, (a, _, _) = seeded
    result = CliRunner().invoke(cli, ["info", str(a), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [p["id"] for p in payload["papers"]] == [a]


def test_info_positional_multiple_and_comma(seeded):
    _, (a, b, c) = seeded
    result = CliRunner().invoke(cli, ["info", str(a), f"{b},{c}", "--json"])
    assert result.exit_code == 0, result.output
    ids = sorted(p["id"] for p in json.loads(result.output)["papers"])
    assert ids == sorted([a, b, c])


def test_info_positional_citekey_and_filename(seeded):
    _, (a, b, _) = seeded
    result = CliRunner().invoke(cli, ["info", "@smith2020", "beta.pdf", "--json"])
    assert result.exit_code == 0, result.output
    ids = sorted(p["id"] for p in json.loads(result.output)["papers"])
    assert ids == sorted([a, b])


def test_info_positional_combines_with_ids(seeded):
    _, (a, b, _) = seeded
    result = CliRunner().invoke(cli, ["info", str(a), "--ids", str(b), "--json"])
    assert result.exit_code == 0, result.output
    ids = sorted(p["id"] for p in json.loads(result.output)["papers"])
    assert ids == sorted([a, b])


def test_info_positional_unresolved_reported_as_partial(seeded):
    _, (a, _, _) = seeded
    result = CliRunner().invoke(cli, ["info", str(a), "ghost", "--json"])
    assert result.exit_code == 3
    assert json.loads(result.output)["not_found"] == ["ghost"]


def test_info_no_identifiers_is_usage_error(seeded):
    result = CliRunner().invoke(cli, ["info", "--json"])
    assert result.exit_code == 64
    assert "error" in json.loads(result.stdout)
