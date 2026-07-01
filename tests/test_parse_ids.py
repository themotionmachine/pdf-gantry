"""Pin the unified parse_ids behavior across all commands that accept comma-separated IDs.

Before this refactor, the same parsing logic existed in 5 places in cli.py with drift:

  - ocr --ids      (line ~485): [int(x.strip()) for x in ids.split(",")]
  - retry --ids    (line ~1302): [int(x.strip()) for x in ids.split(",")]
  - info --ids     (line ~1575): [int(x.strip()) for x in ids.split(",")]
  - search --restrict-to-ids  (line ~605): ... if x.strip()   <- guard present
  - semantic --restrict-to-ids (line ~909): ... if x.strip()  <- guard present

Three lacked the `if x.strip()` guard, so a trailing comma ("1,2,") raised ValueError
and returned an "Invalid --ids" error instead of working. Two had the guard. Classic drift.

All five now call parse_ids() from utils, which always strips blanks.
"""

import json

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.utils import parse_ids

# ---------------------------------------------------------------------------
# Unit tests for the helper itself
# ---------------------------------------------------------------------------


def test_parse_ids_basic():
    """Happy path: clean comma-separated integers."""
    assert parse_ids("1,2,3") == [1, 2, 3]


def test_parse_ids_strips_whitespace():
    """Spaces around commas and values are ignored."""
    assert parse_ids(" 1 , 2 , 3 ") == [1, 2, 3]


def test_parse_ids_trailing_comma():
    """Trailing comma must not crash — previously broken in the --ids variants."""
    assert parse_ids("1,2,3,") == [1, 2, 3]


def test_parse_ids_leading_comma():
    """Leading comma produces a blank segment that is dropped."""
    assert parse_ids(",1,2") == [1, 2]


def test_parse_ids_single():
    """Single ID works."""
    assert parse_ids("42") == [42]


def test_parse_ids_invalid_raises_value_error():
    """Non-integer content raises ValueError."""
    with pytest.raises(ValueError):
        parse_ids("1,foo,3")


def test_parse_ids_empty_string_returns_empty():
    """Pure empty string (or all-blank) yields an empty list, not a crash."""
    assert parse_ids("") == []
    assert parse_ids(",") == []


# ---------------------------------------------------------------------------
# CLI regression: trailing comma must NOT trigger "Invalid --ids" in the
# three commands that previously lacked the guard.
# ---------------------------------------------------------------------------


def _seed_db(tmp_path):
    """Insert one paper; return its ID."""
    from pdf_gantry.db import get_connection

    conn = get_connection(str(tmp_path / "index.db"))
    cur = conn.execute(
        "INSERT INTO papers "
        "(path, filename, file_hash, file_size, file_modified, indexed_at, updated_at) "
        "VALUES (?, ?, 'hash', 100, '2024-01-01', '2024-01-01', '2024-01-01')",
        ("/papers/test.pdf", "test.pdf"),
    )
    paper_id = cur.lastrowid
    conn.commit()
    conn.close()
    return paper_id


@pytest.fixture
def ids_env(tmp_path, monkeypatch):
    """Minimal DB + config wired for CLI tests of ID-parsing paths."""
    paper_id = _seed_db(tmp_path)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg_file)
    return paper_id


@pytest.mark.parametrize("trailing", ["", ","])
def test_info_ids_trailing_comma_not_a_parse_error(ids_env, trailing):
    """info --ids must accept a trailing comma without 'Invalid --ids' error.

    This was broken before the refactor: the inline list-comp lacked `if x.strip()`,
    so "42," raised ValueError and the command returned an error response.
    """
    from pdf_gantry.cli import cli

    paper_id = ids_env
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--ids", f"{paper_id}{trailing}", "--json"])
    assert "Invalid --ids" not in result.output
    payload = json.loads(result.output)
    assert "papers" in payload


@pytest.mark.parametrize("trailing", ["", ","])
def test_search_restrict_to_ids_trailing_comma_not_a_parse_error(ids_env, trailing):
    """search --restrict-to-ids already handled trailing commas; regression guard."""
    from pdf_gantry.cli import cli

    paper_id = ids_env
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["search", "anything", "--fts", "--restrict-to-ids",
         f"{paper_id}{trailing}", "--json"],
    )
    # The parse must succeed; the query may return 0 results, that's fine.
    assert "Invalid --restrict-to-ids" not in result.output
    payload = json.loads(result.output)
    assert "error" not in payload or "Invalid" not in payload.get("error", "")
