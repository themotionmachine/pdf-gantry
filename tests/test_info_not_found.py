"""`gantry info --ids` must report which requested IDs did not resolve.

The composable-retrieval story in CLAUDE.md is: `search --ids-only` pipes
straight into `info --ids`. In a real ~2000-PDF corpus, that ID set goes
stale — a paper gets pruned, a caller typos a comma, an ID comes from a
snapshot taken a `gantry prune` ago. Before this test, `info` silently
returned fewer papers than requested with no signal at all: same JSON
shape, same exit code 0, just a shorter list. This file pins the fix:
`info` now names exactly which requested IDs it couldn't resolve, in both
`--json` and plain output, and reports a partial-failure exit code when
some (but not all) requested IDs are missing.
"""

import json

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection

EXIT_NO_RESULTS = 2
EXIT_PARTIAL = 3


def _seed_db(tmp_path, n=1):
    """Insert ``n`` papers; return their IDs in insertion order."""
    conn = get_connection(str(tmp_path / "index.db"))
    ids = []
    for i in range(n):
        cur = conn.execute(
            "INSERT INTO papers "
            "(path, filename, file_hash, file_size, file_modified, indexed_at, updated_at) "
            "VALUES (?, ?, 'hash', 100, '2024-01-01', '2024-01-01', '2024-01-01')",
            (f"/papers/test{i}.pdf", f"test{i}.pdf"),
        )
        ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    return ids


@pytest.fixture
def info_env(tmp_path, monkeypatch):
    """A configured DB with two known papers, for CLI tests of `info --ids`."""
    ids = _seed_db(tmp_path, n=2)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg_file)
    return ids


def test_info_all_ids_found_reports_empty_not_found(info_env):
    """When every requested ID resolves, not_found is present but empty."""
    id_a, id_b = info_env
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--ids", f"{id_a},{id_b}", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["not_found"] == []
    assert payload["count"] == 2


def test_info_partial_miss_reports_missing_id_and_exit_partial(info_env):
    """One real ID + one bogus ID: the bogus one is named, exit code is partial."""
    id_a, _ = info_env
    bogus = 99999
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--ids", f"{id_a},{bogus}", "--json"])
    payload = json.loads(result.output)
    assert payload["not_found"] == [bogus]
    assert payload["count"] == 1
    assert result.exit_code == EXIT_PARTIAL


def test_info_partial_miss_plain_output_names_the_id(info_env):
    """Plain (non-JSON) output must also surface the missing ID, not just a shorter list."""
    id_a, _ = info_env
    bogus = 99999
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--ids", f"{id_a},{bogus}"])
    assert str(bogus) in result.output
    assert result.exit_code == EXIT_PARTIAL


def test_info_all_missing_still_exits_no_results_but_names_ids(info_env):
    """All-missing case keeps exit code 2 (no results), but now names the IDs too."""
    bogus1, bogus2 = 88888, 99999
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--ids", f"{bogus1},{bogus2}", "--json"])
    assert result.exit_code == EXIT_NO_RESULTS
    payload = json.loads(result.output)
    assert payload["not_found"] == [bogus1, bogus2]


def test_info_duplicate_requested_id_reported_once(info_env):
    """A bogus ID repeated in --ids is only reported once in not_found."""
    id_a, _ = info_env
    bogus = 99999
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--ids", f"{id_a},{bogus},{bogus}", "--json"])
    payload = json.loads(result.output)
    assert payload["not_found"] == [bogus]
