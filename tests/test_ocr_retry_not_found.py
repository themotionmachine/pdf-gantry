"""`gantry ocr --ids` and `gantry retry --ids` must report unresolved IDs.

`gantry info --ids` already closed this gap (missing_ids(), tested in
test_info_not_found.py): when a requested ID doesn't resolve to a row, info
names it in `not_found` and downgrades its exit code instead of silently
returning fewer results than asked for. `ocr --ids` and `retry --ids` accept
the exact same "fetch/act on exactly these papers" style of --ids argument
(unlike `search --restrict-to-ids` / `semantic --restrict-to-ids`, which
scope a query rather than name a fetch target), and had the identical bug:
paper_ids went straight into the SQL `WHERE id IN (...)` inside
process_ocr_documents / process_documents with no existence check anywhere,
so a stale or mistyped ID was dropped with zero signal.

This file pins the fix for both commands, reusing the same missing_ids()
primitive and the same not_found/exit-code convention as info.

ocr's dry-run path is exercised here (not the real OCR path) because it
never imports the optional `surya` dependency -- the not_found computation
happens in cli.py before process_ocr_documents is ever called, so dry-run
is a faithful, dependency-free way to pin the behavior.
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
def ids_env(tmp_path, monkeypatch):
    """A configured DB with two known papers, wired for CLI --ids tests."""
    ids = _seed_db(tmp_path, n=2)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({
        "index_dir": str(tmp_path),
        "papers_dir": str(tmp_path),
    }))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg_file)
    return ids


# --- ocr --ids --dry-run (surya-free) ---


def test_ocr_dry_run_reports_not_found_ids(ids_env):
    """A bogus ID mixed into ocr --ids --dry-run is named, not silently dropped."""
    id_a, _ = ids_env
    bogus = 99999
    result = CliRunner().invoke(
        cli, ["ocr", "--ids", f"{id_a},{bogus}", "--dry-run", "--json"]
    )
    payload = json.loads(result.output)
    assert payload["not_found"] == [bogus]
    assert result.exit_code == EXIT_PARTIAL


def test_ocr_dry_run_all_found_reports_empty_not_found(ids_env):
    """When every requested ID resolves, not_found is present but empty, exit 0."""
    id_a, id_b = ids_env
    result = CliRunner().invoke(
        cli, ["ocr", "--ids", f"{id_a},{id_b}", "--dry-run", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["not_found"] == []


def test_ocr_dry_run_plain_output_names_the_id(ids_env):
    """Plain (non-JSON) dry-run output must also surface the missing ID."""
    id_a, _ = ids_env
    bogus = 99999
    result = CliRunner().invoke(cli, ["ocr", "--ids", f"{id_a},{bogus}", "--dry-run"])
    assert str(bogus) in result.output
    assert result.exit_code == EXIT_PARTIAL


def test_ocr_filter_based_selection_unaffected(ids_env, monkeypatch):
    """Without --ids, there's no requested set to diff against: no not_found key noise."""
    # Nothing needs OCR in this DB, so this should be the ordinary empty-queue path.
    result = CliRunner().invoke(cli, ["ocr", "--dry-run", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "not_found" not in payload


# --- retry --ids ---


def test_retry_reports_not_found_and_exits_partial(ids_env):
    """A bogus ID mixed into retry --ids is named in not_found, exit code partial."""
    id_a, _ = ids_env
    bogus = 99999
    result = CliRunner().invoke(cli, ["retry", "--ids", f"{id_a},{bogus}", "--json"])
    payload = json.loads(result.output)
    assert payload["not_found"] == [bogus]
    assert result.exit_code == EXIT_PARTIAL


def test_retry_all_found_reports_empty_not_found(ids_env):
    """When every requested ID resolves, retry's not_found is empty."""
    id_a, id_b = ids_env
    result = CliRunner().invoke(cli, ["retry", "--ids", f"{id_a},{id_b}", "--json"])
    payload = json.loads(result.output)
    assert payload["not_found"] == []
