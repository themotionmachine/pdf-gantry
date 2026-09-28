"""Usage errors must not masquerade as "no results".

Click's default exit code for a usage error is 2, which collided with
gantry's EXIT_NO_RESULTS=2. Over Quern's SSH shim a mistyped flag then
read as "paper not in library". Usage errors now exit 64 (EX_USAGE), and
when ``--json`` appears anywhere in argv they emit a JSON error object on
stdout.
"""

import json

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import EXIT_USAGE, cli
from pdf_gantry.db import get_connection


@pytest.fixture
def env(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    conn.close()
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg_file)


def test_exit_usage_is_64():
    assert EXIT_USAGE == 64


def test_unknown_option_exits_64(env):
    result = CliRunner().invoke(cli, ["search", "x", "--bogus"])
    assert result.exit_code == 64
    assert "--bogus" in result.output


def test_unknown_option_json_flag_on_subcommand(env):
    result = CliRunner().invoke(cli, ["search", "x", "--bogus", "--json"])
    assert result.exit_code == 64
    payload = json.loads(result.stdout)
    assert "--bogus" in payload["error"]
    assert "Usage:" in payload["usage"]


def test_unknown_option_json_flag_on_group(env):
    result = CliRunner().invoke(cli, ["--json", "info", "--bogus"])
    assert result.exit_code == 64
    payload = json.loads(result.stdout)
    assert "error" in payload and "usage" in payload


def test_bad_choice_exits_64(env):
    result = CliRunner().invoke(cli, ["queue", "--is", "needs_ocr", "--json"])
    assert result.exit_code == 64
    payload = json.loads(result.stdout)
    assert "needs_ocr" in payload["error"]


def test_unexpected_extra_argument_exits_64(env):
    result = CliRunner().invoke(cli, ["pipeline", "Nonexistent.pdf"])
    assert result.exit_code == 64


def test_unknown_command_exits_64(env):
    result = CliRunner().invoke(cli, ["serch", "x", "--json"])
    assert result.exit_code == 64
    payload = json.loads(result.stdout)
    assert "serch" in payload["error"]


def test_nested_group_usage_error_exits_64(env):
    result = CliRunner().invoke(cli, ["config", "set", "only_key"])
    assert result.exit_code == 64


def test_plain_usage_error_goes_to_stderr_not_json(env):
    result = CliRunner().invoke(cli, ["search", "x", "--bogus"])
    assert result.stdout == ""
    assert "No such option" in result.stderr


def test_help_still_exits_zero(env):
    result = CliRunner().invoke(cli, ["search", "--help"])
    assert result.exit_code == 0


def test_real_no_results_still_exits_2(env):
    result = CliRunner().invoke(cli, ["find", "nothing_matches_this", "--json"])
    assert result.exit_code == 2
