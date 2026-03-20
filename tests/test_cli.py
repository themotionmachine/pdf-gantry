"""Basic CLI smoke tests."""

from click.testing import CliRunner

from pdf_gantry.cli import cli


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Agent-friendly" in result.output


def test_config_show() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "papers_dir" in result.output
