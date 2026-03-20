"""Basic CLI smoke tests."""

from click.testing import CliRunner

from pdf_gantry.cli import cli


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Agent-friendly" in result.output


def test_commands_exist() -> None:
    """Verify all expected subcommands are registered on the CLI group."""
    command_names = set(cli.commands.keys())  # type: ignore[attr-defined]
    expected = {"search", "semantic", "status", "process", "queue", "config", "vault", "ingest"}
    assert expected.issubset(command_names)


def test_config_shows_papers_dir() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["config"])
    assert result.exit_code == 0
    assert "papers_dir" in result.output


def test_vault_check_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["vault", "--help"])
    assert result.exit_code == 0
    assert "check" in result.output
