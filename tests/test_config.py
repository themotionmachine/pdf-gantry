"""Tests for config loading, saving, and validation."""

import os
from pathlib import Path
from unittest.mock import patch

import yaml
import pytest

from pdf_gantry.config import (
    Config, load_config, save_config, set_config_value, CONFIG_PATH, GANTRY_DIR,
)


def test_config_defaults():
    """Config with no file and no env vars has no papers_dir."""
    with patch("pdf_gantry.config.CONFIG_PATH", Path("/nonexistent/config.yaml")):
        cfg = load_config()
    assert cfg.papers_dir is None
    assert cfg.index_dir == GANTRY_DIR
    assert cfg.vault_dir is None
    assert cfg.embedding.dimensions == 768
    assert cfg.processing.workers == 4


def test_config_from_yaml(tmp_path):
    """Config loads values from YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "papers_dir": str(tmp_path / "my_papers"),
        "vault_dir": str(tmp_path / "my_vault"),
    }))

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        cfg = load_config()

    assert cfg.papers_dir == tmp_path / "my_papers"
    assert cfg.vault_dir == tmp_path / "my_vault"


def test_config_path_expansion(tmp_path):
    """Tilde paths are expanded."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "papers_dir": "~/my_papers",
    }))

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        cfg = load_config()

    assert str(cfg.papers_dir).startswith(str(Path.home()))
    assert "~" not in str(cfg.papers_dir)


def test_config_env_override(tmp_path):
    """Environment variables override config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "papers_dir": str(tmp_path / "from_file"),
    }))

    env_dir = str(tmp_path / "from_env")
    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch.dict(os.environ, {"GANTRY_PAPERS_DIR": env_dir}):
            cfg = load_config()

    assert cfg.papers_dir == Path(env_dir)


def test_config_save_and_reload(tmp_path):
    """Config round-trips through save/load."""
    config_file = tmp_path / "config.yaml"

    cfg = Config(
        papers_dir=tmp_path / "papers",
        vault_dir=tmp_path / "vault",
    )

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch("pdf_gantry.config.GANTRY_DIR", tmp_path):
            save_config(cfg)
            loaded = load_config()

    assert loaded.papers_dir == cfg.papers_dir
    assert loaded.vault_dir == cfg.vault_dir


def test_config_set_value(tmp_path):
    """set_config_value updates a single key."""
    config_file = tmp_path / "config.yaml"

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch("pdf_gantry.config.GANTRY_DIR", tmp_path):
            cfg = set_config_value("papers_dir", str(tmp_path / "new_papers"))

    assert cfg.papers_dir == tmp_path / "new_papers"


def test_config_set_unknown_key(tmp_path):
    """set_config_value raises for unknown keys."""
    config_file = tmp_path / "config.yaml"

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch("pdf_gantry.config.GANTRY_DIR", tmp_path):
            with pytest.raises(ValueError, match="Unknown config key"):
                set_config_value("nonexistent_key", "value")


def test_config_bib_path_default():
    """bib_path defaults to None."""
    with patch("pdf_gantry.config.CONFIG_PATH", Path("/nonexistent/config.yaml")):
        cfg = load_config()
    assert cfg.bib_path is None


def test_config_bib_path_from_yaml(tmp_path):
    """bib_path loads from YAML."""
    config_file = tmp_path / "config.yaml"
    bib = tmp_path / "library.bib"
    config_file.write_text(yaml.dump({"bib_path": str(bib)}))

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        cfg = load_config()

    assert cfg.bib_path == bib


def test_config_bib_path_env_override(tmp_path):
    """GANTRY_BIB_PATH env var overrides config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"bib_path": str(tmp_path / "from_file.bib")}))

    env_bib = str(tmp_path / "from_env.bib")
    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch.dict(os.environ, {"GANTRY_BIB_PATH": env_bib}):
            cfg = load_config()

    assert cfg.bib_path == Path(env_bib)


def test_config_bib_path_save_roundtrip(tmp_path):
    """bib_path round-trips through save/load."""
    config_file = tmp_path / "config.yaml"
    cfg = Config(bib_path=tmp_path / "library.bib")

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch("pdf_gantry.config.GANTRY_DIR", tmp_path):
            save_config(cfg)
            loaded = load_config()

    assert loaded.bib_path == cfg.bib_path


def test_config_set_bib_path(tmp_path):
    """set_config_value works for bib_path."""
    config_file = tmp_path / "config.yaml"

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch("pdf_gantry.config.GANTRY_DIR", tmp_path):
            cfg = set_config_value("bib_path", str(tmp_path / "refs.bib"))

    assert cfg.bib_path == tmp_path / "refs.bib"


# --- openalex_mailto (issue #20) ---


def test_config_openalex_mailto_default():
    """openalex_mailto defaults to None."""
    with patch("pdf_gantry.config.CONFIG_PATH", Path("/nonexistent/config.yaml")):
        cfg = load_config()
    assert cfg.openalex_mailto is None


def test_config_openalex_mailto_from_yaml(tmp_path):
    """openalex_mailto loads from YAML."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"openalex_mailto": "ryan@example.com"}))

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        cfg = load_config()

    assert cfg.openalex_mailto == "ryan@example.com"


def test_config_openalex_mailto_env_override(tmp_path):
    """GANTRY_OPENALEX_MAILTO env var overrides config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"openalex_mailto": "file@example.com"}))

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch.dict(os.environ, {"GANTRY_OPENALEX_MAILTO": "env@example.com"}):
            cfg = load_config()

    assert cfg.openalex_mailto == "env@example.com"


def test_config_openalex_mailto_save_roundtrip(tmp_path):
    """openalex_mailto round-trips through save/load."""
    config_file = tmp_path / "config.yaml"
    cfg = Config(openalex_mailto="ryan@example.com")

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch("pdf_gantry.config.GANTRY_DIR", tmp_path):
            save_config(cfg)
            loaded = load_config()

    assert loaded.openalex_mailto == "ryan@example.com"


def test_config_set_openalex_mailto(tmp_path):
    """set_config_value works for openalex_mailto."""
    config_file = tmp_path / "config.yaml"

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch("pdf_gantry.config.GANTRY_DIR", tmp_path):
            cfg = set_config_value("openalex_mailto", "ryan@example.com")

    assert cfg.openalex_mailto == "ryan@example.com"


def test_db_path_property():
    """Config.db_path is derived from index_dir."""
    cfg = Config()
    assert cfg.db_path == cfg.index_dir / "index.db"


def test_save_config_omits_unset_papers_dir(tmp_path):
    """save_config doesn't serialize a None papers_dir as the string 'None'."""
    config_file = tmp_path / "config.yaml"

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch("pdf_gantry.config.GANTRY_DIR", tmp_path):
            save_config(Config())
            loaded = load_config()

    data = yaml.safe_load(config_file.read_text())
    assert "papers_dir" not in data
    assert loaded.papers_dir is None


# --- CLI behavior when papers_dir is unconfigured ---

from click.testing import CliRunner  # noqa: E402

from pdf_gantry.cli import cli  # noqa: E402


@pytest.fixture
def no_papers_config(tmp_path, monkeypatch):
    """No config file, no GANTRY_PAPERS_DIR, index isolated to tmp_path."""
    for var in ("GANTRY_PAPERS_DIR", "GANTRY_VAULT_DIR", "GANTRY_BIB_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    with patch("pdf_gantry.config.CONFIG_PATH", tmp_path / "missing.yaml"):
        yield tmp_path


def test_cli_ingest_without_papers_dir_errors(no_papers_config):
    """ingest with no configured papers_dir exits 1 with setup guidance."""
    result = CliRunner().invoke(cli, ["ingest"])
    assert result.exit_code == 1
    assert "config init" in result.stderr
    assert "GANTRY_PAPERS_DIR" in result.stderr


def test_cli_ingest_without_papers_dir_json(no_papers_config):
    """JSON mode reports the missing papers_dir as a structured error."""
    import json as json_mod

    result = CliRunner().invoke(cli, ["ingest", "--json"])
    assert result.exit_code == 1
    data = json_mod.loads(result.output)
    assert "config init" in data["error"]


def test_cli_process_without_papers_dir_errors(no_papers_config):
    """process guards papers_dir before touching the database."""
    result = CliRunner().invoke(cli, ["process"])
    assert result.exit_code == 1
    assert "config init" in result.stderr


def test_cli_config_show_without_papers_dir(no_papers_config):
    """config show displays unset papers_dir as '(not set)', not 'None'."""
    result = CliRunner().invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "(not set)" in result.output
    assert "None" not in result.output


def test_cli_config_init_writes_papers_dir(tmp_path, monkeypatch):
    """config init prompts for papers_dir (no baked-in default) and saves it."""
    for var in ("GANTRY_PAPERS_DIR", "GANTRY_VAULT_DIR", "GANTRY_BIB_PATH"):
        monkeypatch.delenv(var, raising=False)
    config_file = tmp_path / "config.yaml"
    papers = tmp_path / "papers"

    with patch("pdf_gantry.config.CONFIG_PATH", config_file):
        with patch("pdf_gantry.config.GANTRY_DIR", tmp_path):
            result = CliRunner().invoke(cli, ["config", "init"], input=f"{papers}\n\n")

    assert result.exit_code == 0
    data = yaml.safe_load(config_file.read_text())
    assert data["papers_dir"] == str(papers)
