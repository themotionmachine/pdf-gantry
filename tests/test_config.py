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
    """Config with no file and no env vars returns defaults."""
    with patch("pdf_gantry.config.CONFIG_PATH", Path("/nonexistent/config.yaml")):
        cfg = load_config()
    assert cfg.papers_dir.name == "Papers"
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


def test_db_path_property():
    """Config.db_path is derived from index_dir."""
    cfg = Config()
    assert cfg.db_path == cfg.index_dir / "index.db"
