"""Tests that `gantry search` defaults to hybrid mode (with FTS opt-out)."""

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection


@pytest.fixture
def cli_db(tmp_path, monkeypatch):
    """Empty configured DB plus monkeypatched search funcs that record which ran."""
    get_connection(str(tmp_path / "index.db")).close()
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)

    calls = []
    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", lambda model, q: b"qvec")
    monkeypatch.setattr(
        "pdf_gantry.search.hybrid_search",
        lambda *a, **k: (calls.append("hybrid") or []),
    )
    monkeypatch.setattr(
        "pdf_gantry.search.fts_search",
        lambda *a, **k: (calls.append("fts") or []),
    )
    monkeypatch.setattr("pdf_gantry.search.search_count", lambda *a, **k: 0)
    return calls


def test_search_defaults_to_hybrid(cli_db):
    runner = CliRunner()
    runner.invoke(cli, ["search", "climate", "--json"])
    assert "hybrid" in cli_db
    assert "fts" not in cli_db


def test_search_fts_flag_forces_fts(cli_db):
    runner = CliRunner()
    runner.invoke(cli, ["search", "climate", "--fts", "--json"])
    assert "fts" in cli_db
    assert "hybrid" not in cli_db


def test_search_hybrid_flag_still_hybrid(cli_db):
    """Back-compat: explicit --hybrid keeps working."""
    runner = CliRunner()
    runner.invoke(cli, ["search", "climate", "--hybrid", "--json"])
    assert "hybrid" in cli_db
    assert "fts" not in cli_db


def test_search_falls_back_to_fts_when_embeddings_unavailable(tmp_path, monkeypatch):
    """If the embedding model can't load, default search degrades to FTS, not error."""
    get_connection(str(tmp_path / "index.db")).close()
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)

    calls = []

    def _boom(model, q):
        raise ImportError("sentence-transformers not installed")

    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", _boom)
    monkeypatch.setattr(
        "pdf_gantry.search.fts_search",
        lambda *a, **k: (calls.append("fts") or []),
    )
    monkeypatch.setattr("pdf_gantry.search.search_count", lambda *a, **k: 0)

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "climate", "--json"])
    assert "fts" in calls
    # No hard error exit code (2 = no results is fine here).
    assert result.exit_code in (0, 2)
