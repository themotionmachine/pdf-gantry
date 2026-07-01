"""
Search-mode provenance in JSON output.

The wall: ``gantry search --json`` produces the same JSON shape whether
the search ran full hybrid RRF fusion or silently fell back to FTS-only
(e.g. because sentence-transformers isn't installed).  A remote agent
reading the output cannot distinguish a rich two-signal result from a
quietly degraded single-signal result.

The window: a ``mode`` key in the top-level JSON object that carries
one of three machine-readable values:

    "hybrid"   — FTS5 + vector RRF fusion both ran
    "fts"      — FTS5 only (either --fts was passed, or embeddings
                 unavailable and search degraded silently)
    "fts_only" — caller explicitly passed --fts (explicit, not degraded)

The sensor is wired across two previously unconnected parts:
  1. The CLI's execution-path decision (``use_hybrid`` bool, the
     ImportError catch in the search command)
  2. The JSON result payload the agent parses

These tests assert the signal is present and correct BEFORE the code
change is made (all should be RED on the un-patched codebase).
"""

import json

import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cli_env(tmp_path, monkeypatch, *, embed_raises=False, has_results=False):
    """
    Minimal configured environment for CLI search tests.

    ``embed_raises`` — simulate sentence-transformers unavailable.
    ``has_results``  — if True, return a fake SearchResult from both paths.
    """
    db_path = tmp_path / "index.db"
    get_connection(str(db_path)).close()
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)

    from pdf_gantry.models import SearchResult

    fake_result = SearchResult(
        id=1, filename="paper.pdf", path="/papers/paper.pdf",
        score=0.9, snippet="some text", title="A Paper",
    )

    def _embed(model, q):
        if embed_raises:
            raise ImportError("sentence-transformers not installed")
        return b"qvec"

    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", _embed)

    if has_results:
        monkeypatch.setattr(
            "pdf_gantry.search.hybrid_search",
            lambda *a, **k: [fake_result],
        )
        monkeypatch.setattr(
            "pdf_gantry.search.fts_search",
            lambda *a, **k: [fake_result],
        )
    else:
        monkeypatch.setattr(
            "pdf_gantry.search.hybrid_search",
            lambda *a, **k: [],
        )
        monkeypatch.setattr(
            "pdf_gantry.search.fts_search",
            lambda *a, **k: [],
        )

    monkeypatch.setattr("pdf_gantry.search.search_count", lambda *a, **k: 0)


# ---------------------------------------------------------------------------
# Core provenance tests  (RED before this round's patch)
# ---------------------------------------------------------------------------

def test_search_json_has_mode_field_when_hybrid(tmp_path, monkeypatch):
    """
    When full hybrid runs, ``--json`` output must contain ``"mode": "hybrid"``.
    This is RED before the patch because no ``mode`` key is emitted.
    """
    _make_cli_env(tmp_path, monkeypatch, has_results=True)
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "climate", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "mode" in payload, (
        "JSON output missing 'mode' key — agent cannot tell which path ran"
    )
    assert payload["mode"] == "hybrid"


def test_search_json_has_mode_field_when_fts_only_flag(tmp_path, monkeypatch):
    """
    When ``--fts`` is passed explicitly, ``--json`` output must contain
    ``"mode": "fts_only"`` (caller chose it, not a degradation).
    This is RED before the patch.
    """
    _make_cli_env(tmp_path, monkeypatch, has_results=True)
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "climate", "--fts", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "mode" in payload, (
        "JSON output missing 'mode' key — agent cannot tell which path ran"
    )
    assert payload["mode"] == "fts_only"


def test_search_json_mode_fts_degraded_when_embeddings_unavailable(tmp_path, monkeypatch):
    """
    When embeddings are unavailable and search silently falls back, ``--json``
    must contain ``"mode": "fts"`` (degraded) — not ``"hybrid"`` and not
    ``"fts_only"``.  This is the darkest blind spot: the agent trusted a
    result it thought was hybrid; with this sensor it can see the fallback.
    This is RED before the patch.
    """
    _make_cli_env(tmp_path, monkeypatch, embed_raises=True, has_results=True)
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "climate", "--json"])
    # May be exit 0 (results) or 2 (no results); either is fine here
    assert result.exit_code in (0, 2), result.output
    payload = json.loads(result.stdout)
    assert "mode" in payload, (
        "JSON output missing 'mode' key — agent cannot detect silent FTS fallback"
    )
    assert payload["mode"] == "fts", (
        f"Expected mode='fts' (degraded), got mode={payload.get('mode')!r}"
    )


def test_search_json_mode_present_on_empty_results(tmp_path, monkeypatch):
    """
    Even when search returns 0 results, ``mode`` must be in the JSON so the
    agent can determine whether to try a different search strategy.
    This is RED before the patch.
    """
    _make_cli_env(tmp_path, monkeypatch, has_results=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "climate", "--json"])
    assert result.exit_code == 2  # EXIT_NO_RESULTS
    payload = json.loads(result.stdout)
    assert "mode" in payload, (
        "Empty-results JSON missing 'mode' — agent can't distinguish "
        "hybrid-no-results from fts-no-results"
    )
    assert payload["mode"] in ("hybrid", "fts", "fts_only")


def test_search_json_mode_present_on_degraded_empty_results(tmp_path, monkeypatch):
    """
    When embeddings are unavailable AND search returns 0 results, ``mode``
    must still be in the JSON so the agent knows both facts: it got a
    degraded path AND it got nothing.
    This is RED before the patch.
    """
    _make_cli_env(tmp_path, monkeypatch, embed_raises=True, has_results=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "climate", "--json"])
    assert result.exit_code in (0, 2), result.output
    payload = json.loads(result.stdout)
    assert "mode" in payload, (
        "Degraded-empty JSON missing 'mode' — worst case: silent failure, no signal"
    )
    assert payload["mode"] == "fts"
