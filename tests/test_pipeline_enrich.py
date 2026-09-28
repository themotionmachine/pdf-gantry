"""`pipeline --enrich` and positional file arguments (E12, B4).

The nightly ran `pipeline`, which never enriched, so papers added after
09-12 were chunk-embedded without a title (chunk text is prefixed with the
paper title, see chunking.prepare_chunk_text). The acquisition skill ran
`gantry pipeline <filename>`, which Click rejected. These pin: positional
files are an alias for --file; --enrich runs between process and embed so
the title reaches the embeddings; and JSON reports the enrich counts.
"""

import json
import shutil

import pytest
from click.testing import CliRunner

from pdf_gantry import openalex
from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.pipeline import run_pipeline


def _fake_openalex(monkeypatch, title="Enriched Title", calls=None):
    def by_filename(filename, **k):
        if calls is not None:
            calls.append(filename)
        return {"title": title, "authors": ["A B"], "year": 2024,
                "abstract": None, "doi": None, "source_id": "W"}
    monkeypatch.setattr(openalex, "fetch_by_doi", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_filename", by_filename)
    monkeypatch.setattr(openalex, "fetch_by_title", lambda *a, **k: None)


@pytest.fixture
def env(tmp_path, papers_dir, monkeypatch):
    index = tmp_path / "idx"
    index.mkdir()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(index))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(papers_dir))
    return index / "index.db"


def _conn(db_path):
    return get_connection(str(db_path))


# --- run_pipeline -----------------------------------------------------------


def test_run_pipeline_enrich_runs_before_embed(tmp_path, papers_dir, monkeypatch):
    """Chunk embeddings are computed after the title lands."""
    _fake_openalex(monkeypatch)
    import pdf_gantry.embeddings as emb
    seen_titles = []
    real = emb.embed_chunks

    def spy(conn, db_path, paper_ids=None, **kw):
        for pid in paper_ids or []:
            seen_titles.append(conn.execute(
                "SELECT title FROM papers WHERE id = ?", (pid,)).fetchone()[0])
        return real(conn, db_path, paper_ids=paper_ids, **kw)

    monkeypatch.setattr(emb, "embed_chunks", spy)
    db_path = tmp_path / "t.db"
    conn = _conn(db_path)

    stats = run_pipeline(conn, papers_dir, db_path, enrich=True)

    assert stats["enriched"] == 2
    assert stats["enrich"]["matched"] == 2
    assert seen_titles and all(t == "Enriched Title" for t in seen_titles)
    conn.close()


def test_run_pipeline_without_enrich_does_not_call_provider(tmp_path, papers_dir,
                                                            monkeypatch):
    calls = []
    _fake_openalex(monkeypatch, calls=calls)
    db_path = tmp_path / "t.db"
    conn = _conn(db_path)
    stats = run_pipeline(conn, papers_dir, db_path)
    assert calls == []
    assert stats["enriched"] == 0
    assert "enrich" not in stats
    conn.close()


def test_run_pipeline_enrich_reports_titled_after_embed(tmp_path, papers_dir,
                                                        monkeypatch):
    """Papers embedded before they had a title are named, for embed --force."""
    db_path = tmp_path / "t.db"
    conn = _conn(db_path)
    run_pipeline(conn, papers_dir, db_path)  # embedded, untitled
    _fake_openalex(monkeypatch)

    stats = run_pipeline(conn, papers_dir, db_path, enrich=True)

    assert sorted(stats["titled_after_embed"]) == [1, 2]
    conn.close()


def test_run_pipeline_single_file_rerun_does_not_crash(tmp_path, papers_dir):
    """Re-running --file on an already-indexed paper used to hit an unbound
    ingest_stats (NameError)."""
    db_path = tmp_path / "t.db"
    conn = _conn(db_path)
    run_pipeline(conn, papers_dir, db_path, filename="test_climate.pdf")
    stats = run_pipeline(conn, papers_dir, db_path, filename="test_climate.pdf")
    assert stats["processed"] == 0
    assert "error" not in stats
    conn.close()


# --- CLI --------------------------------------------------------------------


def test_cli_pipeline_positional_file(env):
    r = CliRunner().invoke(cli, ["pipeline", "test_climate.pdf", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["processed"] == 1
    conn = _conn(env)
    names = [x[0] for x in conn.execute("SELECT filename FROM papers WHERE has_text = 1")]
    conn.close()
    assert names == ["test_climate.pdf"]
    assert data["ids"] and len(data["ids"]) == 1


def test_cli_pipeline_positional_full_path_inside_papers_dir(env, papers_dir):
    r = CliRunner().invoke(
        cli, ["pipeline", str(papers_dir / "test_climate.pdf"), "--json"]
    )
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["processed"] == 1


def test_cli_pipeline_path_outside_papers_dir_is_error(env, tmp_path, sample_pdf):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    shutil.copy(sample_pdf, outside / "x.pdf")
    r = CliRunner().invoke(cli, ["pipeline", str(outside / "x.pdf"), "--json"])
    assert r.exit_code == 1
    assert "papers directory" in json.loads(r.output)["error"]


def test_cli_pipeline_missing_file_is_error(env):
    r = CliRunner().invoke(cli, ["pipeline", "Nonexistent.pdf", "--json"])
    assert r.exit_code == 1, r.output
    assert "not found" in json.loads(r.output)["error"].lower()


def test_cli_pipeline_multiple_positional_files(env):
    r = CliRunner().invoke(
        cli, ["pipeline", "test_climate.pdf", "ml_nlp_paper.pdf", "--json"]
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["processed"] == 2
    assert len(data["files"]) == 2
    assert sorted(data["ids"]) == [1, 2]


def test_cli_pipeline_file_enrich_one_shot(env, monkeypatch):
    """The acquisition flow: add one paper end to end, metadata included."""
    _fake_openalex(monkeypatch, title="Climate Paper")
    r = CliRunner().invoke(
        cli, ["pipeline", "--file", "test_climate.pdf", "--enrich", "--json"]
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["enriched"] == 1
    assert data["enrich"]["matched"] == 1
    conn = _conn(env)
    row = conn.execute(
        "SELECT title, has_chunk_embeddings FROM papers WHERE filename = 'test_climate.pdf'"
    ).fetchone()
    conn.close()
    assert row["title"] == "Climate Paper"
    assert row["has_chunk_embeddings"] == 1


def test_cli_pipeline_file_enrich_skips_already_enriched(env, monkeypatch):
    calls = []
    _fake_openalex(monkeypatch, calls=calls)
    CliRunner().invoke(cli, ["pipeline", "test_climate.pdf", "--enrich", "--json"])
    r = CliRunner().invoke(cli, ["pipeline", "test_climate.pdf", "--enrich", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["enriched"] == 0
    assert calls == ["test_climate.pdf"]
