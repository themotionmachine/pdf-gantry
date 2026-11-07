"""run_pipeline's final reading.db refresh: additive, optional, never fatal."""

import shutil
from unittest.mock import patch

from pdf_gantry import reading as rd
from pdf_gantry.config import Config
from pdf_gantry.db import get_connection
from pdf_gantry.pipeline import run_pipeline


def _setup(tmp_path):
    db_path = tmp_path / "index.db"
    return db_path, get_connection(str(db_path))


def test_pipeline_does_not_create_the_sidecar(tmp_path, papers_dir):
    db_path, conn = _setup(tmp_path)
    stats = run_pipeline(conn, papers_dir, db_path, workers=1)
    assert not (tmp_path / "reading.db").exists()
    assert "reading" not in stats


def test_pipeline_refreshes_drifted_item_keys(tmp_path, papers_dir):
    db_path, conn = _setup(tmp_path)
    run_pipeline(conn, papers_dir, db_path, workers=1)
    pid, name = conn.execute("SELECT id, filename FROM papers ORDER BY id").fetchone()

    side = rd.connect(Config(index_dir=tmp_path))
    rd.enqueue(side, conn, [pid], "t")

    shutil.move(papers_dir / name, papers_dir / "renamed.pdf")
    stats = run_pipeline(conn, papers_dir, db_path, workers=1)

    assert stats["reading"]["updated"] == 1
    assert side.execute("SELECT path FROM items").fetchone()["path"] == "renamed.pdf"
    side.close()


def test_a_reading_refresh_failure_never_fails_the_pipeline(tmp_path, papers_dir):
    db_path, conn = _setup(tmp_path)
    rd.connect(Config(index_dir=tmp_path)).close()
    with patch("pdf_gantry.reading.refresh", side_effect=RuntimeError("boom")):
        stats = run_pipeline(conn, papers_dir, db_path, workers=1)
    assert stats["processed"] == 2
    assert stats["reading"] == {"error": "boom"}


def test_a_reading_failure_does_not_stop_the_collections_refresh(tmp_path, papers_dir):
    db_path, conn = _setup(tmp_path)
    rd.connect(Config(index_dir=tmp_path)).close()
    from pdf_gantry import collections as coll
    coll.connect(Config(index_dir=tmp_path)).close()
    with patch("pdf_gantry.reading.refresh", side_effect=RuntimeError("boom")):
        stats = run_pipeline(conn, papers_dir, db_path, workers=1)
    assert stats["collections"]["checked"] == 0 and "error" in stats["reading"]


def test_dry_run_does_not_refresh(tmp_path, papers_dir):
    db_path, conn = _setup(tmp_path)
    rd.connect(Config(index_dir=tmp_path)).close()
    with patch("pdf_gantry.reading.refresh") as refresh:
        stats = run_pipeline(conn, papers_dir, db_path, dry_run=True)
    refresh.assert_not_called()
    assert "reading" not in stats
