"""run_pipeline's final collections refresh: additive, optional, never fatal."""

import shutil
from unittest.mock import patch

from pdf_gantry import collections as coll
from pdf_gantry.config import Config
from pdf_gantry.db import get_connection
from pdf_gantry.pipeline import run_pipeline


def _setup(tmp_path):
    db_path = tmp_path / "index.db"
    return db_path, get_connection(str(db_path))


def test_pipeline_does_not_create_the_sidecar(tmp_path, papers_dir):
    db_path, conn = _setup(tmp_path)
    stats = run_pipeline(conn, papers_dir, db_path, workers=1)
    assert not (tmp_path / "collections.db").exists()
    assert "collections" not in stats


def test_pipeline_refreshes_drifted_keys_as_its_last_step(tmp_path, papers_dir):
    db_path, conn = _setup(tmp_path)
    run_pipeline(conn, papers_dir, db_path, workers=1)
    pid, name = conn.execute("SELECT id, filename FROM papers ORDER BY id").fetchone()

    side = coll.connect(Config(index_dir=tmp_path))
    coll.create(side, "c", by="t")
    coll.add(side, conn, "c", [pid], "t")

    shutil.move(papers_dir / name, papers_dir / "renamed.pdf")
    stats = run_pipeline(conn, papers_dir, db_path, workers=1)

    assert stats["collections"]["updated"] == 1
    assert side.execute("SELECT path FROM memberships").fetchone()["path"] == "renamed.pdf"
    side.close()


def test_pipeline_refreshes_when_embeddings_are_unavailable(tmp_path, papers_dir):
    """Stage 3 returns early without sentence-transformers; refresh must still run."""
    db_path, conn = _setup(tmp_path)
    coll.connect(Config(index_dir=tmp_path)).close()
    with patch.dict("sys.modules", {"pdf_gantry.embeddings": None}):
        stats = run_pipeline(conn, papers_dir, db_path, workers=1)
    assert stats["collections"]["checked"] == 0


def test_a_refresh_failure_never_fails_the_pipeline(tmp_path, papers_dir):
    db_path, conn = _setup(tmp_path)
    coll.connect(Config(index_dir=tmp_path)).close()
    with patch("pdf_gantry.collections.refresh", side_effect=RuntimeError("boom")):
        stats = run_pipeline(conn, papers_dir, db_path, workers=1)
    assert stats["processed"] == 2
    assert stats["collections"] == {"error": "boom"}


def test_dry_run_does_not_refresh(tmp_path, papers_dir):
    db_path, conn = _setup(tmp_path)
    coll.connect(Config(index_dir=tmp_path)).close()
    with patch("pdf_gantry.collections.refresh") as refresh:
        stats = run_pipeline(conn, papers_dir, db_path, dry_run=True)
    refresh.assert_not_called()
    assert "collections" not in stats
