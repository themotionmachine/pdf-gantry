"""`embed [--chunk] --ids ... --force` re-embeds specific papers (B9).

Deleting chunk vectors by hand left has_chunk_embeddings=1, so `embed
--chunk` said "Nothing to embed" and the only fix was raw SQL resetting the
flag. --ids scopes embed to named papers; --force re-embeds them even when
flagged done, deleting their stale vectors first.
"""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.embeddings import embed_chunks, embed_documents, reset_embeddings
from pdf_gantry.ingest import ingest_directory
from pdf_gantry.process import process_documents


@pytest.fixture
def embedded(tmp_path, papers_dir, monkeypatch):
    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))
    ingest_directory(conn, papers_dir)
    process_documents(conn, papers_dir, db_path, workers=1)
    embed_chunks(conn, db_path)
    embed_documents(conn, db_path)
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(papers_dir))
    yield conn
    conn.close()


def _chunk_vecs(conn, pid):
    return conn.execute(
        "SELECT COUNT(*) FROM chunk_vec WHERE chunk_id IN "
        "(SELECT chunk_id FROM chunks WHERE doc_id = ?)", (pid,)
    ).fetchone()[0]


def _flags(conn, pid):
    r = conn.execute(
        "SELECT has_embeddings, has_chunk_embeddings FROM papers WHERE id = ?", (pid,)
    ).fetchone()
    return r["has_embeddings"], r["has_chunk_embeddings"]


def test_reset_chunk_embeddings_deletes_vectors_and_flag(embedded):
    assert _chunk_vecs(embedded, 1) > 0
    reset_embeddings(embedded, [1], chunk=True)
    assert _chunk_vecs(embedded, 1) == 0
    assert _flags(embedded, 1) == (1, 0)
    assert _chunk_vecs(embedded, 2) > 0


def test_reset_doc_embeddings(embedded):
    reset_embeddings(embedded, [1], chunk=False)
    n = embedded.execute(
        "SELECT COUNT(*) FROM paper_embeddings WHERE paper_id = 1"
    ).fetchone()[0]
    assert n == 0
    assert _flags(embedded, 1) == (0, 1)


def test_cli_embed_chunk_ids_without_force_skips_done(embedded):
    r = CliRunner().invoke(cli, ["embed", "--chunk", "--ids", "1", "--json"])
    assert r.exit_code == 2, r.output
    data = json.loads(r.output)
    assert data["total"] == 0
    assert "--force" in data["message"]


def test_cli_embed_chunk_ids_force_reembeds(embedded):
    # the B9 situation: vectors gone, flag still 1
    embedded.execute(
        "DELETE FROM chunk_vec WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = 1)"
    )
    embedded.commit()
    assert _flags(embedded, 1)[1] == 1

    r = CliRunner().invoke(cli, ["embed", "--chunk", "--ids", "1", "--force", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["succeeded"] > 0
    assert _chunk_vecs(embedded, 1) > 0
    assert _flags(embedded, 1)[1] == 1


def test_cli_embed_ids_force_doc_level(embedded):
    r = CliRunner().invoke(cli, ["embed", "--ids", "1,2", "--force", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["succeeded"] == 2


def test_cli_embed_ids_reports_not_found(embedded):
    r = CliRunner().invoke(cli, ["embed", "--chunk", "--ids", "1,99", "--force", "--json"])
    assert r.exit_code == 3, r.output  # partial
    assert json.loads(r.output)["not_found"] == [99]


def test_cli_embed_force_requires_ids(embedded):
    r = CliRunner().invoke(cli, ["embed", "--chunk", "--force", "--json"])
    assert r.exit_code == 1
    assert "--ids" in json.loads(r.output)["error"]


def test_cli_embed_ids_dry_run_no_write(embedded):
    r = CliRunner().invoke(
        cli, ["embed", "--chunk", "--ids", "1", "--force", "--dry-run", "--json"]
    )
    assert r.exit_code == 0
    assert json.loads(r.output)["would_embed"] == 1
    assert _chunk_vecs(embedded, 1) > 0
