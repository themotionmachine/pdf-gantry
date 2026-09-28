"""Chunk pages are exposed wherever chunks are serialised (E7): info --query's
top chunk, info --chunks, read --chunks, read --chunk (with/without --context)."""

import json
import struct

import pytest
import yaml
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection


def _vec(axis: int, dim: int = 768) -> bytes:
    v = [0.0] * dim
    v[axis] = 1.0
    return struct.pack(f"{dim}f", *v)


@pytest.fixture
def paged_db(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    doc = conn.execute(
        "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at, has_chunk_embeddings) "
        "VALUES ('a.pdf', 'a.pdf', 'h', 1, 't', 't', 't', 1)"
    ).lastrowid
    ids = []
    for i, (ps, pe) in enumerate([(1, 1), (4, 5)]):
        cid = conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text, page_start, page_end) "
            "VALUES (?, ?, ?, ?, ?)", (doc, i, f"chunk {i} text", ps, pe)
        ).lastrowid
        conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                     (cid, _vec(i)))
        ids.append(cid)
    conn.commit()
    conn.close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"index_dir": str(tmp_path)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", cfg)
    monkeypatch.setattr("pdf_gantry.embeddings.embed_query", lambda model, q: _vec(1))
    return doc, ids


def _run(args):
    r = CliRunner().invoke(cli, args)
    assert r.exit_code == 0, r.output
    return json.loads(r.output)


def test_info_query_top_chunk_has_pages(paged_db):
    doc, _ = paged_db
    tc = _run(["info", "--ids", str(doc), "--query", "x", "--json"])["papers"][0]["top_chunk"]
    assert (tc["page_start"], tc["page_end"]) == (4, 5)


def test_info_query_context_top_chunk_has_pages(paged_db):
    doc, _ = paged_db
    tc = _run(["info", "--ids", str(doc), "--query", "x", "--context", "500",
               "--json"])["papers"][0]["top_chunk"]
    assert (tc["page_start"], tc["page_end"]) == (4, 5)


def test_info_chunks_have_pages(paged_db):
    doc, _ = paged_db
    chunks = _run(["info", "--ids", str(doc), "--chunks", "--json"])["papers"][0]["chunks"]
    assert [(c["page_start"], c["page_end"]) for c in chunks] == [(1, 1), (4, 5)]


def test_read_chunks_list_has_pages(paged_db):
    doc, _ = paged_db
    chunks = _run(["read", str(doc), "--chunks", "--json"])["chunks"]
    assert [(c["page_start"], c["page_end"]) for c in chunks] == [(1, 1), (4, 5)]


def test_read_single_chunk_has_pages(paged_db):
    doc, ids = paged_db
    d = _run(["read", str(doc), "--chunk", str(ids[1]), "--json"])
    assert (d["page_start"], d["page_end"]) == (4, 5)


def test_read_chunk_context_has_pages(paged_db):
    doc, ids = paged_db
    d = _run(["read", str(doc), "--chunk", str(ids[1]), "--context", "500", "--json"])
    assert (d["page_start"], d["page_end"]) == (4, 5)
