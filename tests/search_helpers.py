"""Helpers for building tiny search indexes without PDFs."""

import struct

import yaml

from pdf_gantry.db import get_connection


def vec(seed: int, dim: int = 768) -> bytes:
    """A deterministic unit-ish vector; different seeds point different ways."""
    values = [0.0] * dim
    values[seed % dim] = 1.0
    return struct.pack(f"{dim}f", *values)


def add_paper(conn, pid, filename, *, title=None, authors=None, year=None, citekey=None,
              text="", chunks=(), chunk_vecs=None, doc_vec=None):
    """Insert a paper with text, FTS row, optional chunks and vectors."""
    conn.execute(
        "INSERT INTO papers (id, path, filename, file_hash, file_size, file_modified, title, "
        "authors, year, citekey, has_text, has_chunk_embeddings, has_embeddings, "
        "indexed_at, updated_at) "
        "VALUES (?, ?, ?, ?, 1, '2026-01-01', ?, ?, ?, ?, 1, ?, ?, '2026-01-01', '2026-01-01')",
        (pid, f"/p/{filename}", filename, f"h{pid}", title, authors, year, citekey,
         int(chunk_vecs is not None), int(doc_vec is not None)),
    )
    conn.execute("INSERT INTO paper_text (paper_id, raw_text) VALUES (?, ?)", (pid, text))
    conn.execute(
        "INSERT INTO papers_fts(rowid, filename, title, authors, abstract, text_content) "
        "VALUES (?, ?, ?, ?, '', ?)",
        (pid, filename, title or "", authors or "", text),
    )
    for i, chunk_text in enumerate(chunks):
        cur = conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text) VALUES (?, ?, ?)",
            (pid, i, chunk_text),
        )
        if chunk_vecs is not None:
            conn.execute(
                "INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                (cur.lastrowid, chunk_vecs[i]),
            )
    if doc_vec is not None:
        conn.execute(
            "INSERT INTO paper_embeddings (paper_id, embedding) VALUES (?, ?)", (pid, doc_vec)
        )
    conn.commit()


def new_db(tmp_path):
    db_path = tmp_path / "index.db"
    return get_connection(str(db_path)), db_path


def point_cli_at(tmp_path, monkeypatch, db_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"index_dir": str(db_path.parent)}))
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)
