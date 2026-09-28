"""Enrichment misses are distinguishable from matches (and from never-tried).

Before this, a provider miss stamped `metadata_enriched_at` and nothing else,
so 850 papers in the real index looked like "attempted, source unknown" and
default enrich never retried them. A miss now writes
`metadata_source='none:<provider>'`; `enrich --retry-misses` re-attempts
them; and every consumer of `metadata_source` treats misses as not-enriched.
Legacy rows (enriched_at set, source NULL, title NULL) count as misses too.
"""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry import openalex
from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.metadata import enrich_documents, field_completeness, verify_documents
from pdf_gantry.queue import manual_condition, miss_condition, query_queue


def _seed(conn, paper_id, *, doi=None, title=None, raw_text="", source=None,
          enriched_at=None, suspect=0, authors=None):
    conn.execute(
        "INSERT INTO papers (id, path, filename, file_hash, file_size, file_modified, "
        "indexed_at, updated_at, doi, title, authors, metadata_source, "
        "metadata_enriched_at, metadata_suspect) "
        "VALUES (?, ?, ?, ?, 1, '2026-01-01', '2026-01-01', '2026-01-01', "
        "?, ?, ?, ?, ?, ?)",
        (paper_id, f"/p/p{paper_id}.pdf", f"p{paper_id}.pdf", f"h{paper_id}",
         doi, title, authors, source, enriched_at, suspect),
    )
    if raw_text:
        conn.execute(
            "INSERT INTO paper_text (paper_id, raw_text, markdown, text_length, "
            "markdown_length) VALUES (?, ?, '', ?, 0)",
            (paper_id, raw_text, len(raw_text)),
        )
    conn.commit()


@pytest.fixture
def db(tmp_path):
    conn = get_connection(str(tmp_path / "index.db"))
    yield conn
    conn.close()


def _all_miss(monkeypatch):
    monkeypatch.setattr(openalex, "fetch_by_doi", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_filename", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_title", lambda *a, **k: None)


def _match(title="Matched Title"):
    return lambda *a, **k: {
        "title": title, "authors": ["A B"], "year": 2020,
        "abstract": "abs", "doi": None, "source_id": "W",
    }


def _ids(conn, where):
    return [r[0] for r in conn.execute(f"SELECT id FROM papers WHERE {where} ORDER BY id")]


# --- the miss is recorded ---------------------------------------------------


def test_miss_writes_none_provider_source(db, monkeypatch):
    _seed(db, 1, raw_text="Some Paper Title Here\nbody text")
    _all_miss(monkeypatch)

    stats = enrich_documents(db, paper_ids=[1])

    assert stats.no_match == 1
    row = db.execute(
        "SELECT metadata_source, metadata_enriched_at FROM papers WHERE id = 1"
    ).fetchone()
    assert row["metadata_source"] == "none:openalex"
    assert row["metadata_enriched_at"] is not None


def test_miss_semantic_scholar_provider_name(db, monkeypatch):
    import pdf_gantry.metadata as md
    _seed(db, 1, doi="10.1/x")
    monkeypatch.setattr(md, "_fetch_by_doi", lambda *a, **k: None)
    monkeypatch.setattr(md, "_fetch_by_title", lambda *a, **k: None)

    enrich_documents(db, paper_ids=[1], provider="semantic-scholar")

    src = db.execute("SELECT metadata_source FROM papers WHERE id = 1").fetchone()[0]
    assert src == "none:semantic_scholar"


def test_api_error_is_not_recorded_as_miss(db, monkeypatch):
    """A network failure is not evidence the provider lacks the paper: leave
    the paper unstamped so the next default enrich retries it."""
    _seed(db, 1, doi="10.1/x")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(openalex, "fetch_by_doi", boom)
    monkeypatch.setattr(openalex, "fetch_by_filename", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_title", lambda *a, **k: None)

    stats = enrich_documents(db, paper_ids=[1])

    assert stats.api_errors == 1
    row = db.execute(
        "SELECT metadata_source, metadata_enriched_at FROM papers WHERE id = 1"
    ).fetchone()
    assert row["metadata_source"] is None
    assert row["metadata_enriched_at"] is None


def test_retried_miss_that_matches_gets_real_source(db, monkeypatch):
    _seed(db, 1, raw_text="Matched Title\nbody", source="none:openalex",
          enriched_at="2026-09-10T00:00:00+00:00")
    monkeypatch.setattr(openalex, "fetch_by_doi", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_filename", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_title", _match())

    enrich_documents(db, paper_ids=[1])

    row = db.execute("SELECT metadata_source, title FROM papers WHERE id = 1").fetchone()
    assert row["metadata_source"] == "openalex_title"
    assert row["title"] == "Matched Title"


# --- predicates -------------------------------------------------------------


def _seed_states(conn):
    _seed(conn, 1)  # never attempted
    _seed(conn, 2, source="none:openalex", enriched_at="2026-09-10T00:00:00+00:00")
    # legacy miss: stamped, no source, no title
    _seed(conn, 3, enriched_at="2026-09-10T00:00:00+00:00")
    _seed(conn, 4, source="openalex", title="Real", enriched_at="2026-09-10T00:00:00+00:00",
          raw_text="Real paper body")
    _seed(conn, 5, source="manual:ryan", title="Hand", enriched_at="2026-09-10T00:00:00+00:00")
    # legacy hand-SQL source names
    _seed(conn, 6, source="brev-manual-from-text", title="Hand2",
          enriched_at="2026-09-10T00:00:00+00:00")


def test_miss_condition_covers_tagged_and_legacy(db):
    _seed_states(db)
    assert _ids(db, miss_condition()) == [2, 3]


def test_manual_condition_covers_new_and_legacy(db):
    _seed_states(db)
    assert _ids(db, manual_condition()) == [5, 6]


def test_queue_is_enrich_miss(db):
    _seed_states(db)
    assert [r["id"] for r in query_queue(db, is_prop=["enrich-miss"])] == [2, 3]


def test_queue_is_manual_metadata(db):
    _seed_states(db)
    assert [r["id"] for r in query_queue(db, is_prop=["manual-metadata"])] == [5, 6]


# --- default selection and --retry-misses ------------------------------------


def test_default_enrich_skips_misses_and_manual(db, monkeypatch):
    _seed_states(db)
    seen = []
    monkeypatch.setattr(openalex, "fetch_by_doi", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_filename",
                        lambda fn, **k: seen.append(fn) or None)
    monkeypatch.setattr(openalex, "fetch_by_title", lambda *a, **k: None)

    stats = enrich_documents(db)

    assert seen == ["p1.pdf"]
    assert stats.total == 1


def test_retry_misses_includes_misses(db, monkeypatch):
    _seed_states(db)
    seen = []
    monkeypatch.setattr(openalex, "fetch_by_doi", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_filename",
                        lambda fn, **k: seen.append(fn) or None)
    monkeypatch.setattr(openalex, "fetch_by_title", lambda *a, **k: None)

    enrich_documents(db, retry_misses=True)

    assert sorted(seen) == ["p1.pdf", "p2.pdf", "p3.pdf"]


def test_explicit_ids_never_clobber_manual_metadata(db, monkeypatch):
    _seed_states(db)
    monkeypatch.setattr(openalex, "fetch_by_doi", lambda *a, **k: None)
    monkeypatch.setattr(openalex, "fetch_by_filename", _match("Clobber"))
    monkeypatch.setattr(openalex, "fetch_by_title", lambda *a, **k: None)

    stats = enrich_documents(db, paper_ids=[5, 6])

    assert stats.skipped_manual == 2
    titles = [r[0] for r in db.execute("SELECT title FROM papers WHERE id IN (5,6) ORDER BY id")]
    assert titles == ["Hand", "Hand2"]


def test_cli_enrich_default_is_never_attempted_only(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_states(conn)
    conn.close()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    r = CliRunner().invoke(cli, ["enrich", "--dry-run", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["would_enrich"] == 1

    r = CliRunner().invoke(cli, ["enrich", "--retry-misses", "--dry-run", "--json"])
    assert json.loads(r.output)["would_enrich"] == 3


def test_cli_enrich_ids(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_states(conn)
    conn.close()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))
    _all_miss(monkeypatch)

    r = CliRunner().invoke(cli, ["enrich", "--ids", "4,5,99", "--json"])
    data = json.loads(r.output)
    assert data["total"] == 1
    assert data["skipped_manual"] == 1
    assert data["not_found"] == [99]


# --- consumers --------------------------------------------------------------


def test_verify_ignores_misses(db):
    _seed_states(db)
    checked = {r.paper_id for r in verify_documents(db)}
    assert 2 not in checked and 3 not in checked
    assert 4 in checked


def test_gaps_reports_provider_misses(db):
    _seed_states(db)
    result = field_completeness(db)
    assert result["enrich_misses"] == 2
    assert result["fields"]["title"]["provider_miss"] == 2
    # existing keys keep their meaning
    assert result["fields"]["title"]["attempted_incomplete"] == 2


def test_cli_queue_is_enrich_miss_and_manual(tmp_path, monkeypatch):
    conn = get_connection(str(tmp_path / "index.db"))
    _seed_states(conn)
    conn.close()
    monkeypatch.setenv("GANTRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("GANTRY_PAPERS_DIR", str(tmp_path))

    r = CliRunner().invoke(cli, ["queue", "--is", "enrich-miss", "--count"])
    assert r.exit_code == 0, r.output
    assert r.output.strip().endswith("2")
    r = CliRunner().invoke(cli, ["queue", "--is", "manual-metadata", "--count"])
    assert r.output.strip().endswith("2")
