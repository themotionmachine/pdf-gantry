"""Bib parsing, layout fallback in link, suggest, audit, pull and missing."""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry import bibsync
from pdf_gantry.cli import cli
from pdf_gantry.link import clean_bib_text, match_by_title, parse_bib_file, reconcile
from pdf_gantry.metadata import enrich_documents, field_completeness, verify_documents
from pdf_gantry.queue import manual_condition, pending_counts

from .search_helpers import add_paper, new_db, point_cli_at

BIB = r"""
@article{smith2020trust,
  title = {[C] Civic {Trust} in \textit{Northern} Cities \& ``Towns''},
  author = {Smith, Jane and {World Bank and Partners} and Caf\'e, Ren\'e},
  year = {2020},
  doi = {https://doi.org/10.1000/ABC},
  abstract = {We study {trust}.},
  file = {/Users/Shared/Papers/smith 2020.pdf},
}
@online{doe2021web,
  title = {Zebrafish Governance Systems in Practice},
  author = {Doe, John},
  date = {2021-03-01},
}
@book{roe2019book,
  title = {Quokka Budgets and Municipal Finance},
  author = {Roe, Richard},
  year = {2019},
}
@book{roe2019chapter,
  title = {Quokka Budgets and Municipal Finance: The Long Chapter},
  author = {Roe, Richard},
  year = {2019},
}
@article{lost2001gone,
  title = {A Paper Nobody Has Downloaded},
  author = {Lost, Lee},
  year = {2001},
}
"""

TRUST_TEXT = "Civic Trust in Northern Cities and Towns\nby Jane Smith\nWe study trust."
ZEBRA_TEXT = "Zebrafish Governance Systems in Practice\nJohn Doe\nBody text."
QUOKKA_TEXT = "Quokka Budgets and Municipal Finance\nRichard Roe\nBody text."
OTHER_TEXT = "Walrus Migration in Polar Seas\nSomebody Else\nBody text about walruses."


@pytest.fixture
def bib(tmp_path):
    path = tmp_path / "prime.bib"
    path.write_text(BIB, encoding="utf-8")
    return path


@pytest.fixture
def entries(bib):
    return parse_bib_file(bib)


@pytest.fixture
def db(tmp_path):
    conn, _ = new_db(tmp_path)
    return conn


def _layout(conn, pid, title, confidence="high"):
    conn.execute("UPDATE papers SET title_from_layout = ?, title_layout_confidence = ?, "
                 "title_layout_at = 'x' WHERE id = ?", (title, confidence, pid))
    conn.commit()


def _set(conn, pid, **cols):
    sets = ", ".join(f"{k} = ?" for k in cols)
    conn.execute(f"UPDATE papers SET {sets} WHERE id = ?", [*cols.values(), pid])
    conn.commit()


def _row(conn, pid):
    return conn.execute("SELECT * FROM papers WHERE id = ?", (pid,)).fetchone()


# --- parsing and cleaning ---------------------------------------------------


@pytest.mark.parametrize("raw,clean", [
    (r"{AI} for {Science}", "AI for Science"),
    (r"Law \& Order", "Law & Order"),
    (r"``Fake News'' Law", '"Fake News" Law'),
    (r"`Infodemic' Responses", "'Infodemic' Responses"),
    (r"Caf\'e and na{\"i}ve", "Café and naïve"),
    (r"Ond{\v{r}}ej", "Ondřej"),
    (r"\c{C}elik", "Çelik"),
    (r"\textit{Italic} Claim", "Italic Claim"),
    (r"[C] Measuring Persuasion", "Measuring Persuasion"),
    (r"100\% of\_it", "100% of_it"),
    (r"Stra{\ss}e", "Straße"),
])
def test_clean_bib_text(raw, clean):
    assert clean_bib_text(raw) == clean


def test_parse_keeps_nonstandard_types_and_reads_abstract_and_date(entries):
    by = {e.citekey: e for e in entries}
    assert by["doe2021web"].entry_type == "online"
    assert by["doe2021web"].year == 2021
    assert by["smith2020trust"].abstract == "We study trust."
    assert by["smith2020trust"].title == 'Civic Trust in Northern Cities & "Towns"'


def test_parse_survives_undecodable_bytes(tmp_path):
    path = tmp_path / "bad.bib"
    path.write_bytes(b"@article{a, title = {Ok \xff Title}, year = {2020}}\n")
    assert parse_bib_file(path)[0].citekey == "a"


def test_split_authors_respects_braces_and_drops_others():
    assert bibsync.split_authors(
        "Smith, Jane and {World Bank and Partners} and\n  Caf\\'e, Ren\\'e and others"
    ) == ["Smith, Jane", "World Bank and Partners", "Café, René"]


# --- match_by_title fallback ------------------------------------------------


def test_match_by_title_falls_back_to_layout_but_stays_a_hint(entries):
    papers = [{"id": 1, "filename": "x.pdf", "title": None, "year": 2021,
               "title_from_layout": "Zebrafish Governance Systems in Practice"}]
    (m,) = match_by_title(papers, entries)
    assert m.citekey == "doe2021web" and m.paper_title.startswith("Zebrafish")
    # An exact match on an unverified title is not "certain", year or no year.
    assert m.confidence == "uncertain"


def test_match_by_title_prefers_the_verified_title(entries):
    papers = [{"id": 1, "filename": "x.pdf", "title": "Walrus Migration", "year": None,
               "title_from_layout": "Zebrafish Governance Systems in Practice"}]
    assert match_by_title(papers, entries) == []


def test_reconcile_sees_layout_titles(db, entries):
    add_paper(db, 1, "x.pdf", text=ZEBRA_TEXT)
    _layout(db, 1, "Zebrafish Governance Systems in Practice")
    matches, _ = reconcile(db, entries)
    assert [m.citekey for m in matches] == ["doe2021web"]


# --- suggest ----------------------------------------------------------------


def test_suggest_title_gate_with_evidence(db, entries):
    add_paper(db, 1, "scan0001.pdf", text=ZEBRA_TEXT)
    _layout(db, 1, "Zebrafish Governance Systems in Practice")
    report = bibsync.suggest_links(db, entries)
    (s,) = report["suggestions"]
    assert s["id"] == 1 and s["citekey"] == "doe2021web" and s["tier"] == "title"
    assert s["evidence"]["title_similarity"] == 1.0
    assert s["evidence"]["containment"] == 1.0
    assert s["citekey_taken_by"] is None


def test_suggest_filename_gate_needs_title_in_head(db, entries):
    add_paper(db, 1, "Doe 2021.pdf", text=ZEBRA_TEXT)
    add_paper(db, 2, "Doe 2021 other.pdf", text=OTHER_TEXT)
    report = bibsync.suggest_links(db, entries)
    assert [(s["id"], s["tier"]) for s in report["suggestions"]] == [(1, "filename")]


def test_suggest_both_gates_is_strong(db, entries):
    add_paper(db, 1, "Doe 2021.pdf", text=ZEBRA_TEXT)
    _layout(db, 1, "Zebrafish Governance Systems in Practice")
    assert bibsync.suggest_links(db, entries)["suggestions"][0]["tier"] == "strong"


def test_suggest_low_confidence_layout_is_downgraded(db, entries):
    add_paper(db, 1, "scan.pdf", text=ZEBRA_TEXT)
    _layout(db, 1, "Zebrafish Governance Systems in Practice", "low")
    (s,) = bibsync.suggest_links(db, entries)["suggestions"]
    assert s["tier"] == "weak" and s["evidence"]["layout_confidence"] == "low"


def test_suggest_title_must_be_in_the_document(db, entries):
    add_paper(db, 1, "scan.pdf", text=OTHER_TEXT)
    _layout(db, 1, "Zebrafish Governance Systems in Practice")
    assert bibsync.suggest_links(db, entries)["suggestions"] == []


def test_suggest_skips_titled_and_linked_papers(db, entries):
    add_paper(db, 1, "a.pdf", text=ZEBRA_TEXT, title="Has A Title")
    add_paper(db, 2, "b.pdf", text=ZEBRA_TEXT, citekey="somekey")
    _layout(db, 1, "Zebrafish Governance Systems in Practice")
    _layout(db, 2, "Zebrafish Governance Systems in Practice")
    assert bibsync.suggest_links(db, entries)["suggestions"] == []


def test_suggest_reports_taken_key_and_siblings(db, entries):
    add_paper(db, 1, "owner.pdf", text=QUOKKA_TEXT, title="Quokka", citekey="roe2019book")
    add_paper(db, 2, "Roe 2019.pdf",
              text=QUOKKA_TEXT + " The Long Chapter")
    _layout(db, 2, "Quokka Budgets and Municipal Finance")
    (s,) = bibsync.suggest_links(db, entries)["suggestions"]
    assert s["citekey"] == "roe2019book" and s["citekey_taken_by"] == 1
    assert [a["citekey"] for a in s["alternates"]] == ["roe2019chapter"]
    assert s["ambiguous"] is True


def test_apply_assigns_only_named_ids(db, entries):
    add_paper(db, 1, "a.pdf", text=ZEBRA_TEXT)
    add_paper(db, 2, "b.pdf", text=QUOKKA_TEXT)
    _layout(db, 1, "Zebrafish Governance Systems in Practice")
    _layout(db, 2, "Quokka Budgets and Municipal Finance")
    report = bibsync.apply_suggestions(db, entries, [1])
    assert [a["id"] for a in report["applied"]] == [1]
    assert _row(db, 1)["citekey"] == "doe2021web"
    assert _row(db, 1)["citekey_source"] == "suggest:reviewed"
    assert _row(db, 2)["citekey"] is None


def test_apply_refuses_taken_citekey(db, entries):
    add_paper(db, 1, "owner.pdf", text=ZEBRA_TEXT, title="T", citekey="DOE2021WEB")
    add_paper(db, 2, "b.pdf", text=ZEBRA_TEXT)
    _layout(db, 2, "Zebrafish Governance Systems in Practice")
    report = bibsync.apply_suggestions(db, entries, [2])
    assert report["applied"] == []
    assert report["refused"] == [{"id": 2, "citekey": "doe2021web", "taken_by": 1}]
    assert _row(db, 2)["citekey"] is None


def test_apply_two_ids_for_one_key_assigns_it_once(db, entries):
    for pid in (1, 2):
        add_paper(db, pid, f"{pid}.pdf", text=ZEBRA_TEXT)
        _layout(db, pid, "Zebrafish Governance Systems in Practice")
    report = bibsync.apply_suggestions(db, entries, [1, 2])
    assert [a["id"] for a in report["applied"]] == [1]
    assert report["refused"][0]["taken_by"] == 1


def test_apply_reports_ids_with_no_suggestion_and_dry_run_writes_nothing(db, entries):
    add_paper(db, 1, "a.pdf", text=ZEBRA_TEXT)
    add_paper(db, 2, "b.pdf", text=OTHER_TEXT)
    _layout(db, 1, "Zebrafish Governance Systems in Practice")
    report = bibsync.apply_suggestions(db, entries, [1, 2], dry_run=True)
    assert report["no_suggestion"] == [2] and len(report["applied"]) == 1
    assert _row(db, 1)["citekey"] is None


# --- audit ------------------------------------------------------------------


def _wrong_link(db):
    add_paper(db, 1, "w.pdf", text=OTHER_TEXT, title="Walrus Migration in Polar Seas",
              citekey="doe2021web")
    _set(db, 1, citekey_source="filename")


def test_audit_flags_bib_title_absent_stored_title_present(db, entries):
    _wrong_link(db)
    add_paper(db, 2, "ok.pdf", text=ZEBRA_TEXT, title="Zebrafish", citekey="doe2021web")
    report = bibsync.audit_links(db, entries)
    (f,) = report["flagged"]
    assert f["id"] == 1 and f["citekey"] == "doe2021web" and f["basis"] == "stored"
    assert f["citekey_source"] == "filename" and f["bib_containment"] < 0.6


def test_audit_counts_high_confidence_layout_only(db, entries):
    add_paper(db, 1, "w.pdf", text=OTHER_TEXT, citekey="doe2021web")
    _layout(db, 1, "Walrus Migration in Polar Seas", "low")
    assert bibsync.audit_links(db, entries)["flagged"] == []
    _layout(db, 1, "Walrus Migration in Polar Seas", "high")
    assert bibsync.audit_links(db, entries)["flagged"][0]["basis"] == "layout"


def test_audit_ignores_papers_without_text(db, entries):
    add_paper(db, 1, "w.pdf", text="", title="Whatever", citekey="doe2021web")
    assert bibsync.audit_links(db, entries)["flagged"] == []


def test_audit_clear_records_the_old_key(db, entries, tmp_path):
    _wrong_link(db)
    log = tmp_path / "clears.jsonl"
    report = bibsync.clear_audited(db, entries, [1], log_path=log)
    assert report["cleared"][0]["citekey"] == "doe2021web"
    row = _row(db, 1)
    assert row["citekey"] is None and row["citekey_source"] is None
    logged = json.loads(log.read_text().splitlines()[0])
    assert logged["id"] == 1 and logged["citekey"] == "doe2021web"
    assert logged["citekey_source"] == "filename" and logged["at"]


def test_audit_clear_refuses_unflagged_ids_and_dry_run_is_inert(db, entries, tmp_path):
    _wrong_link(db)
    add_paper(db, 2, "ok.pdf", text=ZEBRA_TEXT, title="Zebrafish", citekey="doe2021web")
    log = tmp_path / "clears.jsonl"
    report = bibsync.clear_audited(db, entries, [1, 2], log_path=log, dry_run=True)
    assert report["not_flagged"] == [2] and [c["id"] for c in report["cleared"]] == [1]
    assert _row(db, 1)["citekey"] == "doe2021web" and not log.exists()


# --- pull -------------------------------------------------------------------


def _trust_paper(db, **cols):
    add_paper(db, 1, "t.pdf", text=TRUST_TEXT, citekey="smith2020trust")
    if cols:
        _set(db, 1, **cols)


def test_pull_fills_an_untitled_paper(db, entries):
    _trust_paper(db, metadata_suspect=1)
    report = bibsync.pull_metadata(db, entries, "prime")
    row = _row(db, 1)
    assert row["title"] == 'Civic Trust in Northern Cities & "Towns"'
    assert json.loads(row["authors"]) == ["Smith, Jane", "World Bank and Partners", "Café, René"]
    assert row["year"] == 2020 and row["doi"] == "10.1000/abc"
    assert row["abstract"] == "We study trust."
    assert row["metadata_source"] == "bib:prime" and row["metadata_suspect"] == 0
    assert "T" in row["metadata_enriched_at"] and "T" in row["metadata_verified_at"]
    assert report["newly_titled_ids"] == [1] and report["updated_ids"] == [1]


def test_pull_reaches_keyword_search(db, entries):
    add_paper(db, 1, "t.pdf", text="Zebrafish Governance Systems in Practice",
              citekey="doe2021web")
    _set(db, 1, abstract=None)
    db.execute("UPDATE papers SET authors = NULL WHERE id = 1")
    bibsync.pull_metadata(db, entries, "prime")
    assert db.execute("SELECT rowid FROM papers_fts WHERE papers_fts MATCH "
                      "'authors : doe'").fetchall()


def test_pull_replaces_provider_values_but_never_blanks(db, entries):
    add_paper(db, 1, "z.pdf", text=ZEBRA_TEXT, title="zebrafish governance", year=1999,
              citekey="doe2021web")
    _set(db, 1, metadata_source="openalex", abstract="Provider abstract", doi="10.9/x")
    bibsync.pull_metadata(db, entries, "prime")
    row = _row(db, 1)
    assert row["title"] == "Zebrafish Governance Systems in Practice"
    assert row["year"] == 2021
    assert row["abstract"] == "Provider abstract" and row["doi"] == "10.9/x"


def test_pull_skips_hand_set_rows(db, entries):
    _trust_paper(db, title="Hand Title", metadata_source="manual:cli")
    report = bibsync.pull_metadata(db, entries, "prime")
    assert report["skipped"]["manual"] == [1] and _row(db, 1)["title"] == "Hand Title"


def test_pull_skips_when_bib_title_not_in_document(db, entries):
    add_paper(db, 1, "w.pdf", text=OTHER_TEXT, citekey="doe2021web")
    report = bibsync.pull_metadata(db, entries, "prime")
    assert report["skipped"]["unverified"] == [1]
    assert _row(db, 1)["title"] is None and _row(db, 1)["metadata_source"] is None


def test_pull_dry_run_ids_and_idempotence(db, entries):
    _trust_paper(db)
    add_paper(db, 2, "z.pdf", text=ZEBRA_TEXT, citekey="doe2021web")
    dry = bibsync.pull_metadata(db, entries, "prime", dry_run=True)
    assert dry["updated_ids"] == [1, 2] and dry["newly_titled_ids"] == [1, 2]
    assert _row(db, 1)["title"] is None

    assert bibsync.pull_metadata(db, entries, "prime", paper_ids=[2])["updated_ids"] == [2]
    assert _row(db, 1)["title"] is None

    bibsync.pull_metadata(db, entries, "prime")
    stamp = _row(db, 1)["updated_at"]
    again = bibsync.pull_metadata(db, entries, "prime")
    assert again["updated_ids"] == [] and sorted(again["skipped"]["no_change"]) == [1, 2]
    assert _row(db, 1)["updated_at"] == stamp


def test_pull_reports_keys_missing_from_the_bib(db, entries):
    add_paper(db, 1, "x.pdf", text="x", citekey="notinbib")
    assert bibsync.pull_metadata(db, entries, "prime")["skipped"]["not_in_bib"] == [1]


# --- metadata_source consumers ----------------------------------------------


def _bib_row(db, entries):
    add_paper(db, 1, "z.pdf", text=ZEBRA_TEXT, citekey="doe2021web")
    bibsync.pull_metadata(db, entries, "prime")


def test_enrich_never_overwrites_bib_rows(db, entries, monkeypatch):
    _bib_row(db, entries)
    monkeypatch.setattr("pdf_gantry.openalex.fetch_by_doi", lambda *a, **k: 1 / 0)
    monkeypatch.setattr("pdf_gantry.openalex.fetch_by_title", lambda *a, **k: 1 / 0)
    monkeypatch.setattr("pdf_gantry.openalex.fetch_by_filename", lambda *a, **k: 1 / 0)
    stats = enrich_documents(db, paper_ids=[1])
    assert stats.skipped_bib == 1 and stats.total == 0
    assert _row(db, 1)["metadata_source"] == "bib:prime"
    assert enrich_documents(db, retry_misses=True).total == 0


def test_bib_named_manual_is_not_hand_set(db, entries):
    add_paper(db, 1, "z.pdf", text=ZEBRA_TEXT, citekey="doe2021web")
    bibsync.pull_metadata(db, entries, "manual-refs")
    assert _row(db, 1)["metadata_source"] == "bib:manual-refs"
    assert db.execute(f"SELECT COUNT(*) FROM papers WHERE {manual_condition()}").fetchone()[0] == 0
    assert 1 in bibsync.pull_metadata(db, entries, "manual-refs")["skipped"]["no_change"]


def test_verify_passes_bib_rows(db, entries):
    _bib_row(db, entries)
    (r,) = verify_documents(db)
    assert r.suspect is False and _row(db, 1)["metadata_suspect"] == 0


def test_gaps_status_and_queue_treat_bib_rows_as_sourced(db, entries):
    _bib_row(db, entries)
    assert field_completeness(db)["enrich_misses"] == 0
    counts = pending_counts(db)
    assert counts["bib_metadata"] == 1 and counts["needs_enrich"] == 0
    assert counts["enrich_misses"] == 0 and counts["manual_metadata"] == 0


# --- missing ----------------------------------------------------------------


def test_missing_matches_by_citekey_file_and_doi(db, entries):
    add_paper(db, 1, "a.pdf", text="", citekey="roe2019book")
    add_paper(db, 2, "SMITH 2020.PDF", text="")
    add_paper(db, 3, "c.pdf", text="")
    _set(db, 3, doi="10.1000/x")
    rows = bibsync.missing_entries(db, entries)
    assert [r["citekey"] for r in rows] == ["doe2021web", "lost2001gone", "roe2019chapter"]
    assert rows[1] == {
        "citekey": "lost2001gone", "type": "article", "year": 2001, "author": "Lost, Lee",
        "title": "A Paper Nobody Has Downloaded", "doi": None,
    }


def test_missing_doi_match_is_normalized(db, entries):
    add_paper(db, 1, "a.pdf", text="")
    _set(db, 1, doi="10.1000/abc")
    assert "smith2020trust" not in [r["citekey"] for r in bibsync.missing_entries(db, entries)]


# --- CLI --------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, monkeypatch, bib):
    conn, db_path = new_db(tmp_path)
    point_cli_at(tmp_path, monkeypatch, db_path)
    return conn, bib, db_path


def _run(*args):
    return CliRunner().invoke(cli, [str(a) for a in args])


def test_cli_suggest_and_apply(cli_env):
    conn, bib, _ = cli_env
    add_paper(conn, 1, "a.pdf", text=ZEBRA_TEXT)
    _layout(conn, 1, "Zebrafish Governance Systems in Practice")
    report = json.loads(_run("link", "suggest", bib, "--json").output)
    assert report["suggestions"][0]["citekey"] == "doe2021web"
    assert report["by_tier"] == {"title": 1}
    assert "doe2021web" in _run("link", "suggest", bib).output
    usage = _run("link", "suggest", bib, "--apply")
    assert usage.exit_code == 64
    applied = json.loads(_run("link", "suggest", bib, "--apply", "--ids", "1", "--json").output)
    assert applied["applied"][0]["citekey"] == "doe2021web"


def test_cli_apply_refusal_exits_partial(cli_env):
    conn, bib, _ = cli_env
    add_paper(conn, 1, "o.pdf", text=ZEBRA_TEXT, title="T", citekey="doe2021web")
    add_paper(conn, 2, "a.pdf", text=ZEBRA_TEXT)
    _layout(conn, 2, "Zebrafish Governance Systems in Practice")
    assert _run("link", "suggest", bib, "--apply", "--ids", "2").exit_code == 1
    add_paper(conn, 3, "b.pdf", text=QUOKKA_TEXT)
    _layout(conn, 3, "Quokka Budgets and Municipal Finance: The Long Chapter")
    assert _run("link", "suggest", bib, "--apply", "--ids", "2,3").exit_code == 3


def test_cli_audit_and_clear_log_under_index_dir(cli_env):
    conn, bib, db_path = cli_env
    _wrong_link(conn)
    report = json.loads(_run("link", "audit", bib, "--json").output)
    assert report["flagged"][0]["id"] == 1
    usage = _run("link", "audit", bib, "--clear")
    assert usage.exit_code == 64
    out = json.loads(_run("link", "audit", bib, "--clear", "--ids", "1", "--json").output)
    assert out["cleared"][0]["citekey"] == "doe2021web"
    assert (db_path.parent / "citekey-clears.jsonl").exists()


def test_cli_pull(cli_env):
    conn, bib, _ = cli_env
    add_paper(conn, 1, "z.pdf", text=ZEBRA_TEXT, citekey="doe2021web")
    dry = json.loads(_run("link", "pull", bib, "--dry-run", "--json").output)
    assert dry["dry_run"] is True and dry["newly_titled_ids"] == [1]
    out = _run("link", "pull", bib)
    assert "embed --chunk --ids 1 --force" in out.output
    assert _row(conn, 1)["metadata_source"] == "bib:prime"


def test_cli_missing_json_and_tsv(cli_env):
    conn, bib, _ = cli_env
    report = json.loads(_run("link", "missing", bib, "--json").output)
    assert report["count"] == 5
    tsv = _run("link", "missing", bib, "--format", "tsv").output.splitlines()
    assert tsv[0].split("\t") == ["citekey", "type", "year", "author", "title", "doi"]
    assert len(tsv) == 6 and all(len(ln.split("\t")) == 6 for ln in tsv)
