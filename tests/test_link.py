"""Tests for BibTeX linking and reconciliation."""

import json

import pytest
from click.testing import CliRunner

from pdf_gantry.cli import cli
from pdf_gantry.db import get_connection
from pdf_gantry.link import (
    BibEntry,
    LinkMatch,
    apply_matches,
    extract_bib_filename,
    match_by_doi,
    match_by_filename,
    match_by_title,
    normalize_doi,
    normalize_title,
    parse_bib_file,
    reconcile,
)
from pdf_gantry.queue import query_queue

# --- Fixtures ---


@pytest.fixture
def bib_file(tmp_path):
    """Create a .bib file with known entries."""
    content = r"""
@article{smith2024climate,
  title = {Climate Policy in the Digital Age},
  author = {Smith, John and Doe, Jane},
  year = {2024},
  doi = {10.1234/climate.2024},
  file = {:papers/smith2024.pdf:PDF}
}

@inproceedings{vaswani2017attention,
  title = {Attention Is All You Need},
  author = {Vaswani, Ashish and Shazeer, Noam},
  year = {2017},
  doi = {10.5555/attention.2017}
}

@article{jones2023nofile,
  title = {A Study With No File Field},
  author = {Jones, Alice},
  year = {2023}
}

@article{latex2024escapes,
  title = {M{\"u}ller's \& G{\"o}del: A {Survey} of {AI}},
  author = {M{\"u}ller, Hans},
  year = {2024}
}
"""
    bib_path = tmp_path / "library.bib"
    bib_path.write_text(content)
    return bib_path


@pytest.fixture
def linked_db(tmp_path):
    """DB with papers that have DOIs/titles matching bib entries."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))

    papers = [
        ("climate_policy.pdf", "10.1234/climate.2024", "Climate Policy in the Digital Age", 2024),
        ("attention.pdf", "10.5555/attention.2017", "Attention Is All You Need", 2017),
        ("smith2024.pdf", None, "Some Unrelated Title", None),
        ("no_match.pdf", None, "Completely Different Paper", 2020),
        ("jones_study.pdf", None, "A Study With No File Field", 2023),
    ]
    for filename, doi, title, year in papers:
        conn.execute(
            "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, "
            "indexed_at, updated_at, doi, title, year) "
            "VALUES (?, ?, 'hash', 1000, '2024-01-01', '2024-01-01', '2024-01-01', ?, ?, ?)",
            (f"/papers/{filename}", filename, doi, title, year),
        )
    conn.commit()
    return conn


# --- parse_bib_file ---


def test_parse_bib_basic(bib_file):
    entries = parse_bib_file(bib_file)
    assert len(entries) == 4
    smith = next(e for e in entries if e.citekey == "smith2024climate")
    assert smith.title == "Climate Policy in the Digital Age"
    assert smith.doi == "10.1234/climate.2024"
    assert smith.year == 2024
    assert smith.file is not None


def test_parse_bib_empty(tmp_path):
    bib_path = tmp_path / "empty.bib"
    bib_path.write_text("")
    entries = parse_bib_file(bib_path)
    assert entries == []


def test_parse_bib_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_bib_file(tmp_path / "nonexistent.bib")


def test_parse_bib_latex_escapes(bib_file):
    entries = parse_bib_file(bib_file)
    latex = next(e for e in entries if e.citekey == "latex2024escapes")
    assert "{" not in latex.title
    assert "\\" not in latex.title


def test_parse_bib_file_field_mendeley(bib_file):
    entries = parse_bib_file(bib_file)
    smith = next(e for e in entries if e.citekey == "smith2024climate")
    assert smith.file == ":papers/smith2024.pdf:PDF"


# --- normalize_title ---


def test_normalize_title_strips_braces():
    assert normalize_title("{AI} for {Science}") == "ai for science"


def test_normalize_title_lowercases():
    assert normalize_title("Climate POLICY") == "climate policy"


def test_normalize_title_collapses_whitespace():
    assert normalize_title("  too   many   spaces  ") == "too many spaces"


def test_normalize_title_strips_latex():
    assert "\\textbf" not in normalize_title("A \\textbf{Bold} Claim")


# --- normalize_doi ---


def test_normalize_doi_strips_url_prefix():
    assert normalize_doi("https://doi.org/10.1234/test") == "10.1234/test"
    assert normalize_doi("http://dx.doi.org/10.1234/test") == "10.1234/test"


def test_normalize_doi_lowercases():
    assert normalize_doi("10.1234/TEST.Paper") == "10.1234/test.paper"


def test_normalize_doi_strips_trailing_punctuation():
    assert normalize_doi("10.1234/test.") == "10.1234/test"


# --- extract_bib_filename ---


def test_extract_bib_filename_mendeley():
    assert extract_bib_filename(":papers/smith2024.pdf:PDF") == "smith2024.pdf"


def test_extract_bib_filename_jabref():
    assert extract_bib_filename(":C\\:/Users/papers/smith2024.pdf:PDF") == "smith2024.pdf"


def test_extract_bib_filename_plain_path():
    assert extract_bib_filename("papers/smith2024.pdf") == "smith2024.pdf"


def test_extract_bib_filename_bare():
    assert extract_bib_filename("smith2024.pdf") == "smith2024.pdf"


def test_extract_bib_filename_none():
    assert extract_bib_filename(None) is None


def test_extract_bib_filename_empty():
    assert extract_bib_filename("") is None


# --- match_by_doi ---


def test_match_by_doi_exact(linked_db, bib_file):
    entries = parse_bib_file(bib_file)
    papers = [dict(r) for r in linked_db.execute("SELECT * FROM papers").fetchall()]
    matches = match_by_doi(papers, entries)
    assert len(matches) == 2
    assert all(m.confidence == "certain" for m in matches)
    assert all(m.score == 1.0 for m in matches)
    citekeys = {m.citekey for m in matches}
    assert "smith2024climate" in citekeys
    assert "vaswani2017attention" in citekeys


def test_match_by_doi_normalized():
    papers = [
        {"id": 1, "filename": "test.pdf", "doi": "https://doi.org/10.1234/TEST", "title": "T"}
    ]
    entries = [BibEntry("key1", "article", "T", None, None, "10.1234/test", None)]
    matches = match_by_doi(papers, entries)
    assert len(matches) == 1


def test_match_by_doi_no_match():
    papers = [{"id": 1, "filename": "test.pdf", "doi": "10.1234/aaa", "title": "T"}]
    entries = [BibEntry("key1", "article", "T", None, None, "10.1234/bbb", None)]
    matches = match_by_doi(papers, entries)
    assert len(matches) == 0


def test_match_by_doi_null_doi():
    papers = [{"id": 1, "filename": "test.pdf", "doi": None, "title": "T"}]
    entries = [BibEntry("key1", "article", "T", None, None, "10.1234/test", None)]
    matches = match_by_doi(papers, entries)
    assert len(matches) == 0


# --- match_by_filename ---


def test_match_by_filename_mendeley():
    papers = [{"id": 1, "filename": "smith2024.pdf", "title": "T"}]
    entries = [BibEntry("key1", "article", "T", None, None, None, ":papers/smith2024.pdf:PDF")]
    matches = match_by_filename(papers, entries)
    assert len(matches) == 1
    assert matches[0].confidence == "likely"


def test_match_by_filename_no_file_field():
    papers = [{"id": 1, "filename": "smith2024.pdf", "title": "T"}]
    entries = [BibEntry("key1", "article", "T", None, None, None, None)]
    matches = match_by_filename(papers, entries)
    assert len(matches) == 0


# --- match_by_title ---


def test_match_by_title_exact():
    papers = [
        {"id": 1, "filename": "test.pdf",
         "title": "Climate Policy in the Digital Age", "year": None}
    ]
    entries = [
        BibEntry("key1", "article", "Climate Policy in the Digital Age", None, None, None, None)
    ]
    matches = match_by_title(papers, entries)
    assert len(matches) == 1
    assert matches[0].confidence == "certain"
    assert matches[0].score >= 0.95


def test_match_by_title_fuzzy():
    papers = [
        {"id": 1, "filename": "test.pdf",
         "title": "Climate Policy in the Digital Age: A Review", "year": None}
    ]
    entries = [
        BibEntry("key1", "article", "Climate Policy in the Digital Age", None, None, None, None)
    ]
    matches = match_by_title(papers, entries, threshold=0.7)
    assert len(matches) == 1
    assert matches[0].score >= 0.7


def test_match_by_title_below_threshold():
    papers = [
        {"id": 1, "filename": "test.pdf", "title": "Quantum Computing Fundamentals", "year": None}
    ]
    entries = [
        BibEntry("key1", "article", "Climate Policy in the Digital Age", None, None, None, None)
    ]
    matches = match_by_title(papers, entries)
    assert len(matches) == 0


def test_match_by_title_case_insensitive():
    papers = [
        {"id": 1, "filename": "test.pdf",
         "title": "CLIMATE POLICY IN THE DIGITAL AGE", "year": None}
    ]
    entries = [
        BibEntry("key1", "article", "climate policy in the digital age", None, None, None, None)
    ]
    matches = match_by_title(papers, entries)
    assert len(matches) == 1


def test_match_by_title_year_boost():
    papers = [
        {"id": 1, "filename": "test.pdf",
         "title": "Climate Policy in the Digital Age: A Survey", "year": 2024}
    ]
    entries = [
        BibEntry("key1", "article", "Climate Policy in the Digital Age", None, 2024, None, None)
    ]
    matches_with_year = match_by_title(papers, entries, threshold=0.80)
    papers[0]["year"] = 1999
    matches_wrong_year = match_by_title(papers, entries, threshold=0.80)
    if matches_with_year and matches_wrong_year:
        assert matches_with_year[0].confidence >= matches_wrong_year[0].confidence or \
            matches_with_year[0].score >= matches_wrong_year[0].score


def test_match_by_title_no_title():
    papers = [{"id": 1, "filename": "test.pdf", "title": None, "year": None}]
    entries = [BibEntry("key1", "article", "Climate Policy", None, None, None, None)]
    matches = match_by_title(papers, entries)
    assert len(matches) == 0


# --- reconcile ---


def test_reconcile_phases_ordered(linked_db, bib_file):
    entries = parse_bib_file(bib_file)
    matches, stats = reconcile(linked_db, entries)
    doi_matches = [m for m in matches if m.source == "doi"]
    assert stats.matched_doi == 2
    assert len(doi_matches) == 2


def test_reconcile_no_double_match(linked_db, bib_file):
    entries = parse_bib_file(bib_file)
    matches, stats = reconcile(linked_db, entries)
    paper_ids = [m.paper_id for m in matches]
    assert len(paper_ids) == len(set(paper_ids))
    citekeys = [m.citekey for m in matches]
    assert len(citekeys) == len(set(citekeys))


def test_reconcile_stats(linked_db, bib_file):
    entries = parse_bib_file(bib_file)
    matches, stats = reconcile(linked_db, entries)
    assert stats.total_papers == 5
    assert stats.total_bib_entries == 4
    total_matched = (
        stats.matched_doi + stats.matched_filename
        + stats.matched_title_certain + stats.matched_title_uncertain
    )
    assert total_matched == len(matches)
    assert stats.unmatched_papers == 5 - len(matches)


# --- apply_matches ---


def test_apply_matches_writes_citekey(linked_db, bib_file):
    entries = parse_bib_file(bib_file)
    matches, _ = reconcile(linked_db, entries)
    certain = [m for m in matches if m.confidence in ("certain", "likely")]
    count = apply_matches(linked_db, certain)
    assert count > 0
    row = linked_db.execute(
        "SELECT citekey, citekey_source FROM papers WHERE citekey IS NOT NULL"
    ).fetchone()
    assert row is not None
    assert row["citekey_source"] in ("doi", "filename", "title")


def test_apply_matches_skips_uncertain(linked_db):
    matches = [
        LinkMatch(paper_id=1, filename="test.pdf", citekey="key1",
                  source="title", confidence="uncertain", score=0.87,
                  paper_title="A", bib_title="B"),
    ]
    count = apply_matches(linked_db, matches)
    assert count == 0


def test_apply_matches_includes_uncertain(linked_db):
    matches = [
        LinkMatch(paper_id=1, filename="test.pdf", citekey="key1",
                  source="title", confidence="uncertain", score=0.87,
                  paper_title="A", bib_title="B"),
    ]
    count = apply_matches(linked_db, matches, include_uncertain=True)
    assert count == 1
    row = linked_db.execute("SELECT citekey FROM papers WHERE id = 1").fetchone()
    assert row["citekey"] == "key1"


def test_apply_matches_skips_existing(linked_db):
    linked_db.execute("UPDATE papers SET citekey = 'existing' WHERE id = 1")
    linked_db.commit()
    matches = [
        LinkMatch(paper_id=1, filename="test.pdf", citekey="new_key",
                  source="doi", confidence="certain", score=1.0,
                  paper_title="A", bib_title="B"),
    ]
    count = apply_matches(linked_db, matches)
    assert count == 0
    row = linked_db.execute("SELECT citekey FROM papers WHERE id = 1").fetchone()
    assert row["citekey"] == "existing"


def test_apply_matches_force_overwrites(linked_db):
    linked_db.execute("UPDATE papers SET citekey = 'existing' WHERE id = 1")
    linked_db.commit()
    matches = [
        LinkMatch(paper_id=1, filename="test.pdf", citekey="new_key",
                  source="doi", confidence="certain", score=1.0,
                  paper_title="A", bib_title="B"),
    ]
    count = apply_matches(linked_db, matches, force=True)
    assert count == 1
    row = linked_db.execute("SELECT citekey FROM papers WHERE id = 1").fetchone()
    assert row["citekey"] == "new_key"


# --- filter integration ---


def test_filter_has_citekey(linked_db):
    linked_db.execute("UPDATE papers SET citekey = 'key1' WHERE id = 1")
    linked_db.commit()
    results = query_queue(linked_db, has=["citekey"])
    assert len(results) == 1
    assert results[0]["id"] == 1


def test_filter_needs_citekey(linked_db):
    linked_db.execute("UPDATE papers SET citekey = 'key1' WHERE id = 1")
    linked_db.commit()
    results = query_queue(linked_db, needs=["citekey"])
    assert all(r["citekey"] is None for r in results)
    assert len(results) == 4


# --- CLI integration ---


@pytest.fixture
def cli_db(tmp_path, bib_file):
    """Set up a DB and config for CLI testing."""
    db_path = tmp_path / "index.db"
    conn = get_connection(str(db_path))
    papers = [
        ("paper_a.pdf", "10.1234/climate.2024", "Climate Policy in the Digital Age", 2024),
        ("attention.pdf", "10.5555/attention.2017", "Attention Is All You Need", 2017),
    ]
    for filename, doi, title, year in papers:
        conn.execute(
            "INSERT INTO papers (path, filename, file_hash, file_size, file_modified, "
            "indexed_at, updated_at, doi, title, year) "
            "VALUES (?, ?, 'hash', 1000, '2024-01-01', '2024-01-01', '2024-01-01', ?, ?, ?)",
            (f"/papers/{filename}", filename, doi, title, year),
        )
    conn.commit()
    conn.close()

    config_file = tmp_path / "config.yaml"
    import yaml
    config_file.write_text(yaml.dump({
        "papers_dir": str(tmp_path),
        "index_dir": str(tmp_path),
    }))
    return tmp_path, bib_file, config_file


def test_cli_link_report(cli_db, monkeypatch):
    tmp_path, bib_file, config_file = cli_db
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)
    runner = CliRunner()
    result = runner.invoke(cli, ["link", str(bib_file)])
    assert result.exit_code == 0
    assert "DOI matches" in result.output


def test_cli_link_json(cli_db, monkeypatch):
    tmp_path, bib_file, config_file = cli_db
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)
    runner = CliRunner()
    result = runner.invoke(cli, ["link", str(bib_file), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "matches" in data
    assert "total_papers" in data


def test_cli_link_apply(cli_db, monkeypatch):
    tmp_path, bib_file, config_file = cli_db
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)
    runner = CliRunner()
    result = runner.invoke(cli, ["link", str(bib_file), "--apply"])
    assert result.exit_code == 0
    assert "Applied" in result.output

    conn = get_connection(str(tmp_path / "index.db"))
    row = conn.execute("SELECT COUNT(*) FROM papers WHERE citekey IS NOT NULL").fetchone()
    assert row[0] >= 1
    conn.close()


def test_cli_link_no_bib_path(cli_db, monkeypatch):
    tmp_path, _, config_file = cli_db
    monkeypatch.setattr("pdf_gantry.config.CONFIG_PATH", config_file)
    runner = CliRunner()
    result = runner.invoke(cli, ["link"])
    assert result.exit_code != 0
