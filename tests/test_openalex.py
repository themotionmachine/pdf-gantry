"""Tests for the OpenAlex metadata provider (issue #20)."""

from pdf_gantry import openalex

# --- reconstruct_abstract ---


def test_reconstruct_abstract_orders_words():
    inverted = {"Climate": [0], "policy": [1], "matters": [2]}
    assert openalex.reconstruct_abstract(inverted) == "Climate policy matters"


def test_reconstruct_abstract_handles_repeats():
    inverted = {"the": [0, 2], "cat": [1], "sat": [3]}
    assert openalex.reconstruct_abstract(inverted) == "the cat the sat"


def test_reconstruct_abstract_none():
    assert openalex.reconstruct_abstract(None) is None
    assert openalex.reconstruct_abstract({}) is None


# --- parse_openalex_work ---


def _sample_work():
    return {
        "id": "https://openalex.org/W123",
        "title": "Climate Policy in the Digital Age",
        "display_name": "Climate Policy in the Digital Age",
        "publication_year": 2024,
        "doi": "https://doi.org/10.1234/climate.2024",
        "authorships": [
            {"author": {"display_name": "John Smith"}},
            {"author": {"display_name": "Jane Doe"}},
        ],
        "abstract_inverted_index": {"A": [0], "study": [1]},
    }


def test_parse_openalex_work_maps_fields():
    meta = openalex.parse_openalex_work(_sample_work())
    assert meta["title"] == "Climate Policy in the Digital Age"
    assert meta["year"] == 2024
    assert meta["authors"] == ["John Smith", "Jane Doe"]
    assert meta["abstract"] == "A study"
    assert meta["source_id"] == "https://openalex.org/W123"


def test_parse_openalex_work_strips_doi_url():
    meta = openalex.parse_openalex_work(_sample_work())
    assert meta["doi"] == "10.1234/climate.2024"


def test_parse_openalex_work_none():
    assert openalex.parse_openalex_work(None) is None


def test_parse_openalex_work_falls_back_to_display_name():
    work = {"display_name": "Fallback Title", "authorships": []}
    meta = openalex.parse_openalex_work(work)
    assert meta["title"] == "Fallback Title"
    assert meta["authors"] == []


def test_parse_openalex_work_handles_null_author_entry():
    """OpenAlex sometimes returns `authorships` entries with `"author": null`
    (group/consortium authorships, deleted/merged author records). The API
    contract doesn't forbid this, so a real response can carry it. One bad
    entry should not nuke a title/DOI/abstract we already paid a network
    round-trip for — it should just be dropped from the author list."""
    work = {
        "id": "https://openalex.org/W999",
        "title": "Group Authorship Study",
        "display_name": "Group Authorship Study",
        "publication_year": 2023,
        "doi": "https://doi.org/10.1234/group.2023",
        "authorships": [
            {"author": {"display_name": "Real Person"}},
            {"author": None},
        ],
        "abstract_inverted_index": {"A": [0], "study": [1]},
    }
    meta = openalex.parse_openalex_work(work)
    assert meta is not None
    assert meta["title"] == "Group Authorship Study"
    assert meta["authors"] == ["Real Person"]
    assert meta["doi"] == "10.1234/group.2023"
    assert meta["abstract"] == "A study"


def test_parse_openalex_work_missing_author_key_entirely():
    """Same trust boundary, adjacent shape: an authorship dict that omits the
    "author" key altogether rather than nulling it. Must not crash either."""
    work = {"display_name": "No Author Key", "authorships": [{}]}
    meta = openalex.parse_openalex_work(work)
    assert meta["authors"] == []


# --- parse_filename ---


def test_parse_filename_underscore_pattern():
    assert openalex.parse_filename("Friston_2010_Free-Energy.pdf") == ("friston", 2010)


def test_parse_filename_hyphen_pattern():
    assert openalex.parse_filename("wagner-payne-2017-trends-in.pdf") == ("wagner", 2017)


def test_parse_filename_no_match():
    assert openalex.parse_filename("random_document.pdf") == (None, None)


# --- fetch_by_doi (httpx wiring) ---


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_fetch_by_doi_returns_normalized(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResp(200, _sample_work())

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)

    meta = openalex.fetch_by_doi("10.1234/climate.2024", mailto="ryan@example.com")
    assert meta["title"] == "Climate Policy in the Digital Age"
    assert "doi:10.1234/climate.2024" in captured["url"]
    assert captured["params"]["mailto"] == "ryan@example.com"


def test_fetch_by_doi_404_returns_none(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp(404, {}))
    assert openalex.fetch_by_doi("10.0/nope") is None


def test_fetch_by_title_uses_search(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["params"] = params
        return _FakeResp(200, {"results": [_sample_work()]})

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)

    meta = openalex.fetch_by_title("Climate Policy")
    assert meta["year"] == 2024
    assert captured["params"]["search"] == "Climate Policy"


def test_fetch_by_title_empty_results(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp(200, {"results": []}))
    assert openalex.fetch_by_title("nothing here") is None
