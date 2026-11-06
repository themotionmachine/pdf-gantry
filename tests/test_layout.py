"""Layout title extraction from page typography."""

import pymupdf
import pytest

from pdf_gantry.layout import extract_layout_title

BODY = [
    "The body of this paper runs on for many lines at an ordinary reading size,",
    "which is what gives the title its contrast against the rest of the page,",
    "and the extractor measures that contrast rather than trusting line order.",
    "Further sentences repeat the point so the body size clearly dominates here.",
]


def _pdf(tmp_path, pages, name="p.pdf"):
    """pages: list of pages; a page is a list of (y, size, text)."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        for y, size, text in lines:
            page.insert_text((60, y), text, fontsize=size)
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


def _body(y=420):
    return [(y + i * 14, 10, t) for i, t in enumerate(BODY)]


def _title(tmp_path, lines, boilerplate=frozenset()):
    return extract_layout_title(_pdf(tmp_path, [lines + _body()]), boilerplate)


def test_largest_font_wins_over_earlier_masthead(tmp_path):
    r = _title(tmp_path, [(50, 9, "Journal of Things 12 (2020) 1-20"),
                          (120, 20, "A Theory of Everything Useful")])
    assert r.title == "A Theory of Everything Useful"
    assert r.confidence == "high"


def test_wrapped_lines_of_same_size_join_and_whitespace_squeezes(tmp_path):
    r = _title(tmp_path, [(120, 20, "Platform Governance   and the"),
                          (145, 20, "Politics of Moderation")])
    assert r.title == "Platform Governance and the Politics of Moderation"


def test_corpus_boilerplate_is_skipped(tmp_path):
    from pdf_gantry.metadata import normalize_line
    boiler = frozenset({normalize_line("Annual Review of Things")})
    r = _title(tmp_path, [(60, 26, "Annual Review of Things"),
                          (130, 18, "Measuring Civic Trust in Cities")], boiler)
    assert r.title == "Measuring Civic Trust in Cities"


def test_page_two_used_when_page_one_is_empty(tmp_path):
    path = _pdf(tmp_path, [[], [(120, 20, "The Second Page Holds the Title")] + _body()])
    assert extract_layout_title(path).title == "The Second Page Holds the Title"


def test_text_below_the_top_sixty_percent_is_ignored(tmp_path):
    r = _title(tmp_path, [(120, 16, "Civic Trust in Northern Cities"),
                          (700, 30, "A Giant Footer Advertisement")])
    assert r.title == "Civic Trust in Northern Cities"


def test_bad_file_returns_empty_result(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"not a pdf")
    assert extract_layout_title(path).title is None


def test_encrypted_returns_empty_result(encrypted_pdf):
    assert extract_layout_title(encrypted_pdf).title is None


def test_missing_file_raises_so_callers_do_not_stamp_it(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_layout_title(tmp_path / "gone.pdf")


# --- split letters ----------------------------------------------------------


def test_small_caps_spans_join_without_inserted_spaces(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    x = 60
    for text, size in (("E", 24), ("CONOMICS OF ", 18), ("T", 24), ("RUST", 18)):
        page.insert_text((x, 120), text, fontsize=size)
        x += pymupdf.get_text_length(text, fontsize=size)
    for y, size, text in _body():
        page.insert_text((60, y), text, fontsize=size)
    path = tmp_path / "caps.pdf"
    doc.save(str(path))
    doc.close()
    assert extract_layout_title(path).title == "ECONOMICS OF TRUST"


def test_letterspaced_title_collapses(tmp_path):
    r = _title(tmp_path, [(120, 20, "T R U S T  I N  C I T I E S  T O D A Y")])
    assert r.title == "TRUST IN CITIES TODAY"


# --- class 1: wrapped across sizes or gaps ----------------------------------


def test_fragment_ending_in_function_word_continues_over_large_gap(tmp_path):
    r = _title(tmp_path, [(120, 24, "MISINFORMATION AND"),
                          (190, 24, "DEMOCRATIC RESILIENCE")])
    assert r.title == "MISINFORMATION AND DEMOCRATIC RESILIENCE"


def test_fragment_ending_in_function_word_continues_on_smaller_line(tmp_path):
    r = _title(tmp_path, [(120, 24, "The Long Shadow of"),
                          (148, 20, "Colonial Land Registries")])
    assert r.title == "The Long Shadow of Colonial Land Registries"


def test_all_caps_title_continues_on_slightly_smaller_caps_line(tmp_path):
    r = _title(tmp_path, [(120, 24, "SECURITY IN SPACE"),
                          (148, 20, "THE NEXT GENERATION OF TREATIES")])
    assert r.title == "SECURITY IN SPACE THE NEXT GENERATION OF TREATIES"


def test_smaller_author_line_is_not_swallowed(tmp_path):
    r = _title(tmp_path, [(120, 24, "The Long Shadow of Land Registries"),
                          (148, 20, "Jane Q. Public and John Doe")])
    assert r.title == "The Long Shadow of Land Registries"


# --- class 2: run swallows headings -----------------------------------------


@pytest.mark.parametrize("heading", [
    "TITLE 1. INTRODUCTORY PROVISIONS", "CHAPTER I", "ARTICLE 1", "Section 2. Scope",
])
def test_run_stops_at_section_marker(tmp_path, heading):
    r = _title(tmp_path, [(120, 20, "ELECTION CODE"), (145, 20, heading),
                          (170, 20, "General rules of the election")])
    assert r.title == "ELECTION CODE"


# --- class 3: short labels --------------------------------------------------


@pytest.mark.parametrize("label", [
    "Focus Article", "Twelfth edition", "3rd Edition", "Strasbourg, 3 May 2018",
    "January 2021", "Research Paper",
])
def test_label_never_beats_the_title(tmp_path, label):
    r = _title(tmp_path, [(60, 28, label),
                          (130, 18, "Measuring Civic Trust in Northern Cities")])
    assert r.title == "Measuring Civic Trust in Northern Cities"


def test_short_series_name_yields_to_a_full_title_below(tmp_path):
    r = _title(tmp_path, [(60, 30, "DEFENCE STRATEGIC"),
                          (130, 18, "Force Posture After the Northern Treaty Review")])
    assert r.title == "Force Posture After the Northern Treaty Review"


def test_genuine_short_title_survives_with_no_better_candidate(tmp_path):
    r = _title(tmp_path, [(120, 26, "Mind Games")])
    assert r.title == "Mind Games"


# --- class 4: cover sheets --------------------------------------------------


@pytest.mark.parametrize("cover", [
    ["Bluebook 21st ed.", "Citations: Some Law Review 12 (2001)"],
    ["Interlibrary Services", "Document delivery request"],
    ["DATE DOWNLOADED: Tue Mar 1 2022", "SOURCE: Content Downloaded from HeinOnline"],
])
def test_cover_sheet_is_skipped_for_the_next_page(tmp_path, cover):
    first = [(100 + i * 40, 22, t) for i, t in enumerate(cover)]
    second = [(120, 20, "The Real Title of the Article")] + _body()
    assert extract_layout_title(_pdf(tmp_path, [first, second])).title == (
        "The Real Title of the Article"
    )


def test_replacement_characters_are_stripped():
    from pdf_gantry.layout import clean_title
    assert clean_title("Trust� in �Cities") == "Trust in Cities"


# --- class 5: no contrast, boilerplate --------------------------------------


def test_body_paragraph_is_rejected_when_nothing_stands_out(tmp_path):
    path = _pdf(tmp_path, [[(100 + i * 14, 10, t) for i, t in enumerate(BODY * 3)]])
    assert extract_layout_title(path).title is None


def test_contrast_free_short_heading_is_low_confidence(tmp_path):
    lines = [(100, 10, "Civic Trust in Northern Cities")]
    r = extract_layout_title(_pdf(tmp_path, [lines + _body(160)]))
    assert r.title == "Civic Trust in Northern Cities"
    assert r.confidence == "low"


@pytest.mark.parametrize("line", [
    "Copyright 2020 Some Press. All rights reserved.",
    "© 2021 The Authors",
    "Jane Public, jane.public@example.edu",
])
def test_copyright_and_email_lines_are_rejected(tmp_path, line):
    r = _title(tmp_path, [(60, 26, line),
                          (130, 18, "Measuring Civic Trust in Cities")])
    assert r.title == "Measuring Civic Trust in Cities"


# --- class 6: garbled text layers -------------------------------------------


@pytest.mark.parametrize("text", [
    "Xq7z Wkrtp Bvczx Mnpqr Tzzkx",
    "(cid:12)(cid:45)(cid:3)(cid:88)",
    "������ ab",
])
def test_garbled_text_is_not_title_like(text):
    from pdf_gantry.layout import clean_title, looks_garbled
    assert looks_garbled(clean_title(text))


@pytest.mark.parametrize("text", [
    "Strengths of Rhythm in Twelfth Century Schools",
    "Démocratie et confiance",
    "Доверие в городах",
    "NLP for Policy: A Survey",
])
def test_real_titles_are_not_garbled(text):
    from pdf_gantry.layout import looks_garbled
    assert not looks_garbled(text)


def test_garbled_page_returns_null_rather_than_garbage(tmp_path):
    r = _title(tmp_path, [(120, 20, "Xq7z Wkrtp Bvczx Mnpqr Tzzkx")])
    assert r.title is None
