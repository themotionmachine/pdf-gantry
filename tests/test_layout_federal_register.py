"""Federal Register PDFs: the title of the document the PDF is for.

A govinfo PDF of a notice or rule starts on the page where it begins, so
page 1 often ends the previous document (and may hold a whole short one)
before ours starts. Largest-font picked lines from those other documents.
Each document opens with the same bold block: agency, docket line, title,
then the AGENCY: label.
"""

import pymupdf

from pdf_gantry.layout import extract_layout_title

HEADER = "Federal Register / Vol. 89, No. 30 / Tuesday, February 13, 2024 / Notices"


def _fr_pdf(tmp_path, columns, header=HEADER, pages=2):
    """columns: lists of (text, bold); each column is written top to bottom."""
    doc = pymupdf.open()
    page = doc.new_page()
    if header:
        page.insert_text((140, 35), header, fontsize=10, fontname="hebo")
    for c, lines in enumerate(columns):
        y = 60
        for text, bold in lines:
            page.insert_text((45 + c * 180, y), text, fontsize=9,
                             fontname="hebo" if bold else "helv")
            y += 11
    for _ in range(pages - 1):
        doc.new_page().insert_text((45, 60), "Continued text of the notice.", fontsize=9)
    path = tmp_path / "fr.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _notice(title_lines, docket="[Docket No. PTO-P-2023-0043]", extra=()):
    return [
        ("DEPARTMENT OF COMMERCE", True),
        ("Patent and Trademark Office", True),
        *[(line, True) for line in extra],
        (docket, True),
        *[(line, True) for line in title_lines],
        ("AGENCY: United States Patent and", False),
        ("Trademark Office, Department of", False),
        ("Commerce.", False),
    ]


PREVIOUS_TAIL = [
    ("cannot guarantee that we will be able to", False),
    ("do so.", False),
    ("Sheleen Dumas,", True),
    ("Department PRA Clearance Officer.", False),
    ("[FR Doc. 2024-02968 Filed 2-12-24; 8:45 am]", False),
    ("BILLING CODE 3510-22-P", True),
    ("SUPPLEMENTARY INFORMATION: Trip", True),
    ("Ticket Update.", False),
]


def test_title_of_the_notice_after_the_previous_documents_tail(tmp_path):
    path = _fr_pdf(tmp_path, [
        PREVIOUS_TAIL,
        _notice(["Inventorship Guidance for AI-", "Assisted Inventions"]),
    ])
    found = extract_layout_title(path)
    assert found.title == "Inventorship Guidance for AI-Assisted Inventions"
    assert found.confidence == "high"
    assert found.page == 1


def test_last_document_starting_on_page_one_wins(tmp_path):
    path = _fr_pdf(tmp_path, [
        _notice(["Grant of Interim Extension of the Term", "of U.S. Patent No. 8,785,125"])
        + [("[FR Doc. 2025-21411 Filed 11-26-25; 8:45 am]", False)],
        _notice(["Revised Inventorship Guidance for AI-", "Assisted Inventions"]),
    ])
    assert extract_layout_title(path).title == (
        "Revised Inventorship Guidance for AI-Assisted Inventions"
    )


def test_rule_with_cfr_and_rin_lines(tmp_path):
    path = _fr_pdf(tmp_path, [
        PREVIOUS_TAIL,
        _notice(["Setting and Adjusting Patent Fees", "During Fiscal Year 2025"],
                extra=["37 CFR Parts 1 and 41"], docket="RIN 0651-AD72"),
    ])
    assert extract_layout_title(path).title == (
        "Setting and Adjusting Patent Fees During Fiscal Year 2025"
    )


def test_without_the_header_the_general_rule_applies(tmp_path):
    path = _fr_pdf(tmp_path, [
        PREVIOUS_TAIL,
        _notice(["Inventorship Guidance for AI-", "Assisted Inventions"]),
    ], header=None)
    assert extract_layout_title(path).title != (
        "Inventorship Guidance for AI-Assisted Inventions"
    )


def test_header_without_an_agency_label_falls_back(tmp_path, monkeypatch):
    from pdf_gantry import layout

    path = _fr_pdf(tmp_path, [PREVIOUS_TAIL])
    fr = extract_layout_title(path)
    monkeypatch.setattr(layout, "_FEDERAL_REGISTER_RE", layout.re.compile("(?!)"))
    assert fr == extract_layout_title(path)
