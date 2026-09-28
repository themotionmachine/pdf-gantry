"""`import fitz` prints a deprecation line to STDOUT on PyMuPDF >= 1.26,
which corrupts every `--json` payload (e.g. `pipeline --json`, which the
nightly job parses). Modules must import `pymupdf` instead."""

import pathlib
import re
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "pdf_gantry"


def test_no_module_imports_the_fitz_alias():
    offenders = [
        f"{p.name}:{n}"
        for p in SRC.glob("*.py")
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if re.match(r"\s*import fitz\b", line) or re.match(r"\s*from fitz\b", line)
    ]
    assert offenders == []


@pytest.mark.parametrize("module", ["pdf_gantry.ingest", "pdf_gantry.ocr", "pdf_gantry.process"])
def test_importing_module_writes_nothing_to_stdout(module):
    out = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert out == ""
