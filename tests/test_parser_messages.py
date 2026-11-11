"""PyMuPDF4LLM's parser messages stay off stdout and print once per run.

to_markdown reports its OCR engine ("Using Tesseract for OCR processing.")
through pymupdf.message, which writes to stdout, once per PDF, after a bare
print(). That broke `--json` output and made a working run look stuck.

The stdout checks run in a subprocess: pymupdf binds its message stream at
import, and pytest's capture swaps stdout around it, so only a real process
shows what a user sees.
"""

import subprocess
import sys
import textwrap

import pymupdf

from pdf_gantry import process as process_mod
from pdf_gantry.process import extract_text_pymupdf

MSG = "Using Tesseract for OCR processing."

CHATTY = textwrap.dedent(f"""
    import pymupdf
    from pdf_gantry import process as process_mod

    def to_markdown(path, **kwargs):
        print()
        pymupdf.message("=== Document parser messages ===")
        pymupdf.message({MSG!r})
        return "# markdown"

    process_mod.pymupdf4llm.to_markdown = to_markdown
""")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
    )


def test_parser_messages_never_reach_stdout(sample_pdf):
    proc = _run(CHATTY + textwrap.dedent(f"""
        from pdf_gantry.process import extract_text_pymupdf
        extract_text_pymupdf({str(sample_pdf)!r})
    """))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_run_reports_each_message_once_on_stderr(tmp_path, papers_dir):
    db_path = tmp_path / "test.db"
    proc = _run(CHATTY + textwrap.dedent(f"""
        from pathlib import Path
        from pdf_gantry.db import get_connection
        from pdf_gantry.ingest import ingest_directory
        from pdf_gantry.process import process_documents

        conn = get_connection({str(db_path)!r})
        ingest_directory(conn, Path({str(papers_dir)!r}))
        stats = process_documents(conn, Path({str(papers_dir)!r}), Path({str(db_path)!r}),
                                  workers=1)
        assert stats.succeeded == 2, stats
        assert stats.parser_messages == [{MSG!r}], stats.parser_messages
    """))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr.count(MSG) == 1
    assert "===" not in proc.stderr


def test_real_parser_keeps_stdout_clean(sample_pdf):
    proc = _run(textwrap.dedent(f"""
        from pdf_gantry.process import extract_text_pymupdf
        extract_text_pymupdf({str(sample_pdf)!r})
    """))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_message_destination_restored(sample_pdf, monkeypatch):
    monkeypatch.setattr(process_mod.pymupdf4llm, "to_markdown", lambda path, **kw: "# md")
    before_messages, before_stdout = pymupdf._g_out_message, sys.stdout
    extract_text_pymupdf(sample_pdf)
    assert pymupdf._g_out_message is before_messages
    assert sys.stdout is before_stdout
