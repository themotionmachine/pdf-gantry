"""PDF text extraction."""

from pathlib import Path


class PDFExtractor:
    """Stub for PDF text extraction.

    Will support direct text extraction (pymupdf / pdfminer) and
    an OCR fallback for scanned-only PDFs.
    """

    def extract(self, path: Path) -> str:
        """Extract plain text from a PDF file."""
        return ""
