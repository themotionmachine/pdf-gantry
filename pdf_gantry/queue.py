"""Processing queue for PDFs that need attention."""

from pathlib import Path
from typing import List


class ProcessingQueue:
    """Queue of PDFs awaiting processing.

    Stub — will be backed by a SQLite database in index_dir tracking:
    path, status (pending/processing/done/error), content hash,
    queued_at, processed_at, error_message.
    """

    def add(self, path: Path) -> None:
        """Enqueue a PDF for processing."""
        pass

    def list_pending(self) -> List[Path]:
        """Return all PDFs currently in pending state."""
        return []

    def mark_done(self, path: Path) -> None:
        """Mark a PDF as successfully processed."""
        pass
