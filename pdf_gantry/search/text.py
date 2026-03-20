"""Full-text search over indexed PDFs."""

from pathlib import Path
from typing import List


class TextSearch:
    """Stub for full-text search.

    Will be backed by SQLite FTS5 or a dedicated search index.
    """

    def search(self, query: str, limit: int = 10) -> List[Path]:
        """Search indexed PDFs for the given query string."""
        return []
