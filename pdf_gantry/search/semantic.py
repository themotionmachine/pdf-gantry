"""Semantic/embedding-based search over indexed PDFs."""

from pathlib import Path
from typing import List


class SemanticSearch:
    """Stub for semantic search.

    Will use an embedding model and a vector store (TBD) to find
    semantically similar documents given a natural-language query.
    """

    def search(self, query: str, limit: int = 10) -> List[Path]:
        """Search indexed PDFs using semantic similarity."""
        return []
