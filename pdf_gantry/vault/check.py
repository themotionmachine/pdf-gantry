"""Read-only comparison of PDFs against Obsidian vault source notes."""

from pathlib import Path
from typing import List


class VaultChecker:
    """Stub for vault integrity checking (read-only).

    Compares the set of PDFs in the papers directory against source notes
    in the Obsidian vault to surface orphaned PDFs (no note) and broken
    links (note references a PDF that no longer exists).

    This class performs NO write operations on the vault.
    """

    def __init__(self, vault_dir: Path) -> None:
        self.vault_dir = vault_dir

    def check(self, papers_dir: Path) -> List[Path]:
        """Return PDFs that have no associated vault note."""
        return []
