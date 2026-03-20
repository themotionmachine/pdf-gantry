"""CLI entry point for pdf-gantry."""

import click
from rich.console import Console
from rich.table import Table

from pdf_gantry.config import load_config

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """Agent-friendly CLI for managing academic PDF libraries."""


@cli.command()
@click.argument("query")
@click.option("-n", "--limit", default=10, show_default=True, help="Max results to return.")
def search(query: str, limit: int) -> None:
    """Full-text search across indexed PDFs."""
    pass


@cli.command()
@click.argument("query")
@click.option("-n", "--limit", default=10, show_default=True, help="Max results to return.")
def semantic(query: str, limit: int) -> None:
    """Semantic/embedding-based search across indexed PDFs."""
    pass


@cli.command()
def status() -> None:
    """Show index coverage: how many PDFs indexed vs total, any in error state."""
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--force", is_flag=True, help="Re-process even if already indexed.")
def process(path: str, force: bool) -> None:
    """Process a PDF: extract text, generate markdown, compute embeddings."""
    pass


@cli.command("queue")
def show_queue() -> None:
    """Show PDFs pending processing."""
    pass


@cli.command()
def ingest() -> None:
    """Scan the papers folder and queue any new or changed PDFs for processing."""
    pass


@cli.group()
def config() -> None:
    """Show or set configuration values."""


@config.command("show")
def config_show() -> None:
    """Print current configuration."""
    cfg = load_config()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="bold cyan")
    table.add_column("value")
    table.add_row("papers_dir", str(cfg.papers_dir))
    table.add_row("index_dir", str(cfg.index_dir))
    table.add_row("vault_dir", str(cfg.vault_dir) if cfg.vault_dir else "(not set)")
    console.print(table)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value (persisted via env var or config file)."""
    pass


@cli.group()
def vault() -> None:
    """Obsidian vault integration (read-only)."""


@vault.command("check")
def vault_check() -> None:
    """Compare PDFs against vault source notes: which have notes, which don't."""
    pass
