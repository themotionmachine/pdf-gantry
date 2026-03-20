"""Click CLI entry point and command group."""

import json
import sys
from functools import wraps
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from . import __version__
from .config import Config, load_config, save_config, set_config_value
from .db import get_connection
from .models import StatusInfo
from .utils import format_count, format_pct, format_size, format_duration

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_NO_RESULTS = 2
EXIT_PARTIAL = 3
EXIT_DB_ERROR = 4

err_console = Console(stderr=True)


def filter_options(f):
    """Shared Click options for document filtering."""
    @click.option("--needs", multiple=True,
                  type=click.Choice(["text", "markdown", "embeddings", "ocr", "metadata"]),
                  help="Filter to documents missing this property")
    @click.option("--has", "has_prop", multiple=True,
                  type=click.Choice(["text", "markdown", "embeddings", "errors"]),
                  help="Filter to documents with this property")
    @click.option("--is", "is_prop", multiple=True,
                  type=click.Choice(["scanned", "digital"]),
                  help="Filter by document type")
    @click.option("--stale-embeddings", is_flag=True,
                  help="Documents with outdated embedding model version")
    @click.option("--limit", type=int, default=None,
                  help="Max documents to process")
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


@click.group()
@click.version_option(version=__version__)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def cli(ctx, json_output):
    """Agent-friendly CLI for managing academic PDF libraries."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output
    ctx.obj["config"] = load_config()


# --- config ---

@cli.group()
def config():
    """Show or modify configuration."""
    pass


@config.command("show")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def config_show(ctx, json_output):
    """Print current configuration."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    data = {
        "papers_dir": str(cfg.papers_dir),
        "index_dir": str(cfg.index_dir),
        "vault_dir": str(cfg.vault_dir) if cfg.vault_dir else None,
        "database": str(cfg.db_path),
    }

    if use_json:
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"papers_dir: {data['papers_dir']}")
        click.echo(f"index_dir:  {data['index_dir']}")
        if data["vault_dir"]:
            click.echo(f"vault_dir:  {data['vault_dir']}")
        click.echo(f"database:   {data['database']}")


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """Set a config value."""
    try:
        set_config_value(key, value)
        click.echo(f"Set {key} = {value}")
    except ValueError as e:
        click.echo(str(e), err=True)
        ctx.exit(EXIT_ERROR)


@config.command("init")
@click.pass_context
def config_init(ctx):
    """Interactive first-run setup."""
    click.echo("pdf_gantry setup")
    click.echo()

    papers = click.prompt(
        "Papers directory",
        default=str(Config().papers_dir),
    )
    papers_path = Path(papers).expanduser()
    if not papers_path.is_dir():
        click.echo(f"Warning: {papers_path} does not exist yet")

    vault = click.prompt(
        "Obsidian vault directory (optional, press Enter to skip)",
        default="",
    )

    cfg = Config(papers_dir=papers_path)
    if vault:
        cfg.vault_dir = Path(vault).expanduser()

    save_config(cfg)
    click.echo()
    click.echo(f"Config saved to {cfg.index_dir / 'config.yaml'}")


# --- ingest ---

@cli.command()
@click.option("--dry-run", is_flag=True, help="Report what would happen without writing")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def ingest(ctx, dry_run, json_output):
    """Scan papers folder, register new/changed PDFs."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.papers_dir.is_dir():
        msg = f"Papers directory not found: {cfg.papers_dir}"
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    from .ingest import ingest_directory

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    conn = get_connection(cfg.db_path)

    if not use_json:
        err_console.print(f"Scanning {cfg.papers_dir}")

    progress = None
    if not use_json:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=err_console,
        )
        task = progress.add_task("Ingesting", total=0)
        progress.start()

    def on_progress(current, total):
        if progress:
            progress.update(task, total=total, completed=current)

    try:
        stats = ingest_directory(
            conn, cfg.papers_dir,
            dry_run=dry_run,
            scan_threshold=cfg.processing.scan_threshold,
            progress_callback=on_progress,
        )
    finally:
        if progress:
            progress.stop()

    conn.close()

    if use_json:
        click.echo(json.dumps({
            "total_pdfs": stats.total_pdfs,
            "already_indexed": stats.already_indexed,
            "new": stats.new,
            "changed": stats.changed,
            "missing": stats.missing,
            "elapsed_seconds": stats.elapsed_seconds,
            "dry_run": dry_run,
        }, indent=2))
    else:
        prefix = "[DRY RUN] " if dry_run else ""
        click.echo(
            f"{prefix}Found {format_count(stats.total_pdfs)} PDFs "
            f"({format_count(stats.already_indexed)} indexed, "
            f"{format_count(stats.new)} new, "
            f"{format_count(stats.changed)} changed, "
            f"{format_count(stats.missing)} missing)"
        )
        if not dry_run and stats.new > 0:
            click.echo(f"Indexed {format_count(stats.new)} new documents in {stats.elapsed_seconds}s")


# --- status ---

@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def status(ctx, json_output):
    """Show index coverage and database stats."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.db_path.exists():
        msg = "No database found. Run 'gantry ingest' first."
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)

    info = StatusInfo(db_path=str(cfg.db_path))
    info.total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    info.with_text = conn.execute("SELECT COUNT(*) FROM papers WHERE has_text = 1").fetchone()[0]
    info.with_markdown = conn.execute("SELECT COUNT(*) FROM papers WHERE has_markdown = 1").fetchone()[0]
    info.with_embeddings = conn.execute("SELECT COUNT(*) FROM papers WHERE has_embeddings = 1").fetchone()[0]
    info.needs_ocr = conn.execute("SELECT COUNT(*) FROM papers WHERE needs_ocr = 1").fetchone()[0]
    info.has_errors = conn.execute("SELECT COUNT(*) FROM papers WHERE error_count > 0").fetchone()[0]
    info.db_size_bytes = cfg.db_path.stat().st_size

    conn.close()

    if use_json:
        click.echo(json.dumps({
            "db_path": info.db_path,
            "total": info.total,
            "with_text": info.with_text,
            "with_markdown": info.with_markdown,
            "with_embeddings": info.with_embeddings,
            "needs_ocr": info.needs_ocr,
            "has_errors": info.has_errors,
            "db_size_bytes": info.db_size_bytes,
            "pct_text": round(info.with_text / info.total * 100, 1) if info.total else 0,
            "pct_markdown": round(info.with_markdown / info.total * 100, 1) if info.total else 0,
            "pct_embeddings": round(info.with_embeddings / info.total * 100, 1) if info.total else 0,
        }, indent=2))
    else:
        click.echo(f"pdf_gantry index: {info.db_path}")
        click.echo()
        click.echo(f"Documents:       {format_count(info.total)}")
        click.echo(f"  With text:     {format_count(info.with_text)} ({format_pct(info.with_text, info.total)})")
        click.echo(f"  With markdown: {format_count(info.with_markdown)} ({format_pct(info.with_markdown, info.total)})")
        click.echo(f"  With embeddings: {format_count(info.with_embeddings)} ({format_pct(info.with_embeddings, info.total)})")
        click.echo(f"  Needs OCR:     {format_count(info.needs_ocr)} ({format_pct(info.needs_ocr, info.total)})")
        click.echo(f"  Has errors:    {format_count(info.has_errors)} ({format_pct(info.has_errors, info.total)})")
        click.echo()
        click.echo(f"Database size: {format_size(info.db_size_bytes)}")


# --- process ---

@cli.command()
@click.argument("path", required=False, type=click.Path(exists=True))
@click.option("--method", default="pymupdf4llm",
              type=click.Choice(["pymupdf4llm", "marker"]),
              help="Extraction method")
@click.option("--quality", is_flag=True, help="Use Marker for high-quality extraction")
@click.option("--workers", type=int, default=None, help="Number of concurrent workers")
@click.option("--force", is_flag=True, help="Re-process even if already processed")
@filter_options
@click.option("--dry-run", is_flag=True, help="Report what would happen")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def process(ctx, path, method, quality, workers, force, needs, has_prop, is_prop,
            stale_embeddings, limit, dry_run, json_output):
    """Extract text and markdown from PDFs."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if quality:
        method = "marker"

    if method == "marker":
        try:
            import marker  # noqa: F401
        except ImportError:
            msg = "Marker not installed. Run: pip install pdf-gantry[quality]"
            if use_json:
                click.echo(json.dumps({"error": msg}))
            else:
                click.echo(msg, err=True)
            ctx.exit(EXIT_ERROR)
            return

    if workers is None:
        workers = cfg.processing.workers

    conn = get_connection(cfg.db_path)

    from .process import process_documents
    from .queue import build_filter_query

    if path:
        # Single file mode
        p = Path(path)
        row = conn.execute("SELECT id FROM papers WHERE filename = ?", (p.name,)).fetchone()
        if not row:
            msg = f"File not found in index: {p.name}. Run 'gantry ingest' first."
            if use_json:
                click.echo(json.dumps({"error": msg}))
            else:
                click.echo(msg, err=True)
            ctx.exit(EXIT_ERROR)
            return
        paper_ids = [row["id"]]
    else:
        # Batch mode - apply filters
        if not needs and not has_prop and not is_prop and not force:
            needs = ("text",)
            is_prop = ("digital",)

        where, params = build_filter_query(
            needs=list(needs) if needs else None,
            has=list(has_prop) if has_prop else None,
            is_prop=list(is_prop) if is_prop else None,
        )

        rows = conn.execute(f"SELECT id FROM papers {where}", params).fetchall()
        paper_ids = [r["id"] for r in rows]

    if dry_run:
        count = len(paper_ids) if limit is None else min(len(paper_ids), limit)
        if use_json:
            click.echo(json.dumps({"would_process": count, "method": method}))
        else:
            click.echo(f"Would process {format_count(count)} documents with {method}")
        return

    if not paper_ids:
        if use_json:
            click.echo(json.dumps({"total": 0, "message": "Nothing to process"}))
        else:
            click.echo("Nothing to process")
        ctx.exit(EXIT_NO_RESULTS)
        return

    progress_bar = None
    task = None
    if not use_json:
        actual = len(paper_ids) if limit is None else min(len(paper_ids), limit)
        err_console.print(f"Processing {format_count(actual)} documents with {method} ({workers} workers)")
        progress_bar = Progress(
            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=err_console,
        )
        task = progress_bar.add_task("Processing", total=actual)
        progress_bar.start()

    def on_progress(current, total):
        if progress_bar and task is not None:
            progress_bar.update(task, completed=current)

    try:
        stats = process_documents(
            conn, cfg.papers_dir, cfg.db_path,
            paper_ids=paper_ids, method=method,
            workers=workers, limit=limit,
            progress_callback=on_progress,
        )
    finally:
        if progress_bar:
            progress_bar.stop()

    conn.close()

    if use_json:
        click.echo(json.dumps({
            "total": stats.total,
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "elapsed_seconds": stats.elapsed_seconds,
            "method": method,
        }, indent=2))
    else:
        click.echo(f"Processed {format_count(stats.total)} documents in {format_duration(stats.elapsed_seconds)}")
        click.echo(f"  Succeeded: {format_count(stats.succeeded)}")
        if stats.failed > 0:
            click.echo(f"  Failed: {format_count(stats.failed)} (use 'gantry queue --has errors' to see failures)")

    if stats.failed > 0 and stats.succeeded > 0:
        ctx.exit(EXIT_PARTIAL)
    elif stats.failed > 0 and stats.succeeded == 0:
        ctx.exit(EXIT_ERROR)


# --- search ---

@cli.command()
@click.argument("query")
@click.option("-n", "--limit", type=int, default=20, help="Max results")
@click.option("--hybrid", is_flag=True, help="Combine FTS5 and vector search")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def search(ctx, query, limit, hybrid, json_output):
    """Full-text search across indexed PDFs."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.db_path.exists():
        msg = "No database found. Run 'gantry ingest' first."
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)
    from .search import fts_search, search_count, hybrid_search

    try:
        if hybrid:
            from .embeddings import embed_query
            query_vec = embed_query(cfg.embedding.model, query)
            results = hybrid_search(conn, query, query_vec, limit=limit)
            total = len(results)
        else:
            total = search_count(conn, query)
            results = fts_search(conn, query, limit=limit)
    except ImportError as e:
        msg = str(e)
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return
    except Exception as e:
        msg = f"Search error: {e}"
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return
    finally:
        conn.close()

    if not results:
        if use_json:
            click.echo(json.dumps({"query": query, "total": 0, "results": []}))
        else:
            click.echo(f'No results for "{query}"')
        ctx.exit(EXIT_NO_RESULTS)
        return

    if use_json:
        click.echo(json.dumps({
            "query": query,
            "total": total,
            "results": [
                {
                    "id": r.id,
                    "filename": r.filename,
                    "path": r.path,
                    "score": r.score,
                    "snippet": r.snippet,
                    "has_markdown": r.has_markdown,
                    "has_embeddings": r.has_embeddings,
                }
                for r in results
            ],
        }, indent=2))
    else:
        click.echo(f'Found {total} results for "{query}"')
        click.echo()
        for i, r in enumerate(results, 1):
            click.echo(f" {i:2d}. [{r.score:.2f}] {r.filename}")
            if r.snippet:
                click.echo(f"     \"{r.snippet}\"")
            click.echo()


# --- queue ---

@cli.command()
@filter_options
@click.option("--count", is_flag=True, help="Just print the count")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def queue(ctx, needs, has_prop, is_prop, stale_embeddings, limit, count, json_output):
    """Show documents matching a filter."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.db_path.exists():
        msg = "No database found. Run 'gantry ingest' first."
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)
    from .queue import query_queue, queue_count

    n_list = list(needs) if needs else None
    h_list = list(has_prop) if has_prop else None
    i_list = list(is_prop) if is_prop else None

    # Default: show all needing any processing
    if not needs and not has_prop and not is_prop and not stale_embeddings:
        n_list = ["text"]

    if count:
        c = queue_count(conn, needs=n_list, has=h_list, is_prop=i_list,
                        stale_embeddings=stale_embeddings)
        conn.close()
        if use_json:
            click.echo(json.dumps({"count": c}))
        else:
            click.echo(str(c))
        return

    rows = query_queue(conn, needs=n_list, has=h_list, is_prop=i_list,
                       stale_embeddings=stale_embeddings, limit=limit)
    conn.close()

    if not rows:
        if use_json:
            click.echo(json.dumps({"count": 0, "documents": []}))
        else:
            click.echo("No documents match the filter")
        ctx.exit(EXIT_NO_RESULTS)
        return

    if use_json:
        click.echo(json.dumps({
            "count": len(rows),
            "documents": [
                {
                    "id": r["id"],
                    "filename": r["filename"],
                    "page_count": r["page_count"],
                    "is_scanned": bool(r["is_scanned"]) if r["is_scanned"] is not None else None,
                    "has_text": bool(r["has_text"]),
                    "has_markdown": bool(r["has_markdown"]),
                    "has_embeddings": bool(r["has_embeddings"]),
                    "error_count": r["error_count"],
                }
                for r in rows
            ],
        }, indent=2))
    else:
        label_parts = []
        if n_list:
            label_parts.append(f"need {', '.join(n_list)}")
        if h_list:
            label_parts.append(f"have {', '.join(h_list)}")
        if i_list:
            label_parts.append(f"are {', '.join(i_list)}")
        label = " and ".join(label_parts) if label_parts else "match"

        click.echo(f"{len(rows)} documents {label}")
        click.echo()
        for i, r in enumerate(rows[:50], 1):  # Cap display at 50
            extra = []
            if r["is_scanned"]:
                extra.append("scanned")
            if r["page_count"]:
                extra.append(f"{r['page_count']} pages")
            if r["error_count"] > 0:
                extra.append(f"{r['error_count']} errors")
            suffix = f" ({', '.join(extra)})" if extra else ""
            click.echo(f"  {i:3d}. {r['filename']}{suffix}")

        if len(rows) > 50:
            click.echo(f"  ... and {len(rows) - 50} more")


# --- errors ---

@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def errors(ctx, json_output):
    """Show documents with processing errors."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.db_path.exists():
        msg = "No database found."
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)
    rows = conn.execute(
        "SELECT id, filename, last_error, error_count, last_error_at FROM papers WHERE error_count > 0 ORDER BY error_count DESC"
    ).fetchall()
    conn.close()

    if not rows:
        if use_json:
            click.echo(json.dumps({"count": 0, "errors": []}))
        else:
            click.echo("No errors")
        return

    if use_json:
        click.echo(json.dumps({
            "count": len(rows),
            "errors": [
                {
                    "id": r["id"],
                    "filename": r["filename"],
                    "last_error": r["last_error"],
                    "error_count": r["error_count"],
                    "last_error_at": r["last_error_at"],
                }
                for r in rows
            ],
        }, indent=2))
    else:
        click.echo(f"{len(rows)} documents with errors")
        click.echo()
        for r in rows:
            click.echo(f"  {r['filename']}: {r['last_error']} (x{r['error_count']})")


# --- semantic ---

@cli.command()
@click.argument("query")
@click.option("-n", "--limit", type=int, default=20, help="Max results")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def semantic(ctx, query, limit, json_output):
    """Semantic similarity search (requires embeddings)."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.db_path.exists():
        msg = "No database found. Run 'gantry ingest' first."
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    from .embeddings import embed_query
    from .search import semantic_search

    try:
        query_vec = embed_query(cfg.embedding.model, query)
    except ImportError as e:
        if use_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(str(e), err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)
    results = semantic_search(conn, query_vec, limit=limit)
    conn.close()

    if not results:
        if use_json:
            click.echo(json.dumps({"query": query, "total": 0, "results": []}))
        else:
            click.echo(f'No results for "{query}"')
        ctx.exit(EXIT_NO_RESULTS)
        return

    if use_json:
        click.echo(json.dumps({
            "query": query,
            "total": len(results),
            "results": [
                {
                    "id": r.id,
                    "filename": r.filename,
                    "path": r.path,
                    "score": r.score,
                    "snippet": r.snippet,
                    "has_markdown": r.has_markdown,
                    "has_embeddings": r.has_embeddings,
                }
                for r in results
            ],
        }, indent=2))
    else:
        click.echo(f'Found {len(results)} results for "{query}"')
        click.echo()
        for i, r in enumerate(results, 1):
            click.echo(f" {i:2d}. [{r.score:.4f}] {r.filename}")
            if r.snippet:
                click.echo(f'     "{r.snippet[:100]}..."')
            click.echo()


# --- embed ---

@cli.command()
@filter_options
@click.option("--batch-size", type=int, default=None, help="Batch size for encoding")
@click.option("--dry-run", is_flag=True, help="Report what would happen")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def embed(ctx, needs, has_prop, is_prop, stale_embeddings, limit, batch_size, dry_run, json_output):
    """Generate embeddings for documents with text."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.db_path.exists():
        msg = "No database found. Run 'gantry ingest' first."
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)
    from .embeddings import embed_documents
    from .queue import build_filter_query

    if batch_size is None:
        batch_size = cfg.embedding.batch_size

    # Default filter: needs embeddings, has text
    if not needs and not has_prop and not is_prop and not stale_embeddings:
        needs = ("embeddings",)
        has_prop = ("text",)

    where, params = build_filter_query(
        needs=list(needs) if needs else None,
        has=list(has_prop) if has_prop else None,
        is_prop=list(is_prop) if is_prop else None,
        stale_embeddings=stale_embeddings,
        current_model_version=cfg.embedding.model.split("/")[-1] if stale_embeddings else None,
    )

    rows = conn.execute(f"SELECT id FROM papers {where}", params).fetchall()
    paper_ids = [r["id"] for r in rows]

    if limit:
        paper_ids = paper_ids[:limit]

    if dry_run:
        if use_json:
            click.echo(json.dumps({"would_embed": len(paper_ids), "model": cfg.embedding.model}))
        else:
            click.echo(f"Would embed {format_count(len(paper_ids))} documents with {cfg.embedding.model}")
        conn.close()
        return

    if not paper_ids:
        if use_json:
            click.echo(json.dumps({"total": 0, "message": "Nothing to embed"}))
        else:
            click.echo("Nothing to embed")
        conn.close()
        ctx.exit(EXIT_NO_RESULTS)
        return

    progress_bar = None
    task = None
    if not use_json:
        err_console.print(
            f"Embedding {format_count(len(paper_ids))} documents with "
            f"{cfg.embedding.model} (batch_size={batch_size})"
        )
        progress_bar = Progress(
            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=err_console,
        )
        task = progress_bar.add_task("Embedding", total=len(paper_ids))
        progress_bar.start()

    def on_progress(current, total):
        if progress_bar and task is not None:
            progress_bar.update(task, completed=current)

    try:
        stats = embed_documents(
            conn, cfg.db_path,
            paper_ids=paper_ids,
            model_name=cfg.embedding.model,
            dimensions=cfg.embedding.dimensions,
            batch_size=batch_size,
            progress_callback=on_progress,
        )
    except ImportError as e:
        if progress_bar:
            progress_bar.stop()
        if use_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            click.echo(str(e), err=True)
        conn.close()
        ctx.exit(EXIT_ERROR)
        return
    finally:
        if progress_bar:
            progress_bar.stop()

    conn.close()

    if use_json:
        click.echo(json.dumps({
            "total": stats.total,
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "elapsed_seconds": stats.elapsed_seconds,
            "model": cfg.embedding.model,
        }, indent=2))
    else:
        click.echo(f"Done. {format_count(stats.succeeded)} documents embedded in {format_duration(stats.elapsed_seconds)}")
        if stats.failed > 0:
            click.echo(f"  Failed: {format_count(stats.failed)}")

    if stats.failed > 0 and stats.succeeded > 0:
        ctx.exit(EXIT_PARTIAL)
    elif stats.failed > 0 and stats.succeeded == 0:
        ctx.exit(EXIT_ERROR)


# --- enrich ---

@cli.command()
@filter_options
@click.option("--dry-run", is_flag=True, help="Report what would happen")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def enrich(ctx, needs, has_prop, is_prop, stale_embeddings, limit, dry_run, json_output):
    """Fetch metadata from Semantic Scholar."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.db_path.exists():
        msg = "No database found. Run 'gantry ingest' first."
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)
    from .metadata import enrich_documents
    from .queue import build_filter_query

    if not needs and not has_prop and not is_prop:
        needs = ("metadata",)

    where, params = build_filter_query(
        needs=list(needs) if needs else None,
        has=list(has_prop) if has_prop else None,
        is_prop=list(is_prop) if is_prop else None,
    )

    rows = conn.execute(f"SELECT id FROM papers {where}", params).fetchall()
    paper_ids = [r["id"] for r in rows]

    if limit:
        paper_ids = paper_ids[:limit]

    if dry_run:
        if use_json:
            click.echo(json.dumps({"would_enrich": len(paper_ids)}))
        else:
            click.echo(f"Would enrich {format_count(len(paper_ids))} documents")
        conn.close()
        return

    if not paper_ids:
        if use_json:
            click.echo(json.dumps({"total": 0, "message": "Nothing to enrich"}))
        else:
            click.echo("Nothing to enrich")
        conn.close()
        ctx.exit(EXIT_NO_RESULTS)
        return

    progress_bar = None
    task = None
    if not use_json:
        err_console.print(f"Enriching metadata for {format_count(len(paper_ids))} documents")
        progress_bar = Progress(
            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=err_console,
        )
        task = progress_bar.add_task("Enriching", total=len(paper_ids))
        progress_bar.start()

    def on_progress(current, total):
        if progress_bar and task is not None:
            progress_bar.update(task, completed=current)

    try:
        stats = enrich_documents(
            conn, paper_ids=paper_ids, limit=limit,
            progress_callback=on_progress,
        )
    finally:
        if progress_bar:
            progress_bar.stop()

    conn.close()

    if use_json:
        click.echo(json.dumps({
            "total": stats.total,
            "doi_found": stats.doi_found,
            "matched_by_title": stats.matched_by_title,
            "no_match": stats.no_match,
            "api_errors": stats.api_errors,
            "elapsed_seconds": stats.elapsed_seconds,
        }, indent=2))
    else:
        click.echo(f"Enriched {format_count(stats.total)} documents in {format_duration(stats.elapsed_seconds)}")
        click.echo(f"  DOI found in text: {format_count(stats.doi_found)}")
        click.echo(f"  Matched via title: {format_count(stats.matched_by_title)}")
        click.echo(f"  No match found: {format_count(stats.no_match)}")
        if stats.api_errors:
            click.echo(f"  API errors: {format_count(stats.api_errors)}")


# --- retry ---

@cli.command()
@click.option("--max-attempts", type=int, default=3, help="Max retry attempts")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def retry(ctx, max_attempts, json_output):
    """Re-process documents that previously failed."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.db_path.exists():
        msg = "No database found."
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)
    from .process import process_documents

    rows = conn.execute(
        "SELECT id FROM papers WHERE error_count > 0 AND error_count < ?",
        (max_attempts,),
    ).fetchall()
    paper_ids = [r["id"] for r in rows]

    if not paper_ids:
        if use_json:
            click.echo(json.dumps({"total": 0, "message": "No documents to retry"}))
        else:
            click.echo("No documents to retry")
        conn.close()
        ctx.exit(EXIT_NO_RESULTS)
        return

    if not use_json:
        err_console.print(f"Retrying {format_count(len(paper_ids))} failed documents")

    progress_bar = None
    task = None
    if not use_json:
        progress_bar = Progress(
            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=err_console,
        )
        task = progress_bar.add_task("Retrying", total=len(paper_ids))
        progress_bar.start()

    def on_progress(current, total):
        if progress_bar and task is not None:
            progress_bar.update(task, completed=current)

    try:
        stats = process_documents(
            conn, cfg.papers_dir, cfg.db_path,
            paper_ids=paper_ids, workers=1,
            progress_callback=on_progress,
        )
    finally:
        if progress_bar:
            progress_bar.stop()

    conn.close()

    if use_json:
        click.echo(json.dumps({
            "total": stats.total,
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "elapsed_seconds": stats.elapsed_seconds,
        }, indent=2))
    else:
        click.echo(f"Retried {format_count(stats.total)} documents")
        click.echo(f"  Succeeded: {format_count(stats.succeeded)}")
        click.echo(f"  Still failing: {format_count(stats.failed)}")


# --- vault ---

@cli.group()
def vault():
    """Read-only Obsidian vault integration."""
    pass


@vault.command("check")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def vault_check(ctx, json_output):
    """Cross-reference PDFs with vault source notes."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.vault_dir:
        msg = "No vault_dir configured. Run 'gantry config set vault_dir /path/to/vault'"
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    if not cfg.vault_dir.is_dir():
        msg = f"Vault directory not found: {cfg.vault_dir}"
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)
    from .vault import check_vault

    stats = check_vault(conn, cfg.vault_dir)
    conn.close()

    if use_json:
        click.echo(json.dumps({
            "vault_dir": str(cfg.vault_dir),
            "total_pdfs": stats.total_pdfs,
            "with_notes": stats.with_notes,
            "without_notes": stats.without_notes,
            "orphan_references": len(stats.orphan_references),
        }, indent=2))
    else:
        click.echo(f"Vault: {cfg.vault_dir}")
        pct = format_pct(stats.with_notes, stats.total_pdfs)
        click.echo(f"PDFs with source notes:    {format_count(stats.with_notes)} / {format_count(stats.total_pdfs)} ({pct})")
        click.echo(f"PDFs without source notes: {format_count(stats.without_notes)}")
        if stats.orphan_references:
            click.echo(f"\nOrphaned references (PDFs not in library): {len(stats.orphan_references)}")
            for o in stats.orphan_references[:10]:
                click.echo(f"  - [[{o['reference']}]] in {o['note_path']}")
            if len(stats.orphan_references) > 10:
                click.echo(f"  ... and {len(stats.orphan_references) - 10} more")


@vault.command("orphans")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def vault_orphans(ctx, json_output):
    """Find vault notes referencing missing PDFs."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.vault_dir or not cfg.vault_dir.is_dir():
        msg = "Vault directory not configured or not found"
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)
    from .vault import get_orphan_references

    orphans = get_orphan_references(conn, cfg.vault_dir)
    conn.close()

    if use_json:
        click.echo(json.dumps({"count": len(orphans), "orphans": orphans}, indent=2))
    else:
        if not orphans:
            click.echo("No orphaned references found")
        else:
            click.echo(f"{len(orphans)} orphaned references")
            for o in orphans:
                click.echo(f"  - [[{o['reference']}]] in {o['note_path']}")


@vault.command("coverage")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def vault_coverage(ctx, json_output):
    """Find PDFs without vault source notes."""
    cfg = ctx.obj["config"]
    use_json = json_output or ctx.obj["json"]

    if not cfg.db_path.exists():
        msg = "No database found."
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)
    from .vault import get_uncovered_pdfs

    uncovered = get_uncovered_pdfs(conn)
    conn.close()

    if use_json:
        click.echo(json.dumps({
            "count": len(uncovered),
            "documents": [{"id": u["id"], "filename": u["filename"]} for u in uncovered],
        }, indent=2))
    else:
        if not uncovered:
            click.echo("All PDFs have vault source notes (or vault check hasn't been run)")
        else:
            click.echo(f"{len(uncovered)} PDFs without vault source notes")
            for u in uncovered[:50]:
                click.echo(f"  - {u['filename']}")
            if len(uncovered) > 50:
                click.echo(f"  ... and {len(uncovered) - 50} more")


cli.add_command(vault)
