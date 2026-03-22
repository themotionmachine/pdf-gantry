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
                  type=click.Choice(["text", "markdown", "embeddings", "chunk_embeddings", "ocr", "metadata"]),
                  help="Filter to documents missing this property")
    @click.option("--has", "has_prop", multiple=True,
                  type=click.Choice(["text", "markdown", "embeddings", "chunk_embeddings", "errors"]),
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
            "evicted": stats.evicted,
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
        if stats.evicted > 0:
            click.echo(
                f"\n{format_count(stats.evicted)} files evicted from iCloud (skipped). "
                f"To re-download:\n"
                f"  find \"{cfg.papers_dir}\" -name '*.pdf' -exec brctl download {{}} +"
            )

    if stats.evicted > 0:
        ctx.exit(EXIT_PARTIAL)


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
    info.with_chunk_embeddings = conn.execute("SELECT COUNT(*) FROM papers WHERE has_chunk_embeddings = 1").fetchone()[0]
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
        click.echo(f"  With chunk embeddings: {format_count(info.with_chunk_embeddings)} ({format_pct(info.with_chunk_embeddings, info.total)})")
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
@click.option("--components", is_flag=True, help="Include FTS and vector component scores (hybrid only)")
@click.option("--fields", "field_list", type=str, default=None,
              help="Comma-separated fields to include in JSON output")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def search(ctx, query, limit, hybrid, components, field_list, json_output):
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
        result_dicts = []
        for r in results:
            d = {
                "id": r.id,
                "filename": r.filename,
                "path": r.path,
                "score": r.score,
                "snippet": r.snippet,
                "has_markdown": r.has_markdown,
                "has_embeddings": r.has_embeddings,
            }
            if components and hybrid:
                d["score_fts"] = r.score_fts
                d["score_vector"] = r.score_vector
                d["rank_fts"] = r.rank_fts
                d["rank_vector"] = r.rank_vector
            if field_list:
                fields = {f.strip() for f in field_list.split(",")}
                d = {k: v for k, v in d.items() if k in fields}
            result_dicts.append(d)

        click.echo(json.dumps({
            "query": query,
            "total": total,
            "results": result_dicts,
        }, indent=2))
    else:
        click.echo(f'Found {total} results for "{query}"')
        click.echo()
        for i, r in enumerate(results, 1):
            click.echo(f" {i:2d}. [{r.score:.2f}] {r.filename}")
            if components and hybrid and (r.score_fts is not None or r.score_vector is not None):
                fts_str = f"fts={r.score_fts:.2f}" if r.score_fts is not None else "fts=--"
                vec_str = f"vec={r.score_vector:.4f}" if r.score_vector is not None else "vec=--"
                click.echo(f"     {fts_str}  {vec_str}")
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
@click.option("--doc-only", is_flag=True, help="Use doc-level embeddings only (skip chunk cascade)")
@click.option("--fields", "field_list", type=str, default=None,
              help="Comma-separated fields to include in JSON output")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def semantic(ctx, query, limit, doc_only, field_list, json_output):
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
    from .search import semantic_search, cascade_search

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

    # Use cascade search if chunk embeddings exist, unless --doc-only
    has_chunks = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE has_chunk_embeddings = 1"
    ).fetchone()[0] > 0

    if has_chunks and not doc_only:
        results = cascade_search(conn, query_vec, limit=limit)
    else:
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
        result_dicts = []
        for r in results:
            d = {
                "id": r.id,
                "filename": r.filename,
                "path": r.path,
                "score": r.score,
                "snippet": r.snippet,
                "has_markdown": r.has_markdown,
                "has_embeddings": r.has_embeddings,
            }
            if field_list:
                fields = {f.strip() for f in field_list.split(",")}
                d = {k: v for k, v in d.items() if k in fields}
            result_dicts.append(d)

        click.echo(json.dumps({
            "query": query,
            "total": len(results),
            "results": result_dicts,
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
@click.option("--chunk", "chunk_mode", is_flag=True, help="Generate chunk-level embeddings")
@click.option("--batch-size", type=int, default=None, help="Batch size for encoding")
@click.option("--dry-run", is_flag=True, help="Report what would happen")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def embed(ctx, needs, has_prop, is_prop, stale_embeddings, limit, chunk_mode, batch_size, dry_run, json_output):
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
    from .embeddings import embed_documents, embed_chunks
    from .queue import build_filter_query

    if batch_size is None:
        batch_size = cfg.embedding.batch_size if not chunk_mode else 64

    # Default filter depends on mode
    if not needs and not has_prop and not is_prop and not stale_embeddings:
        if chunk_mode:
            needs = ("chunk_embeddings",)
            has_prop = ("text",)
        else:
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
        embed_fn = embed_chunks if chunk_mode else embed_documents
        stats = embed_fn(
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


# --- pipeline ---

@cli.command()
@click.option("--file", "filename", type=str, default=None, help="Process a single file by name")
@click.option("--limit", type=int, default=None, help="Max papers to process end-to-end")
@click.option("--workers", type=int, default=None, help="Number of concurrent workers")
@click.option("--dry-run", is_flag=True, help="Report what would happen")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def pipeline(ctx, filename, limit, workers, dry_run, json_output):
    """Run full ingestion pipeline: ingest → process → embed."""
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

    if workers is None:
        workers = cfg.processing.workers

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    conn = get_connection(cfg.db_path)

    from .pipeline import run_pipeline

    if not use_json and not dry_run:
        if filename:
            err_console.print(f"Pipeline: {filename}")
        else:
            err_console.print(f"Pipeline: scanning {cfg.papers_dir}")

    stats = run_pipeline(
        conn, cfg.papers_dir, cfg.db_path,
        workers=workers, limit=limit, dry_run=dry_run,
        filename=filename,
        scan_threshold=cfg.processing.scan_threshold,
    )
    conn.close()

    if use_json:
        click.echo(json.dumps(stats, indent=2))
    else:
        prefix = "[DRY RUN] " if dry_run else ""
        click.echo(f"{prefix}Ingested: {stats['ingested']}, "
                    f"Processed: {stats['processed']}, "
                    f"Embedded: {stats['embedded']}, "
                    f"Errors: {stats['errors']}")

    if stats.get("errors", 0) > 0 and stats.get("processed", 0) > 0:
        ctx.exit(EXIT_PARTIAL)
    elif stats.get("errors", 0) > 0 and stats.get("processed", 0) == 0:
        ctx.exit(EXIT_ERROR)


# --- prune ---

@cli.command()
@click.option("--dry-run", is_flag=True, help="Report what would be removed")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def prune(ctx, dry_run, json_output):
    """Remove database entries for files no longer on disk."""
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
    from .prune import prune_missing

    stats = prune_missing(conn, cfg.papers_dir, dry_run=dry_run)
    conn.close()

    if use_json:
        click.echo(json.dumps(stats, indent=2))
    else:
        prefix = "[DRY RUN] " if dry_run else ""
        if stats["pruned"] == 0:
            click.echo(f"{prefix}No ghost entries found ({stats['remaining']} papers in index)")
        else:
            click.echo(f"{prefix}Pruned {stats['pruned']} ghost entries ({stats['remaining']} remaining)")
            for f in stats["pruned_files"]:
                click.echo(f"  - {f}")


# --- find ---

@cli.command()
@click.argument("fragment")
@click.option("-n", "--limit", type=int, default=20, help="Max results")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def find(ctx, fragment, limit, json_output):
    """Fuzzy filename lookup. FRAGMENT matches anywhere in the filename."""
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
    from .search import find_papers

    results = find_papers(conn, fragment, limit=limit)
    conn.close()

    if not results:
        if use_json:
            click.echo(json.dumps({"fragment": fragment, "count": 0, "results": []}))
        else:
            click.echo(f'No papers matching "{fragment}"')
        ctx.exit(EXIT_NO_RESULTS)
        return

    if use_json:
        click.echo(json.dumps({
            "fragment": fragment,
            "count": len(results),
            "results": [
                {
                    "id": r["id"],
                    "filename": r["filename"],
                    "title": r["title"],
                    "page_count": r["page_count"],
                    "has_text": bool(r["has_text"]),
                }
                for r in results
            ],
        }, indent=2))
    else:
        click.echo(f'{len(results)} papers matching "{fragment}"')
        click.echo()
        for r in results:
            extra = f" — {r['title']}" if r["title"] else ""
            click.echo(f"  [{r['id']:4d}] {r['filename']}{extra}")


# --- info ---

@cli.command()
@click.option("--ids", type=str, required=True, help="Comma-separated paper IDs")
@click.option("--fields", "field_list", type=str, default=None,
              help="Comma-separated fields to include")
@click.option("--chunks", "include_chunks", is_flag=True,
              help="Include chunk texts for each paper")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def info(ctx, ids, field_list, include_chunks, json_output):
    """Fetch metadata for specific papers by ID."""
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

    try:
        paper_ids = [int(x.strip()) for x in ids.split(",")]
    except ValueError:
        msg = "Invalid --ids: must be comma-separated integers"
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    conn = get_connection(cfg.db_path)

    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"""SELECT p.id, p.filename, p.path, p.title, p.authors, p.year, p.doi,
                   p.abstract, p.has_text, p.has_markdown, p.has_embeddings,
                   p.has_chunk_embeddings, p.page_count, p.is_scanned,
                   p.text_method, p.error_count, p.last_error,
                   SUBSTR(pt.raw_text, 1, 300) as snippet
            FROM papers p
            LEFT JOIN paper_text pt ON pt.paper_id = p.id
            WHERE p.id IN ({placeholders})""",
        paper_ids,
    ).fetchall()

    if not rows:
        if use_json:
            click.echo(json.dumps({"count": 0, "papers": []}))
        else:
            click.echo("No papers found for given IDs")
        conn.close()
        ctx.exit(EXIT_NO_RESULTS)
        return

    results = []
    for r in rows:
        d = {
            "id": r["id"],
            "filename": r["filename"],
            "path": r["path"],
            "title": r["title"],
            "authors": r["authors"],
            "year": r["year"],
            "doi": r["doi"],
            "abstract": r["abstract"],
            "snippet": r["snippet"],
            "page_count": r["page_count"],
            "has_text": bool(r["has_text"]),
            "has_markdown": bool(r["has_markdown"]),
            "has_embeddings": bool(r["has_embeddings"]),
            "has_chunk_embeddings": bool(r["has_chunk_embeddings"]),
            "is_scanned": bool(r["is_scanned"]) if r["is_scanned"] is not None else None,
            "error_count": r["error_count"],
        }

        if include_chunks:
            chunks = conn.execute(
                "SELECT chunk_id, chunk_index, section_header, text FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
                (r["id"],),
            ).fetchall()
            d["chunks"] = [
                {
                    "chunk_id": c["chunk_id"],
                    "chunk_index": c["chunk_index"],
                    "section_header": c["section_header"],
                    "text": c["text"],
                }
                for c in chunks
            ]

        if field_list:
            fields = {f.strip() for f in field_list.split(",")}
            d = {k: v for k, v in d.items() if k in fields}

        results.append(d)

    conn.close()

    if use_json:
        click.echo(json.dumps({"count": len(results), "papers": results}, indent=2))
    else:
        for d in results:
            click.echo(f"[{d.get('id')}] {d.get('filename', '?')}")
            if d.get("title"):
                click.echo(f"  Title: {d['title']}")
            if d.get("authors"):
                click.echo(f"  Authors: {d['authors']}")
            if d.get("year"):
                click.echo(f"  Year: {d['year']}")
            if d.get("doi"):
                click.echo(f"  DOI: {d['doi']}")
            click.echo()


# --- read ---

@cli.command()
@click.argument("identifier")
@click.option("--chunk", "chunk_id", type=int, default=None, help="Read a specific chunk by ID")
@click.option("--chunks", "list_chunks", is_flag=True, help="List all chunks for a document")
@click.option("--context", "context_chars", type=int, default=None,
              help="Expand chunk with surrounding context (chars). Use with --chunk")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def read(ctx, identifier, chunk_id, list_chunks, context_chars, json_output):
    """Read document text or chunks. IDENTIFIER is a paper ID or filename."""
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

    if chunk_id is not None and context_chars is not None:
        # Scoped context window around a chunk
        from .search import get_chunk_context
        result = get_chunk_context(conn, chunk_id, max_chars=context_chars)
        conn.close()
        if not result:
            msg = f"Chunk {chunk_id} not found"
            if use_json:
                click.echo(json.dumps({"error": msg}))
            else:
                click.echo(msg, err=True)
            ctx.exit(EXIT_ERROR)
            return
        if use_json:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"[{result['filename']}] chunk {result['chunk_index']} "
                        f"(±context, {len(result['context']):,} chars, "
                        f"{result['total_chunks']} total chunks)")
            if result["section_header"]:
                click.echo(f"Section: {result['section_header']}")
            click.echo()
            click.echo(result["context"])
        return

    if chunk_id is not None:
        # Read a specific chunk by ID
        row = conn.execute(
            """SELECT c.*, p.filename, p.title
            FROM chunks c JOIN papers p ON p.id = c.doc_id
            WHERE c.chunk_id = ?""",
            (chunk_id,),
        ).fetchone()
        conn.close()
        if not row:
            msg = f"Chunk {chunk_id} not found"
            if use_json:
                click.echo(json.dumps({"error": msg}))
            else:
                click.echo(msg, err=True)
            ctx.exit(EXIT_ERROR)
            return
        if use_json:
            click.echo(json.dumps({
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "title": row["title"],
                "chunk_index": row["chunk_index"],
                "section_header": row["section_header"],
                "text": row["text"],
            }, indent=2))
        else:
            click.echo(f"[{row['filename']}] chunk {row['chunk_index']}")
            if row["section_header"]:
                click.echo(f"Section: {row['section_header']}")
            click.echo()
            click.echo(row["text"])
        return

    # Resolve identifier to paper
    try:
        paper_id = int(identifier)
        paper = conn.execute("SELECT id, filename, title FROM papers WHERE id = ?", (paper_id,)).fetchone()
    except ValueError:
        paper = conn.execute("SELECT id, filename, title FROM papers WHERE filename = ?", (identifier,)).fetchone()

    if not paper:
        msg = f"Paper not found: {identifier}"
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        conn.close()
        ctx.exit(EXIT_ERROR)
        return

    if list_chunks:
        chunks = conn.execute(
            "SELECT chunk_id, chunk_index, section_header, LENGTH(text) as text_len FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (paper["id"],),
        ).fetchall()
        conn.close()
        if use_json:
            click.echo(json.dumps({
                "paper_id": paper["id"],
                "filename": paper["filename"],
                "chunks": [
                    {
                        "chunk_id": c["chunk_id"],
                        "chunk_index": c["chunk_index"],
                        "section_header": c["section_header"],
                        "text_length": c["text_len"],
                    }
                    for c in chunks
                ],
            }, indent=2))
        else:
            click.echo(f"{paper['filename']} — {len(chunks)} chunks")
            for c in chunks:
                header = f" [{c['section_header']}]" if c["section_header"] else ""
                click.echo(f"  {c['chunk_index']:3d}. (id={c['chunk_id']}) {c['text_len']:,} chars{header}")
        return

    # Default: read full markdown
    text = conn.execute(
        "SELECT markdown, raw_text FROM paper_text WHERE paper_id = ?",
        (paper["id"],),
    ).fetchone()
    conn.close()

    if not text:
        msg = f"No text extracted for {paper['filename']}. Run 'gantry process' first."
        if use_json:
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(msg, err=True)
        ctx.exit(EXIT_ERROR)
        return

    content = text["markdown"] or text["raw_text"]
    if use_json:
        click.echo(json.dumps({
            "paper_id": paper["id"],
            "filename": paper["filename"],
            "title": paper["title"],
            "text": content,
        }, indent=2))
    else:
        click.echo(content)


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
