"""Property-based filtering and queue display."""

import sqlite3

# A permanently-broken PDF (e.g. one PyMuPDF can't open) fails on every pipeline
# run, incrementing error_count without end and wasting work. Once a paper has
# failed this many times it is "quarantined": excluded from default process/embed
# selection and surfaced via `queue --is broken`. Resettable via `gantry retry
# --ids`. Configurable as processing.max_retries in config.yaml.
DEFAULT_MAX_RETRIES = 3


def quarantine_condition(max_retries: int = DEFAULT_MAX_RETRIES) -> str:
    """SQL predicate (against papers) for a quarantined, retry-exhausted paper."""
    return f"error_count >= {int(max_retries)}"


def not_quarantined_condition(max_retries: int = DEFAULT_MAX_RETRIES) -> str:
    """SQL predicate for a paper still eligible for (re)processing."""
    return f"error_count < {int(max_retries)}"


# A digital PDF whose body is rendered as bitmap extracts to almost nothing
# (PyMuPDF returns "picture intentionally omitted" placeholders), yet still
# reports has_text=1 and usually needs_ocr=1. Flag papers whose extracted text
# is implausibly thin for their page count so the silent failure becomes loud.
# Threshold is a calibrated heuristic, not a hard rule (see issue #17).
SUSPICIOUS_CHARS_PER_PAGE = 500


def suspicious_extraction_condition() -> str:
    """SQL predicate (against the papers table) for a likely-broken extraction."""
    return (
        "has_text = 1 AND needs_ocr = 1 AND page_count > 0 "
        "AND COALESCE("
        "(SELECT text_length FROM paper_text WHERE paper_text.paper_id = papers.id), 0"
        f") < page_count * {SUSPICIOUS_CHARS_PER_PAGE}"
    )


# --- metadata provenance -----------------------------------------------------
#
# `metadata_source` has three kinds of value beyond a provider name:
#   - 'none:<provider>' -- enrichment ran and the provider had no match. Before
#     2026-09 a miss stamped `metadata_enriched_at` only, so legacy misses are
#     rows with the stamp but no source and no title; both count as misses.
#   - 'manual:<who>' -- set by `gantry meta set`. Legacy hand-SQL writes used
#     free-form names containing "manual" (e.g. 'brev-manual-from-text').
#     Manual metadata is never overwritten by enrich.
#   - NULL with no stamp -- never attempted.
MISS_SOURCE_PREFIX = "none:"
MANUAL_SOURCE_PREFIX = "manual:"


def miss_condition() -> str:
    """SQL predicate (against papers) for an enrichment attempt that found nothing."""
    return (
        "(metadata_source LIKE 'none:%' OR (metadata_source IS NULL "
        "AND metadata_enriched_at IS NOT NULL "
        "AND (title IS NULL OR title = '')))"
    )


def manual_condition() -> str:
    """SQL predicate for hand-set metadata (``meta set`` or legacy hand SQL)."""
    return "(metadata_source LIKE 'manual:%' OR metadata_source LIKE '%manual%')"


def enriched_condition() -> str:
    """SQL predicate for metadata that actually came from somewhere (not a miss)."""
    return "(metadata_source IS NOT NULL AND metadata_source NOT LIKE 'none:%')"


def never_enriched_condition() -> str:
    """SQL predicate for papers default `enrich` should pick up: never
    attempted, and not hand-set."""
    return (
        "(metadata_enriched_at IS NULL "
        "AND (metadata_source IS NULL OR NOT " + manual_condition() + "))"
    )


def build_filter_query(
    needs: list[str] | None = None,
    has: list[str] | None = None,
    is_prop: list[str] | None = None,
    has_errors: bool = False,
    stale_embeddings: bool = False,
    current_model_version: str | None = None,
    exclude_quarantined: bool = False,
    exclude_encrypted: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[str, list]:
    """
    Build a WHERE clause for filtering the papers table.
    Returns (where_clause, params).
    """
    conditions = []
    params: list = []

    if needs:
        for n in needs:
            if n == "text":
                conditions.append("has_text = 0")
            elif n == "markdown":
                conditions.append("has_markdown = 0")
            elif n == "embeddings":
                conditions.append("has_embeddings = 0")
            elif n == "ocr":
                conditions.append("needs_ocr = 1 AND ocr_completed_at IS NULL")
            elif n == "chunk_embeddings":
                conditions.append("has_chunk_embeddings = 0")
            elif n == "metadata":
                conditions.append("(doi IS NULL OR metadata_enriched_at IS NULL)")
            elif n == "citekey":
                conditions.append("citekey IS NULL")

    if has:
        for h in has:
            if h == "text":
                conditions.append("has_text = 1")
            elif h == "markdown":
                conditions.append("has_markdown = 1")
            elif h == "embeddings":
                conditions.append("has_embeddings = 1")
            elif h == "chunk_embeddings":
                conditions.append("has_chunk_embeddings = 1")
            elif h == "errors":
                conditions.append("error_count > 0")
            elif h == "citekey":
                conditions.append("citekey IS NOT NULL")

    if is_prop:
        for p in is_prop:
            if p == "scanned":
                conditions.append("is_scanned = 1")
            elif p == "digital":
                conditions.append("is_scanned = 0")
            elif p == "suspicious":
                conditions.append(f"({suspicious_extraction_condition()})")
            elif p == "broken":
                conditions.append(quarantine_condition(max_retries))
            elif p == "metadata-suspect":
                conditions.append("metadata_suspect = 1")
            elif p == "encrypted":
                conditions.append("is_encrypted = 1")
            elif p == "enrich-miss":
                conditions.append(miss_condition())
            elif p == "manual-metadata":
                conditions.append(manual_condition())

    if has_errors:
        conditions.append("error_count > 0")

    if exclude_quarantined:
        conditions.append(not_quarantined_condition(max_retries))

    if exclude_encrypted:
        conditions.append("is_encrypted = 0")

    if stale_embeddings and current_model_version:
        conditions.append("has_embeddings = 1 AND embedding_model_version != ?")
        params.append(current_model_version)

    if not conditions:
        return "", params

    return "WHERE " + " AND ".join(conditions), params


def query_queue(
    conn: sqlite3.Connection,
    needs: list[str] | None = None,
    has: list[str] | None = None,
    is_prop: list[str] | None = None,
    has_errors: bool = False,
    stale_embeddings: bool = False,
    current_model_version: str | None = None,
    limit: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict]:
    """Query papers matching the given filters."""
    where, params = build_filter_query(
        needs=needs, has=has, is_prop=is_prop,
        has_errors=has_errors, stale_embeddings=stale_embeddings,
        current_model_version=current_model_version,
        max_retries=max_retries,
    )

    sql = f"SELECT * FROM papers {where} ORDER BY filename"
    if limit:
        sql += f" LIMIT {limit}"

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def queue_count(
    conn: sqlite3.Connection,
    needs: list[str] | None = None,
    has: list[str] | None = None,
    is_prop: list[str] | None = None,
    has_errors: bool = False,
    stale_embeddings: bool = False,
    current_model_version: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> int:
    """Count papers matching the given filters."""
    where, params = build_filter_query(
        needs=needs, has=has, is_prop=is_prop,
        has_errors=has_errors, stale_embeddings=stale_embeddings,
        current_model_version=current_model_version,
        max_retries=max_retries,
    )
    row = conn.execute(f"SELECT COUNT(*) FROM papers {where}", params).fetchone()
    return row[0]
