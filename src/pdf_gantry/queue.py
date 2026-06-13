"""Property-based filtering and queue display."""

import sqlite3

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


def build_filter_query(
    needs: list[str] | None = None,
    has: list[str] | None = None,
    is_prop: list[str] | None = None,
    has_errors: bool = False,
    stale_embeddings: bool = False,
    current_model_version: str | None = None,
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

    if has_errors:
        conditions.append("error_count > 0")

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
) -> list[dict]:
    """Query papers matching the given filters."""
    where, params = build_filter_query(
        needs=needs, has=has, is_prop=is_prop,
        has_errors=has_errors, stale_embeddings=stale_embeddings,
        current_model_version=current_model_version,
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
) -> int:
    """Count papers matching the given filters."""
    where, params = build_filter_query(
        needs=needs, has=has, is_prop=is_prop,
        has_errors=has_errors, stale_embeddings=stale_embeddings,
        current_model_version=current_model_version,
    )
    row = conn.execute(f"SELECT COUNT(*) FROM papers {where}", params).fetchone()
    return row[0]
