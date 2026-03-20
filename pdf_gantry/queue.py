"""Processing queue for PDFs that need attention."""

# TODO: Implement queue backed by a SQLite database in index_dir.
# Each entry tracks: path, status (pending/processing/done/error),
# content hash, queued_at, processed_at, error_message.
