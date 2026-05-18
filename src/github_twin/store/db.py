from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Create the sqlite-vec virtual table for chunk embeddings, sized to `dim`.

    sqlite-vec encodes the dimension into the table definition, so this must match
    the embedder. Re-running with a different dim is a fatal error — caller must
    drop and re-embed explicitly.
    """
    cur = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunk'")
    row = cur.fetchone()
    if row is None:
        conn.execute(
            f"CREATE VIRTUAL TABLE vec_chunk USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        return
    existing_sql = row[0] or ""
    if f"FLOAT[{dim}]" not in existing_sql:
        raise RuntimeError(
            f"vec_chunk exists with a different embedding dimension than {dim}. "
            "Delete data/db.sqlite or run `gt embed --rebuild` to recreate the index."
        )


def open_db(db_path: Path, embed_dim: int) -> sqlite3.Connection:
    """Open SQLite with sqlite-vec loaded, run schema, return the connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit; we manage txns
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    with SCHEMA_PATH.open() as f:
        conn.executescript(f.read())
    _ensure_vec_table(conn, embed_dim)
    _migrate_artifact_columns(conn)
    _backfill_fts(conn)
    return conn


def _migrate_artifact_columns(conn: sqlite3.Connection) -> None:
    """Backfill columns onto pre-existing `artifact` tables.

    `CREATE TABLE IF NOT EXISTS` is a no-op when the table exists, so newly
    added columns in schema.sql don't reach DBs that were created before the
    column was added. Detect missing columns via PRAGMA and ALTER them in.
    """
    have = {row["name"] for row in conn.execute("PRAGMA table_info(artifact)").fetchall()}
    if "content_hash" not in have:
        conn.execute("ALTER TABLE artifact ADD COLUMN content_hash TEXT")


def _backfill_fts(conn: sqlite3.Connection) -> None:
    """Seed chunk_fts from existing chunk rows when the index is empty.

    Triggers in schema.sql keep chunk_fts in sync for new writes; a brand-new
    DB never needs this path. The probe lives in `chunk_fts_docsize` (one row
    per indexed document) — a plain `SELECT ... FROM chunk_fts` proxies through
    to the content table and would return rows regardless of index state.
    """
    has_chunk = conn.execute("SELECT 1 FROM chunk LIMIT 1").fetchone()
    if has_chunk is None:
        return
    has_index = conn.execute("SELECT 1 FROM chunk_fts_docsize LIMIT 1").fetchone()
    if has_index is not None:
        return
    conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN")
    try:
        yield
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


@contextmanager
def db_session(db_path: Path, embed_dim: int) -> Iterator[sqlite3.Connection]:
    """`open_db` paired with a guaranteed `close()` on context exit.

    Use this for long-lived connections (the MCP server, async tasks)
    where explicit shutdown matters — WAL-mode SQLite is durable
    against process kill, but a clean `close()` runs the
    `wal-checkpoint`-style finalizer, releases file handles, and
    surfaces any close-time errors at the right place in the stack.

    Short-lived CLI commands still call `open_db()` directly; process
    exit handles cleanup via Python's `sqlite3` finalizer."""
    import contextlib

    conn = open_db(db_path, embed_dim)
    try:
        yield conn
    finally:
        # Close-time errors don't propagate — surrounding code already
        # finished its work; we don't want teardown to mask success.
        with contextlib.suppress(Exception):
            conn.close()
