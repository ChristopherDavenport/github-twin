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
    """Open SQLite with sqlite-vec loaded, run migrations, return the connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit; we manage txns
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    _run_pre_schema_migrations(conn)
    with SCHEMA_PATH.open() as f:
        conn.executescript(f.read())
    _ensure_vec_table(conn, embed_dim)
    _backfill_fts(conn)
    return conn


def _backfill_fts(conn: sqlite3.Connection) -> None:
    """Seed chunk_fts from existing chunk rows on first open after FTS5 lands.

    Triggers in schema.sql keep chunk_fts in sync for new writes, but DBs that
    predate this migration have populated chunk rows and a freshly created,
    empty chunk_fts index. The FTS5 'rebuild' command repopulates the index
    from the external content table (`chunk`) — the canonical approach for
    external-content FTS5.

    Detection nuance: `SELECT ... FROM chunk_fts` on an external-content table
    proxies through to the content table, so it returns rows whenever `chunk`
    has rows regardless of index state. The actual index population lives in
    the shadow table `chunk_fts_docsize` — one row per indexed document. We
    probe that to decide whether a rebuild is needed.
    """
    has_chunk = conn.execute("SELECT 1 FROM chunk LIMIT 1").fetchone()
    if has_chunk is None:
        return
    has_index = conn.execute("SELECT 1 FROM chunk_fts_docsize LIMIT 1").fetchone()
    if has_index is not None:
        return
    conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")


def _run_pre_schema_migrations(conn: sqlite3.Connection) -> None:
    """Add columns to existing tables before schema.sql replays.

    schema.sql is idempotent (CREATE TABLE IF NOT EXISTS) so it won't fix
    existing tables. Anything that mutates an existing table's shape must run
    here, *before* schema.sql, so the indexes in schema.sql see the new columns.
    """
    chunk_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk'"
    ).fetchone()
    if chunk_exists:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chunk)").fetchall()}
        if "language" not in cols:
            conn.execute("ALTER TABLE chunk ADD COLUMN language TEXT")
        if "node_kind" not in cols:
            conn.execute("ALTER TABLE chunk ADD COLUMN node_kind TEXT")
        if "symbol_name" not in cols:
            conn.execute("ALTER TABLE chunk ADD COLUMN symbol_name TEXT")
        if "summary" not in cols:
            # NL summary written by `gt summarize`. NULL = not yet summarized;
            # `pending_summary_chunks` filters on it. Embed-time prefix
            # includes the summary in the chunk header when non-NULL.
            conn.execute("ALTER TABLE chunk ADD COLUMN summary TEXT")

    # artifact.author_login — needed for org-mode "who wrote this" filtering.
    artifact_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifact'"
    ).fetchone()
    if artifact_exists:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(artifact)").fetchall()}
        if "author_login" not in cols:
            conn.execute("ALTER TABLE artifact ADD COLUMN author_login TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS artifact_author ON artifact(author_login)")

    # repo.last_commits_walked_sha — added when commits ingest moved from the
    # GitHub API to a local git walk over deep clones. Old org-mode DBs need
    # the column before schema.sql replays.
    repo_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='repo'"
    ).fetchone()
    if repo_exists:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(repo)").fetchall()}
        if "last_commits_walked_sha" not in cols:
            conn.execute("ALTER TABLE repo ADD COLUMN last_commits_walked_sha TEXT")

    # developer_profile_cache — added with the prompt-management tools.
    # No data migration needed; just create the table if missing.
    profile_cache_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='developer_profile_cache'"
    ).fetchone()
    if not profile_cache_exists:
        conn.execute(
            "CREATE TABLE developer_profile_cache ("
            "login TEXT PRIMARY KEY, "
            "profile_md TEXT NOT NULL, "
            "sample_hash TEXT NOT NULL, "
            "n_samples INTEGER NOT NULL, "
            "generated_at TEXT NOT NULL)"
        )

    # identity (singleton, user-mode-only) → target (singleton, kind-discriminated).
    identity_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='identity'"
    ).fetchone()
    target_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='target'"
    ).fetchone()
    if identity_exists and not target_exists:
        conn.execute(
            "CREATE TABLE target ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), "
            "kind TEXT NOT NULL, "
            "name TEXT NOT NULL, "
            "external_id INTEGER NOT NULL, "
            "emails_json TEXT, "
            "discovered_at TEXT NOT NULL)"
        )
        row = conn.execute(
            "SELECT username, user_id, emails_json, discovered_at FROM identity"
        ).fetchone()
        if row is not None:
            conn.execute(
                "INSERT INTO target (id, kind, name, external_id, emails_json, "
                "discovered_at) VALUES (1, 'user', ?, ?, ?, ?)",
                (row[0], row[1], row[2], row[3]),
            )
        conn.execute("DROP TABLE identity")


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
