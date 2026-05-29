"""SQLite-backed `HttpCache` adapter.

This module is the ONLY place where the HTTP cache touches SQLite. It
implements the `HttpCache` Protocol against a `sqlite3.Connection`
passed in by the caller — typically the same connection the rest of
ingest uses, so cache hits/misses participate in the existing WAL.

Schema is owned here (inline `CREATE TABLE IF NOT EXISTS`) so the
cache stays an opt-in concern: a DB that never instantiates
`SqliteHttpCache` simply never has an `http_cache` table.
"""

from __future__ import annotations

import sqlite3
import threading

from github_twin.ingest.http_cache import CacheEntry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
  url           TEXT PRIMARY KEY,
  etag          TEXT,
  last_modified TEXT,
  body          BLOB NOT NULL,
  fetched_at    TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


class SqliteHttpCache:
    """SQLite-backed HttpCache. One row per canonical URL.

    Stores body as a BLOB inline. For the bulk reviews endpoints a
    single page is on the order of 50-500 KB, and we only cache the
    first-page response (see `_request_conditional` / `paginate_cached`
    in `github_client.py`), so the on-disk cost is bounded by the
    number of distinct `(repo, endpoint, since-cursor)` tuples.

    The lock guards both reads and writes because ingest workers share
    a single `sqlite3.Connection` and SQLite's per-connection cursor
    state can otherwise interleave badly under concurrent access.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)

    def get(self, url: str) -> CacheEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT etag, last_modified, body FROM http_cache WHERE url = ?",
                (url,),
            ).fetchone()
        if row is None:
            return None
        return CacheEntry(etag=row[0], last_modified=row[1], body=row[2])

    def put(
        self,
        url: str,
        *,
        etag: str | None,
        last_modified: str | None,
        body: bytes,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO http_cache (url, etag, last_modified, body) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(url) DO UPDATE SET "
                "etag=excluded.etag, "
                "last_modified=excluded.last_modified, "
                "body=excluded.body, "
                "fetched_at=datetime('now')",
                (url, etag, last_modified, body),
            )
