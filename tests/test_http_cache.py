"""Tests for the HttpCache Protocol and its backends.

Pins:
- `NoopHttpCache` always misses and never raises on put.
- `SqliteHttpCache` (against `:memory:`) round-trips an entry and
  upserts on a repeat put for the same URL.
- The Protocol's contract — body is round-tripped verbatim — holds for
  both backends.
"""

from __future__ import annotations

import sqlite3

from github_twin.ingest.http_cache import CacheEntry, NoopHttpCache
from github_twin.store.sqlite_http_cache import SqliteHttpCache


def test_noop_cache_always_misses():
    cache = NoopHttpCache()
    cache.put("https://example/x", etag='"v1"', last_modified=None, body=b"hello")
    assert cache.get("https://example/x") is None


def test_sqlite_cache_round_trips_entry():
    conn = sqlite3.connect(":memory:")
    cache = SqliteHttpCache(conn)

    cache.put(
        "https://api.github.com/repos/o/r/pulls/comments?since=2024-01-01",
        etag='"abc"',
        last_modified="Wed, 01 Jan 2025 00:00:00 GMT",
        body=b'[{"id":1}]',
    )

    entry = cache.get("https://api.github.com/repos/o/r/pulls/comments?since=2024-01-01")
    assert isinstance(entry, CacheEntry)
    assert entry.etag == '"abc"'
    assert entry.last_modified == "Wed, 01 Jan 2025 00:00:00 GMT"
    assert entry.body == b'[{"id":1}]'


def test_sqlite_cache_upserts_on_repeat_put():
    """Same URL, new validators + body → row is replaced, not duplicated."""
    conn = sqlite3.connect(":memory:")
    cache = SqliteHttpCache(conn)

    url = "https://api.github.com/repos/o/r/issues/comments"
    cache.put(url, etag='"v1"', last_modified=None, body=b"v1-body")
    cache.put(url, etag='"v2"', last_modified=None, body=b"v2-body")

    entry = cache.get(url)
    assert entry is not None
    assert entry.etag == '"v2"'
    assert entry.body == b"v2-body"

    # And only one row exists.
    rows = conn.execute("SELECT COUNT(*) FROM http_cache").fetchone()
    assert rows[0] == 1


def test_sqlite_cache_distinguishes_query_strings():
    """Different `since=` cursors are different cache slots."""
    conn = sqlite3.connect(":memory:")
    cache = SqliteHttpCache(conn)

    base = "https://api.github.com/repos/o/r/pulls/comments"
    cache.put(f"{base}?since=2024-01-01", etag='"a"', last_modified=None, body=b"A")
    cache.put(f"{base}?since=2024-06-01", etag='"b"', last_modified=None, body=b"B")

    a = cache.get(f"{base}?since=2024-01-01")
    b = cache.get(f"{base}?since=2024-06-01")
    assert a is not None and a.body == b"A"
    assert b is not None and b.body == b"B"


def test_sqlite_cache_get_miss_returns_none():
    conn = sqlite3.connect(":memory:")
    cache = SqliteHttpCache(conn)
    assert cache.get("https://api.github.com/never-fetched") is None
