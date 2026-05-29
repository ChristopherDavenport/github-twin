"""HTTP validator + body cache for conditional GitHub requests.

This module defines the seam between the HTTP client (which decides
when to send `If-None-Match` / `If-Modified-Since` and how to handle a
`304`) and whatever backing store holds the cache. The Protocol is
deliberately tiny:

  - `get(url)` → previously-seen validators + body, or None on miss
  - `put(url, etag=..., last_modified=..., body=...)` → store the
    response from a 200 so the next request can be conditional

The Protocol lives here (in `ingest/`) so the client can import it
without pulling in `sqlite3` or any other backend. Backends (e.g.
`store.sqlite_http_cache.SqliteHttpCache`) implement this Protocol
and are wired in by callers that own the DB connection.

Why 304 matters: GitHub's REST docs state that a 304 response to an
authorized conditional request does NOT count against the primary
rate-limit budget. That makes a populated cache effectively free for
unchanged resources — the exact case where ingest spends most of its
budget today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CacheEntry:
    """What the cache returns about a previously-seen URL.

    `body` is the raw response bytes — exactly what `httpx.Response.content`
    held on the 200 that produced this entry. The client round-trips them
    on a 304 so the caller never needs to know caching happened.

    `etag` and `last_modified` are the validator headers we should echo
    on the next request. Both may be `None` (some endpoints return one
    and not the other); the client picks whichever exist.
    """

    etag: str | None
    last_modified: str | None
    body: bytes


class HttpCache(Protocol):
    """Validator + body store for conditional HTTP requests.

    Implementations are responsible for:

    - **Keying**: cache rows are keyed by **canonical URL including
      query string**. Implementations MUST NOT normalize query order,
      strip parameters, or otherwise collapse distinct URLs — the
      caller treats `?a=1&b=2` and `?b=2&a=1` as different cache slots
      because GitHub treats them as different cache keys. The
      canonical URL is whatever the client passes to `get`/`put`,
      which is `str(httpx.Request.url)` at call time.

    - **Thread safety**: parallel ingest workers share one client and
      therefore one cache. `get` and `put` must be safe to call
      concurrently from multiple threads.

    - **Body durability**: a `put` must persist `body` such that a
      subsequent `get(url)` from a different process can still
      reproduce the response on a 304. In-memory adapters are fine
      for tests but won't help cross-run.

    The cache is never consulted for non-GET methods or for
    non-cacheable status codes — the client decides what to store and
    when. Returning `None` from `get` always falls back to a normal
    request (no harm beyond a wasted round-trip).
    """

    def get(self, url: str) -> CacheEntry | None: ...

    def put(
        self,
        url: str,
        *,
        etag: str | None,
        last_modified: str | None,
        body: bytes,
    ) -> None: ...


class NoopHttpCache:
    """Default implementation for callers that don't want caching.

    Stateless, thread-safe by construction. The `GitHubClient` defaults
    to this when no cache is injected — that keeps cache-less call
    sites (the `/user` probe in `cli.py`, `target.discover_user`) free
    of any DB dependency.
    """

    def get(self, url: str) -> CacheEntry | None:
        return None

    def put(
        self,
        url: str,
        *,
        etag: str | None,
        last_modified: str | None,
        body: bytes,
    ) -> None:
        return None
