"""Tests for `GitHubClient.get_json_cached` / `paginate_cached`.

Uses an `httpx.MockTransport` to script responses without hitting the
network. Pins:
- 200 stores the entry; the next call sends `If-None-Match`.
- 304 returns the cached body verbatim (re-parses without re-fetching).
- `cache=None` (default → `NoopHttpCache`) behaves identically to a
  non-conditional `paginate` — no extra headers, no extra calls.
- `paginate_cached` short-circuits subsequent pages when page 1 is 304.
"""

from __future__ import annotations

import json

import httpx

from github_twin.ingest.github_client import GitHubClient
from github_twin.ingest.http_cache import CacheEntry, HttpCache


class _DictCache:
    """Minimal in-memory HttpCache for tests."""

    def __init__(self) -> None:
        self.entries: dict[str, CacheEntry] = {}

    def get(self, url: str) -> CacheEntry | None:
        return self.entries.get(url)

    def put(self, url: str, *, etag, last_modified, body) -> None:
        self.entries[url] = CacheEntry(etag=etag, last_modified=last_modified, body=body)


def _client_with(transport: httpx.MockTransport, *, cache: HttpCache | None = None) -> GitHubClient:
    """GitHubClient with a stub transport and a fake token."""
    gh = GitHubClient(token="t", cache=cache)
    gh._client = httpx.Client(transport=transport)  # type: ignore[assignment]
    return gh


def test_first_call_stores_then_second_sends_if_none_match():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "If-None-Match" not in request.headers:
            return httpx.Response(
                200,
                headers={"ETag": '"abc"'},
                content=b'[{"id": 1}]',
            )
        # Second call: client echoed our ETag → respond 304.
        return httpx.Response(304, headers={"ETag": '"abc"'})

    cache = _DictCache()
    gh = _client_with(httpx.MockTransport(handler), cache=cache)

    data1 = gh.get_json_cached("/repos/o/r/pulls/comments", params={"since": "2024-01-01"})
    assert data1 == [{"id": 1}]
    assert any(
        url.startswith("https://api.github.com/repos/o/r/pulls/comments") for url in cache.entries
    )

    data2 = gh.get_json_cached("/repos/o/r/pulls/comments", params={"since": "2024-01-01"})
    assert data2 == [{"id": 1}]
    # Second request must have sent the validator.
    assert seen[-1].headers.get("If-None-Match") == '"abc"'


def test_304_returns_cached_body_without_reparsing_network():
    """When the cache has an entry and the server returns 304, the
    parsed result must come from the cached bytes, not from the empty
    304 body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(304, headers={"ETag": '"abc"'})

    cache = _DictCache()
    # Pre-seed the cache with a known body keyed on the URL httpx will
    # canonicalize to (params get folded in deterministically).
    seed_url = "https://api.github.com/repos/o/r/issues/comments?since=2024-01-01"
    cache.entries[seed_url] = CacheEntry(etag='"abc"', last_modified=None, body=b'[{"id": 42}]')
    gh = _client_with(httpx.MockTransport(handler), cache=cache)

    data = gh.get_json_cached("/repos/o/r/issues/comments", params={"since": "2024-01-01"})
    assert data == [{"id": 42}]


def test_no_cache_means_no_conditional_headers():
    """Default (no cache) must behave identically to the plain client —
    no `If-None-Match` ever leaves the wire."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"ETag": '"abc"'},
            content=b'[{"id": 1}]',
        )

    gh = _client_with(httpx.MockTransport(handler))  # no cache → NoopHttpCache
    gh.get_json_cached("/repos/o/r/pulls/comments")
    gh.get_json_cached("/repos/o/r/pulls/comments")
    assert all("If-None-Match" not in r.headers for r in seen)


def test_paginate_cached_short_circuits_when_page1_is_304():
    """A 304 on page 1 means nothing new in the whole result set under
    our `direction=desc&since=...` contract — so subsequent pages must
    NOT be requested. We replay the cached body verbatim."""
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(304, headers={"ETag": '"abc"'})

    cache = _DictCache()
    url = (
        "https://api.github.com/repos/o/r/pulls/comments"
        "?sort=updated&direction=desc&since=2024-01-01&per_page=100"
    )
    cache.entries[url] = CacheEntry(
        etag='"abc"',
        last_modified=None,
        body=json.dumps([{"id": 1}, {"id": 2}]).encode(),
    )
    gh = _client_with(httpx.MockTransport(handler), cache=cache)

    items = list(
        gh.paginate_cached(
            "/repos/o/r/pulls/comments",
            params={
                "sort": "updated",
                "direction": "desc",
                "since": "2024-01-01",
                "per_page": 100,
            },
        )
    )
    assert items == [{"id": 1}, {"id": 2}]
    # Exactly one HTTP call — the conditional GET that returned 304.
    assert len(requests_seen) == 1


def test_paginate_cached_follows_link_when_page1_is_fresh():
    """Two-page result: page 1 returns 200 + Link rel=next; page 2 must
    be fetched (no caching on subsequent pages)."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "page=2" in str(request.url):
            return httpx.Response(200, content=b'[{"id": 2}]')
        return httpx.Response(
            200,
            headers={
                "ETag": '"page1"',
                "Link": '<https://api.github.com/repos/o/r/pulls/comments?page=2>; rel="next"',
            },
            content=b'[{"id": 1}]',
        )

    gh = _client_with(httpx.MockTransport(handler), cache=_DictCache())
    items = list(gh.paginate_cached("/repos/o/r/pulls/comments"))
    assert items == [{"id": 1}, {"id": 2}]
    assert any("page=2" in c for c in calls)
