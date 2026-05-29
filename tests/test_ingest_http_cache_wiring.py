"""Pin the conditional-GET wiring for ingest call sites that opted in.

The end-to-end ETag behaviour of `get_json_cached` / `paginate_cached`
is exercised in `test_github_client_conditional.py`. These tests assert
the *integration*: that each opted-in caller actually routes through
the conditional method (not the unconditional one) and respects the
cached body on a 304.

Call sites covered:
- `_fetch_repo_pushed_at` in `ingest/commits.py`
- `_iter_prs_until_cursor` in `ingest/reviews.py`
- `_fetch_pr_payload`'s per-PR review-comments fetch in `ingest/reviews.py`
"""

from __future__ import annotations

import json

import httpx

from github_twin.ingest.commits import _fetch_repo_pushed_at
from github_twin.ingest.github_client import GitHubClient
from github_twin.ingest.http_cache import CacheEntry
from github_twin.ingest.reviews import _fetch_pr_payload, _iter_prs_until_cursor


class _DictCache:
    """In-memory HttpCache mirror of the test_github_client_conditional fake."""

    def __init__(self) -> None:
        self.entries: dict[str, CacheEntry] = {}

    def get(self, url: str) -> CacheEntry | None:
        return self.entries.get(url)

    def put(self, url: str, *, etag, last_modified, body) -> None:
        self.entries[url] = CacheEntry(etag=etag, last_modified=last_modified, body=body)


def _client_with(transport: httpx.MockTransport, *, cache=None) -> GitHubClient:
    gh = GitHubClient(token="t", cache=cache)
    gh._client = httpx.Client(transport=transport)  # type: ignore[assignment]
    return gh


def test_fetch_repo_pushed_at_sends_if_none_match_on_second_call():
    """The /repos/{r} pushed_at probe must use the conditional variant —
    on the second sync it should echo the stored ETag, so an unchanged
    repo returns 304 and never re-downloads the metadata."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "If-None-Match" not in request.headers:
            return httpx.Response(
                200,
                headers={"ETag": '"r1"'},
                content=json.dumps({"pushed_at": "2024-05-01T00:00:00Z"}).encode(),
            )
        return httpx.Response(304, headers={"ETag": '"r1"'})

    cache = _DictCache()
    gh = _client_with(httpx.MockTransport(handler), cache=cache)

    first = _fetch_repo_pushed_at(gh, ["org/r"], max_workers=1)
    assert first == {"org/r": "2024-05-01T00:00:00Z"}
    # Cache populated after first sync.
    assert any(k.endswith("/repos/org/r") for k in cache.entries)

    second = _fetch_repo_pushed_at(gh, ["org/r"], max_workers=1)
    # On 304, the cached JSON is replayed and pushed_at survives.
    assert second == {"org/r": "2024-05-01T00:00:00Z"}
    assert seen[-1].headers.get("If-None-Match") == '"r1"'


def test_iter_prs_until_cursor_short_circuits_on_304():
    """`/repos/{r}/pulls?sort=updated&direction=desc` cached at page 1.
    On 304 we replay the cached PR list — no second HTTP call should
    fire (the helper consumes the iterator and stops when updated_at
    falls past the cursor or the iterator ends)."""
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(304, headers={"ETag": '"p1"'})

    cache = _DictCache()
    # The canonical URL httpx will produce, including all params.
    url = (
        "https://api.github.com/repos/org/r/pulls"
        "?state=all&sort=updated&direction=desc&per_page=100"
    )
    cached_prs = [
        {"number": 9, "updated_at": "2024-04-10T00:00:00Z"},
        {"number": 8, "updated_at": "2024-04-05T00:00:00Z"},
    ]
    cache.entries[url] = CacheEntry(
        etag='"p1"', last_modified=None, body=json.dumps(cached_prs).encode()
    )
    gh = _client_with(httpx.MockTransport(handler), cache=cache)

    items = list(_iter_prs_until_cursor(gh, repo_full="org/r", cursor="2024-04-01T00:00:00Z"))
    assert [p["number"] for p in items] == [9, 8]
    # Exactly one HTTP call: the conditional GET that returned 304.
    assert len(requests_seen) == 1


def test_fetch_pr_payload_review_comments_send_if_none_match_on_second_call():
    """The per-PR `/pulls/{n}/comments` fetch must be conditional. Run
    `_fetch_pr_payload` twice with the same PR and assert that the
    second call echoes the stored ETag.

    The other two sub-endpoints (`/reviews`, `/issues/{n}/comments`)
    are still unconditional by design and respond 200 every time."""
    seen_paths: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen_paths.append((url, dict(request.headers)))
        if "/pulls/42/comments" in url:
            if "If-None-Match" not in request.headers:
                return httpx.Response(
                    200,
                    headers={"ETag": '"c42"'},
                    content=b"[]",
                )
            return httpx.Response(304, headers={"ETag": '"c42"'})
        # /reviews and /issues/42/comments — always empty list.
        return httpx.Response(200, content=b"[]")

    cache = _DictCache()
    gh = _client_with(httpx.MockTransport(handler), cache=cache)
    pr_item = {"number": 42}

    p1 = _fetch_pr_payload(gh, "org/r", pr_item)
    assert p1 is not None
    p2 = _fetch_pr_payload(gh, "org/r", pr_item)
    assert p2 is not None

    # Find the *second* /pulls/42/comments request and confirm the
    # validator header was attached.
    comment_calls = [(u, h) for (u, h) in seen_paths if "/pulls/42/comments" in u]
    assert len(comment_calls) >= 2
    assert comment_calls[-1][1].get("if-none-match") == '"c42"'
