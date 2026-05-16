"""Tests for `ingest/identity.resolve_login`.

The resolver is the linchpin of the git-local commits ingest path in org mode:
it converts the email on a local commit object into the GitHub login we store
on `artifact.author_login`. We assert:

- DB-cached hits short-circuit (no API call)
- Cached misses (login IS NULL) are honored (no API call)
- `*@users.noreply.github.com` addresses resolve locally without an API call
- Real-world emails fall through to /search/commits and the result is cached
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github_twin.ingest.identity import resolve_login
from github_twin.store import queries as q
from github_twin.store.db import open_db


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    yield db
    db.close()


class CountingGH:
    """GitHubClient stand-in that records every paginate() call."""

    def __init__(self, results: list[dict[str, Any]]):
        self._results = results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def paginate(self, path: str, *, params: dict[str, Any] | None = None):
        self.calls.append((path, params or {}))
        yield from self._results


def test_noreply_resolves_without_api(conn):
    gh = CountingGH([])
    assert resolve_login(conn, gh, "12345+octocat@users.noreply.github.com") == "octocat"
    # Plain (no numeric id) form also supported.
    assert resolve_login(conn, gh, "octocat@users.noreply.github.com") == "octocat"
    assert gh.calls == []
    # And the resolution is cached, so the next call doesn't even parse.
    login, resolved = q.get_email_login(conn, "12345+octocat@users.noreply.github.com")
    assert resolved and login == "octocat"


def test_api_fallback_caches_result(conn):
    gh = CountingGH([{"author": {"login": "alice"}}])
    assert resolve_login(conn, gh, "alice@example.com") == "alice"
    assert len(gh.calls) == 1
    # Cache hit on second call: no more API traffic.
    assert resolve_login(conn, gh, "alice@example.com") == "alice"
    assert len(gh.calls) == 1


def test_api_miss_is_cached_as_null(conn):
    gh = CountingGH([])  # empty search result
    assert resolve_login(conn, gh, "nobody@example.com") is None
    assert len(gh.calls) == 1
    # Repeated lookups don't re-query.
    assert resolve_login(conn, gh, "nobody@example.com") is None
    assert len(gh.calls) == 1
    login, resolved = q.get_email_login(conn, "nobody@example.com")
    assert resolved is True
    assert login is None


def test_email_lookup_is_case_insensitive(conn):
    gh = CountingGH([{"author": {"login": "bob"}}])
    assert resolve_login(conn, gh, "Bob@Example.Com") == "bob"
    # Hit comes back through the cache regardless of input case.
    assert resolve_login(conn, gh, "bob@example.com") == "bob"
    assert len(gh.calls) == 1


def test_resolve_login_handles_none_email(conn):
    gh = CountingGH([])
    assert resolve_login(conn, gh, None) is None
    assert resolve_login(conn, gh, "") is None
    assert gh.calls == []


def test_gh_none_disables_api_fallback(conn):
    # Cache-only mode: unknown email returns None without raising.
    assert resolve_login(conn, None, "unknown@example.com") is None
