"""Tests for `ingest_reviews_org` — per-repo PR walk that retains ALL authors.

We verify:
- Comments from multiple authors are stored (no `username` filter applies).
- The cursor stops the walk at PRs <= `last_reviews_at`.
- Per-repo `last_reviews_at` advances after the walk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github_twin.config import IngestCfg
from github_twin.ingest.cache import RawCache
from github_twin.ingest.reviews import ingest_reviews_org
from github_twin.store import queries as q
from github_twin.store.db import open_db


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    yield db
    db.close()


class FakeGH:
    """Routes /repos/{r}/pulls + per-PR subresources from in-memory fixtures."""

    def __init__(self, repos: dict[str, dict[str, Any]]):
        self.repos = repos

    def paginate(self, path: str, *, params: dict | None = None):
        # /repos/<repo>/pulls
        if path.endswith("/pulls") and path.count("/") == 4:
            full = path.removeprefix("/repos/").removesuffix("/pulls")
            yield from self.repos.get(full, {}).get("prs", [])
            return
        # /repos/<repo>/pulls/<n>/comments
        if "/pulls/" in path and path.endswith("/comments"):
            full, _, rest = path.removeprefix("/repos/").partition("/pulls/")
            n = int(rest.split("/")[0])
            yield from self.repos[full]["review_comments"].get(n, [])
            return
        # /repos/<repo>/pulls/<n>/reviews
        if path.endswith("/reviews"):
            full, _, rest = path.removeprefix("/repos/").partition("/pulls/")
            n = int(rest.split("/")[0])
            yield from self.repos[full]["reviews"].get(n, [])
            return
        # /repos/<repo>/issues/<n>/comments
        if "/issues/" in path and path.endswith("/comments"):
            full, _, rest = path.removeprefix("/repos/").partition("/issues/")
            n = int(rest.split("/")[0])
            yield from self.repos[full]["issue_comments"].get(n, [])
            return
        raise AssertionError(f"unexpected paginate: {path}")


def _pr(n: int, updated: str, title: str = "x") -> dict:
    return {"number": n, "updated_at": updated, "title": title, "state": "open", "html_url": ""}


def _rc(id_: int, login: str, body: str) -> dict:
    return {
        "id": id_,
        "user": {"login": login},
        "body": body,
        "path": "src/x.py",
        "diff_hunk": "@@ -1,1 +1,2 @@\n+new",
        "created_at": "2024-02-01T00:00:00Z",
        "html_url": f"https://gh/comment/{id_}",
    }


def _ic(id_: int, login: str, body: str) -> dict:
    return {
        "id": id_,
        "user": {"login": login},
        "body": body,
        "created_at": "2024-02-01T00:00:00Z",
        "html_url": f"https://gh/issue_comment/{id_}",
    }


def test_org_reviews_keeps_all_authors(conn, tmp_path: Path):
    q.upsert_repo(
        conn,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    gh = FakeGH(
        repos={
            "org/r": {
                "prs": [_pr(1, "2024-03-01T00:00:00Z")],
                "review_comments": {
                    1: [
                        _rc(101, "alice", "use Set instead of List"),
                        _rc(102, "bob", "this needs a test"),
                    ],
                },
                "reviews": {1: []},
                "issue_comments": {1: [_ic(201, "carol", "lgtm")]},
            },
        },
    )
    stats = ingest_reviews_org(conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg())
    assert stats.prs_seen == 1
    assert stats.review_comments == 2
    assert stats.issue_comments == 1

    rows = conn.execute(
        "SELECT kind, author_login FROM artifact "
        "WHERE kind IN ('review_comment','issue_comment') "
        "ORDER BY author_login"
    ).fetchall()
    assert [(r["kind"], r["author_login"]) for r in rows] == [
        ("review_comment", "alice"),
        ("review_comment", "bob"),
        ("issue_comment", "carol"),
    ]


def test_org_reviews_stops_at_cursor(conn, tmp_path: Path):
    q.upsert_repo(
        conn,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    q.set_repo_cursor(conn, full_name="org/r", reviews_at="2024-02-15T00:00:00Z")

    gh = FakeGH(
        repos={
            "org/r": {
                "prs": [
                    _pr(2, "2024-03-01T00:00:00Z"),  # newer than cursor: kept
                    _pr(1, "2024-02-01T00:00:00Z"),  # older: walker stops here
                ],
                "review_comments": {
                    2: [_rc(101, "alice", "hi")],
                    1: [_rc(999, "alice", "should not be touched")],
                },
                "reviews": {1: [], 2: []},
                "issue_comments": {1: [], 2: []},
            },
        },
    )
    stats = ingest_reviews_org(conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg())
    assert stats.prs_seen == 1
    # Only the new PR's comments landed; the older PR's comment was filtered.
    ids = {
        r["external_id"]
        for r in conn.execute(
            "SELECT external_id FROM artifact WHERE kind='review_comment'"
        ).fetchall()
    }
    assert ids == {"101"}


def test_org_reviews_advances_cursor(conn, tmp_path: Path):
    q.upsert_repo(
        conn,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    gh = FakeGH(
        repos={"org/r": {"prs": [], "review_comments": {}, "reviews": {}, "issue_comments": {}}}
    )
    ingest_reviews_org(conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg())
    assert q.get_repo(conn, "org/r")["last_reviews_at"] is not None
