"""Tests for `ingest_reviews` — user-mode PR walk that filters to one login.

Mirrors `test_ingest_reviews_org.py` but exercises the user-mode path:
- Comments are filtered to `username`; other authors are dropped.
- The PR-level `decision` artifact column is set from the user's review
  state (drives `predict_review_outcome`).
- Cursor advances via `q.set_cursor("reviews", ...)` after the sync.
- Content-hash short-circuit makes re-running idempotent at chunk level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github_twin.config import IngestCfg
from github_twin.ingest.cache import RawCache
from github_twin.ingest.reviews import ingest_reviews
from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    seed_target(db)
    yield db
    db.close()


class FakeGH:
    """Routes `/search/issues` + per-PR subresources from in-memory fixtures."""

    token = "fake-token"

    def __init__(
        self,
        prs: list[dict[str, Any]] | None = None,
        review_comments: dict[tuple[str, int], list[dict[str, Any]]] | None = None,
        reviews: dict[tuple[str, int], list[dict[str, Any]]] | None = None,
        issue_comments: dict[tuple[str, int], list[dict[str, Any]]] | None = None,
    ):
        self.prs = prs or []
        self.review_comments = review_comments or {}
        self.reviews = reviews or {}
        self.issue_comments = issue_comments or {}

    def paginate(self, path: str, *, params: dict | None = None):
        if path == "/search/issues":
            yield from self.prs
            return
        # /repos/<owner>/<name>/pulls/<n>/comments
        if "/pulls/" in path and path.endswith("/comments"):
            full, _, rest = path.removeprefix("/repos/").partition("/pulls/")
            n = int(rest.split("/")[0])
            yield from self.review_comments.get((full, n), [])
            return
        # /repos/<owner>/<name>/pulls/<n>/reviews
        if path.endswith("/reviews"):
            full, _, rest = path.removeprefix("/repos/").partition("/pulls/")
            n = int(rest.split("/")[0])
            yield from self.reviews.get((full, n), [])
            return
        # /repos/<owner>/<name>/issues/<n>/comments
        if "/issues/" in path and path.endswith("/comments"):
            full, _, rest = path.removeprefix("/repos/").partition("/issues/")
            n = int(rest.split("/")[0])
            yield from self.issue_comments.get((full, n), [])
            return
        raise AssertionError(f"unexpected paginate: {path}")


def _pr(repo_full: str, n: int, updated: str, title: str = "x") -> dict:
    return {
        "number": n,
        "updated_at": updated,
        "title": title,
        "state": "open",
        "html_url": f"https://gh/{repo_full}/pull/{n}",
        "repository_url": f"https://api.github.com/repos/{repo_full}",
        "user": {"login": "anyone"},
        "created_at": "2024-01-01T00:00:00Z",
    }


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


def _review(login: str, state: str, submitted: str = "2024-02-01T00:00:00Z") -> dict:
    return {"user": {"login": login}, "state": state, "submitted_at": submitted}


def test_user_mode_filters_comments_by_login(conn, tmp_path: Path):
    """Only the target user's comments are stored; other authors are dropped."""
    gh = FakeGH(
        prs=[_pr("org/r", 1, "2024-03-01T00:00:00Z")],
        review_comments={
            ("org/r", 1): [
                _rc(101, "me", "use Set instead of List"),
                _rc(102, "alice", "drop this"),
            ]
        },
        reviews={("org/r", 1): []},
        issue_comments={
            ("org/r", 1): [
                _ic(201, "me", "lgtm modulo nit"),
                _ic(202, "bob", "thanks"),
            ]
        },
    )
    stats = ingest_reviews(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_path / "raw"),
        username="me",
        cfg=IngestCfg(),
        target_id=1,
    )
    assert stats.prs_seen == 1
    assert stats.review_comments == 1
    assert stats.issue_comments == 1

    rows = conn.execute(
        "SELECT kind, author_login FROM artifact "
        "WHERE kind IN ('review_comment','issue_comment') "
        "ORDER BY author_login, kind"
    ).fetchall()
    assert [(r["kind"], r["author_login"]) for r in rows] == [
        ("issue_comment", "me"),
        ("review_comment", "me"),
    ]


def test_user_mode_sets_pr_decision_from_user_review(conn, tmp_path: Path):
    """`decision` column on PR artifact comes from the target user's most
    recent review state — drives `predict_review_outcome` aggregation."""
    gh = FakeGH(
        prs=[_pr("org/r", 5, "2024-03-01T00:00:00Z", title="auth refactor")],
        reviews={
            ("org/r", 5): [
                _review("me", "approved", "2024-02-20T00:00:00Z"),
                _review("alice", "changes_requested", "2024-02-21T00:00:00Z"),
            ]
        },
    )
    ingest_reviews(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_path / "raw"),
        username="me",
        cfg=IngestCfg(),
        target_id=1,
    )
    row = conn.execute(
        "SELECT decision FROM artifact WHERE kind='pr' AND external_id='org/r#5'"
    ).fetchone()
    assert row is not None
    assert row["decision"] == "approved"


def test_user_mode_advances_reviews_cursor(conn, tmp_path: Path):
    """Cursor moves to the newest `updated_at` (+1s) after a successful walk."""
    gh = FakeGH(
        prs=[
            _pr("org/r", 1, "2024-03-01T00:00:00Z"),
            _pr("org/r", 2, "2024-04-01T00:00:00Z"),
        ],
        reviews={("org/r", 1): [], ("org/r", 2): []},
    )
    ingest_reviews(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_path / "raw"),
        username="me",
        cfg=IngestCfg(),
        target_id=1,
    )
    cursor = q.get_cursor(conn, "reviews", target_id=1)
    assert cursor is not None
    # Newest is 2024-04-01T00:00:00Z; _bump_iso adds 1s.
    assert cursor.startswith("2024-04-01T00:00:01")


def test_user_mode_is_idempotent_via_content_hash(conn, tmp_path: Path):
    """Re-running the same sync writes no new chunks (content_hash short-circuit)."""
    gh = FakeGH(
        prs=[_pr("org/r", 1, "2024-03-01T00:00:00Z")],
        review_comments={("org/r", 1): [_rc(101, "me", "nit: rename")]},
        reviews={("org/r", 1): []},
        issue_comments={("org/r", 1): [_ic(201, "me", "lgtm")]},
    )
    cfg = IngestCfg()
    ingest_reviews(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_path / "raw"),
        username="me",
        cfg=cfg,
        target_id=1,
    )
    n1 = conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    c1 = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]

    # Re-running with no upstream changes shouldn't grow either table.
    # The cursor is now ahead of all PRs so the search returns nothing —
    # to actually re-touch the same PR we reset the cursor first.
    q.set_cursor(conn, "reviews", "2020-01-01T00:00:00Z", target_id=1)
    ingest_reviews(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_path / "raw"),
        username="me",
        cfg=cfg,
        target_id=1,
    )
    n2 = conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    c2 = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    assert n1 == n2
    assert c1 == c2


def test_user_mode_respects_exclude_repos(conn, tmp_path: Path):
    """`cfg.exclude_repos` patterns drop PRs upfront, no fetch attempted."""
    gh = FakeGH(
        prs=[
            _pr("org/keep", 1, "2024-03-01T00:00:00Z"),
            _pr("org/drop", 2, "2024-03-01T00:00:00Z"),
        ],
        review_comments={("org/keep", 1): [_rc(101, "me", "hi")]},
        reviews={("org/keep", 1): []},
        issue_comments={("org/keep", 1): []},
    )
    stats = ingest_reviews(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_path / "raw"),
        username="me",
        cfg=IngestCfg(exclude_repos=["org/drop"]),
        target_id=1,
    )
    assert stats.prs_seen == 1
    repos_seen = {
        r["repo"]
        for r in conn.execute(
            "SELECT DISTINCT repo FROM artifact WHERE repo IS NOT NULL"
        ).fetchall()
    }
    assert repos_seen == {"org/keep"}
