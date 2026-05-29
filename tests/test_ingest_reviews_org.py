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
from tests.conftest import seed_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    seed_target(db)
    yield db
    db.close()


class FakeGH:
    """Routes /repos/{r}/pulls + per-PR subresources from in-memory fixtures."""

    token = "fake-token"

    def __init__(self, repos: dict[str, dict[str, Any]]):
        self.repos = repos
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_json(self, path: str, *, params: dict | None = None):
        # Backs `_fetch_repo_pushed_at`. Default: report no pushed_at so
        # `_needs_walk` returns True and the existing test behaviour is
        # preserved (walk every repo).
        if path.startswith("/repos/") and "/" not in path[len("/repos/") :]:
            return {}
        # /repos/<owner>/<name> — single repo info
        rest = path.removeprefix("/repos/")
        if rest.count("/") == 1:
            return self.repos.get(rest, {}).get("info", {})
        raise AssertionError(f"unexpected get_json: {path}")

    def get_json_cached(self, path: str, *, params: dict | None = None):
        # `_fetch_repo_pushed_at` now uses the conditional variant. The
        # fake doesn't model 304s; route to the unconditional path.
        return self.get_json(path, params=params)

    def paginate_cached(self, path: str, *, params: dict | None = None):
        # The fake doesn't model conditional requests; the new bulk
        # helpers in `reviews.py` call this method, so route it to the
        # same fixture lookup as `paginate`.
        yield from self.paginate(path, params=params)

    def paginate(self, path: str, *, params: dict | None = None):
        # Track call counts so tests can assert per-endpoint cost.
        self.calls.append((path, dict(params or {})))
        # /repos/<repo>/pulls
        if path.endswith("/pulls") and path.count("/") == 4:
            full = path.removeprefix("/repos/").removesuffix("/pulls")
            yield from self.repos.get(full, {}).get("prs", [])
            return
        # /repos/<repo>/pulls/comments  (repo-wide bulk review comments)
        if path.endswith("/pulls/comments"):
            full = path.removeprefix("/repos/").removesuffix("/pulls/comments")
            since = (params or {}).get("since")
            for rc in self.repos.get(full, {}).get("bulk_review_comments", []):
                if since and rc.get("updated_at", "") <= since:
                    continue
                yield rc
            return
        # /repos/<repo>/issues/comments  (repo-wide bulk issue comments)
        if path.endswith("/issues/comments"):
            full = path.removeprefix("/repos/").removesuffix("/issues/comments")
            since = (params or {}).get("since")
            for ic in self.repos.get(full, {}).get("bulk_issue_comments", []):
                if since and ic.get("updated_at", "") <= since:
                    continue
                yield ic
            return
        # /repos/<repo>/pulls/<n>/comments  (per-PR, no longer used by org)
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
        # /repos/<repo>/issues/<n>/comments  (per-PR, no longer used by org)
        if "/issues/" in path and path.endswith("/comments"):
            full, _, rest = path.removeprefix("/repos/").partition("/issues/")
            n = int(rest.split("/")[0])
            yield from self.repos[full]["issue_comments"].get(n, [])
            return
        raise AssertionError(f"unexpected paginate: {path}")


def _pr(n: int, updated: str, title: str = "x") -> dict:
    return {"number": n, "updated_at": updated, "title": title, "state": "open", "html_url": ""}


def _rc(
    id_: int,
    login: str,
    body: str,
    *,
    pr: int = 1,
    repo: str = "org/r",
    updated_at: str = "2024-03-01T00:00:00Z",
) -> dict:
    return {
        "id": id_,
        "user": {"login": login},
        "body": body,
        "path": "src/x.py",
        "diff_hunk": "@@ -1,1 +1,2 @@\n+new",
        "created_at": "2024-02-01T00:00:00Z",
        "updated_at": updated_at,
        "html_url": f"https://gh/comment/{id_}",
        "pull_request_url": f"https://api.github.com/repos/{repo}/pulls/{pr}",
    }


def _ic(
    id_: int,
    login: str,
    body: str,
    *,
    pr: int = 1,
    repo: str = "org/r",
    updated_at: str = "2024-03-01T00:00:00Z",
) -> dict:
    return {
        "id": id_,
        "user": {"login": login},
        "body": body,
        "created_at": "2024-02-01T00:00:00Z",
        "updated_at": updated_at,
        "html_url": f"https://gh/issue_comment/{id_}",
        "issue_url": f"https://api.github.com/repos/{repo}/issues/{pr}",
    }


def _with_bulk(repos: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Synthesize the repo-wide bulk lists from the per-PR maps so existing
    fixtures keep working with the new bulk-fetch worker. Re-keys each
    comment's `pull_request_url` / `issue_url` from the dict key so a
    factory default like `pr=1` doesn't fight the fixture's true PR
    number."""
    for repo_full, cfg in repos.items():
        if "bulk_review_comments" not in cfg:
            flat: list[dict[str, Any]] = []
            for pr_n, items in (cfg.get("review_comments") or {}).items():
                for it in items:
                    it["pull_request_url"] = (
                        f"https://api.github.com/repos/{repo_full}/pulls/{pr_n}"
                    )
                    flat.append(it)
            cfg["bulk_review_comments"] = flat
        if "bulk_issue_comments" not in cfg:
            flat = []
            for pr_n, items in (cfg.get("issue_comments") or {}).items():
                for it in items:
                    it["issue_url"] = f"https://api.github.com/repos/{repo_full}/issues/{pr_n}"
                    flat.append(it)
            cfg["bulk_issue_comments"] = flat
    return repos


def test_org_reviews_keeps_all_authors(conn, tmp_path: Path):
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    gh = FakeGH(
        repos=_with_bulk(
            {
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
            }
        ),
    )
    stats = ingest_reviews_org(
        conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg(), target_id=1
    )
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
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    q.set_repo_cursor(conn, target_id=1, full_name="org/r", reviews_at="2024-02-15T00:00:00Z")

    gh = FakeGH(
        repos=_with_bulk(
            {
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
            }
        ),
    )
    stats = ingest_reviews_org(
        conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg(), target_id=1
    )
    assert stats.prs_seen == 1
    # Only the new PR's comments landed; the older PR's comment was filtered.
    ids = {
        r["external_id"]
        for r in conn.execute(
            "SELECT external_id FROM artifact WHERE kind='review_comment'"
        ).fetchall()
    }
    assert ids == {"101"}


def _pr_url_calls(gh: FakeGH) -> dict[str, int]:
    """Count paginate calls by endpoint suffix for `bulk_*` assertions."""
    counts: dict[str, int] = {}
    for path, _ in gh.calls:
        counts[path] = counts.get(path, 0) + 1
    return counts


def test_org_bulk_endpoints_called_once_per_repo(conn, tmp_path: Path):
    """The two bulk comment endpoints fire exactly once per repo,
    regardless of PR count — the whole point of this refactor."""
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    prs = [_pr(n, "2024-03-01T00:00:00Z") for n in range(1, 6)]
    review_comments = {n: [_rc(n * 100, "alice", "x", pr=n)] for n in range(1, 6)}
    issue_comments = {n: [_ic(n * 1000, "alice", "y", pr=n)] for n in range(1, 6)}
    reviews = {n: [] for n in range(1, 6)}
    gh = FakeGH(
        repos=_with_bulk(
            {
                "org/r": {
                    "prs": prs,
                    "review_comments": review_comments,
                    "reviews": reviews,
                    "issue_comments": issue_comments,
                },
            }
        ),
    )
    ingest_reviews_org(
        conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg(), target_id=1
    )
    counts = _pr_url_calls(gh)
    assert counts.get("/repos/org/r/pulls/comments", 0) == 1
    assert counts.get("/repos/org/r/issues/comments", 0) == 1


def test_org_bulk_skips_reviews_fetch_for_quiet_prs(conn, tmp_path: Path):
    """A PR with no new comments since the cursor and updated_at <= cursor
    must NOT trigger /pulls/{n}/reviews — that's where the savings live."""
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-04-01T00:00:00Z",
        size_kb=10,
    )
    q.set_repo_cursor(conn, target_id=1, full_name="org/r", reviews_at="2024-02-15T00:00:00Z")
    # PR 7 is newer than the cursor but has no comments at all in the
    # bulk results → reviews must NOT be fetched.
    quiet_pr = _pr(7, "2024-03-01T00:00:00Z")
    # PR 8 has a fresh review comment → reviews MUST be fetched.
    noisy_pr = _pr(8, "2024-03-02T00:00:00Z")
    gh = FakeGH(
        repos=_with_bulk(
            {
                "org/r": {
                    "prs": [noisy_pr, quiet_pr],
                    "review_comments": {8: [_rc(801, "alice", "look", pr=8)]},
                    "reviews": {7: [], 8: []},
                    "issue_comments": {},
                },
            }
        ),
    )
    ingest_reviews_org(
        conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg(), target_id=1
    )
    review_paths = [p for p, _ in gh.calls if p.endswith("/reviews")]
    assert review_paths == ["/repos/org/r/pulls/8/reviews"]


def test_org_bulk_preserves_reviewer_decisions_on_quiet_pr(conn, tmp_path: Path):
    """When the bulk path skips /pulls/{n}/reviews for a PR, the previously
    stored `reviewer_decisions` must survive — otherwise we'd clobber
    history every sync."""
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-04-01T00:00:00Z",
        size_kb=10,
    )
    # First sync: PR 9 has a review from "carol" approving. After this
    # the artifact's meta.reviewer_decisions should record carol/approved.
    first = FakeGH(
        repos=_with_bulk(
            {
                "org/r": {
                    "prs": [_pr(9, "2024-02-10T00:00:00Z")],
                    "review_comments": {9: [_rc(901, "alice", "lgtm", pr=9)]},
                    "reviews": {
                        9: [
                            {
                                "user": {"login": "carol"},
                                "state": "APPROVED",
                                "submitted_at": "2024-02-10T01:00:00Z",
                            }
                        ]
                    },
                    "issue_comments": {},
                },
            }
        ),
    )
    ingest_reviews_org(
        conn=conn, gh=first, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg(), target_id=1
    )
    pr_meta_before = q.get_artifact_meta(conn, target_id=1, kind="pr", external_id="org/r#9")
    assert pr_meta_before is not None
    decisions_before = pr_meta_before.get("reviewer_decisions") or []
    assert any(d.get("login") == "carol" and d.get("state") == "approved" for d in decisions_before)

    # Second sync: PR 9 has nothing new since the (now-advanced) cursor.
    # Bulk comments are empty, no comments arrived, and the PR's
    # updated_at is older than the cursor → reviews skip → decisions
    # must survive untouched.
    second = FakeGH(
        repos=_with_bulk(
            {
                "org/r": {
                    "prs": [_pr(9, "2024-02-10T00:00:00Z")],
                    "review_comments": {},
                    "reviews": {9: []},
                    "issue_comments": {},
                },
            }
        ),
    )
    ingest_reviews_org(
        conn=conn, gh=second, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg(), target_id=1
    )
    pr_meta_after = q.get_artifact_meta(conn, target_id=1, kind="pr", external_id="org/r#9")
    assert pr_meta_after is not None
    assert pr_meta_after.get("reviewer_decisions") == decisions_before
    # And /pulls/9/reviews must NOT have been called in the second sync.
    assert all(p != "/repos/org/r/pulls/9/reviews" for p, _ in second.calls)


def test_org_first_ever_sync_ingests_all_comments(conn, tmp_path: Path):
    """No cursor → all comments are pulled through the bulk endpoints
    (sanity: the no-cursor path must not drop everything)."""
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-04-01T00:00:00Z",
        size_kb=10,
    )
    gh = FakeGH(
        repos=_with_bulk(
            {
                "org/r": {
                    "prs": [_pr(1, "2024-03-01T00:00:00Z")],
                    "review_comments": {
                        1: [_rc(101, "alice", "a", pr=1), _rc(102, "bob", "b", pr=1)]
                    },
                    "reviews": {1: []},
                    "issue_comments": {1: [_ic(201, "carol", "c", pr=1)]},
                },
            }
        ),
    )
    stats = ingest_reviews_org(
        conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg(), target_id=1
    )
    assert stats.review_comments == 2
    assert stats.issue_comments == 1


def test_org_reviews_advances_cursor(conn, tmp_path: Path):
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    gh = FakeGH(
        repos=_with_bulk(
            {"org/r": {"prs": [], "review_comments": {}, "reviews": {}, "issue_comments": {}}}
        ),
    )
    ingest_reviews_org(
        conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=IngestCfg(), target_id=1
    )
    assert q.get_repo(conn, target_id=1, full_name="org/r")["last_reviews_at"] is not None
