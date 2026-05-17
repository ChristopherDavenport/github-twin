"""Tests for `ingest_commits_org` (per-repo commits walk) on the legacy API path.

These pin `use_local_git_for_commits=False` to exercise the GitHub-API
fallback. The default (git-local) path is covered in
`test_ingest_commits_org_local.py` against a real fixture repo.

The GitHubClient is faked so the tests are hermetic. We assert:
- author_login is captured from `item.author.login`
- per-repo `last_commits_at` cursor advances after the walk
- artifacts are keyed on (kind='commit', external_id=sha)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github_twin.config import IngestCfg
from github_twin.ingest.cache import RawCache
from github_twin.ingest.commits import ingest_commits_org
from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    seed_target(db)
    yield db
    db.close()


PY_DIFF = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,5 @@
 import os
+def f(): return 1
+def g(): return 2
+def h(): return 3
"""


class FakeGH:
    """Minimal stand-in for GitHubClient. Routes paginate by path prefix
    and serves diffs from a {sha: text} map for `get_text`."""

    def __init__(
        self,
        commits_by_repo: dict[str, list[dict[str, Any]]],
        diffs: dict[str, str],
    ):
        self.commits_by_repo = commits_by_repo
        self.diffs = diffs

    def paginate(self, path: str, *, params: dict | None = None):
        # `/repos/<owner>/<name>/commits`
        if path.startswith("/repos/") and path.endswith("/commits"):
            full_name = path.removeprefix("/repos/").removesuffix("/commits")
            yield from self.commits_by_repo.get(full_name, [])
            return
        raise AssertionError(f"unexpected paginate path: {path}")

    def get_text(self, path: str, *, accept: str) -> str:
        # `/repos/<owner>/<name>/commits/<sha>`
        assert accept.endswith("diff"), accept
        sha = path.rsplit("/", 1)[-1]
        return self.diffs[sha]


def _commit(sha: str, login: str, email: str, date: str) -> dict:
    return {
        "sha": sha,
        "html_url": f"https://gh/example/{sha}",
        "commit": {
            "author": {"name": "X", "email": email, "date": date},
            "message": "first line\n\nbody describing change with enough chars",
        },
        "author": {"login": login},
    }


def test_ingest_commits_org_captures_author_login(conn, tmp_path: Path):
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    gh = FakeGH(
        commits_by_repo={
            "org/r": [
                _commit("aaa1", "alice", "a@x.com", "2024-02-01T00:00:00Z"),
                _commit("bbb2", "bob", "b@x.com", "2024-02-02T00:00:00Z"),
            ],
        },
        diffs={"aaa1": PY_DIFF, "bbb2": PY_DIFF},
    )
    cache = RawCache(tmp_path / "raw")
    stats = ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=cache,
        cfg=IngestCfg(use_local_git_for_commits=False),
        target_id=1,
    )

    assert stats.fetched == 2
    rows = conn.execute(
        "SELECT external_id, repo, author_login, author_email FROM artifact "
        "WHERE kind='commit' ORDER BY external_id"
    ).fetchall()
    assert [r["author_login"] for r in rows] == ["alice", "bob"]
    assert all(r["repo"] == "org/r" for r in rows)


def test_ingest_commits_org_advances_per_repo_cursor(conn, tmp_path: Path):
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/a",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/b",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    gh = FakeGH(
        commits_by_repo={
            "org/a": [_commit("a1", "alice", "a@x.com", "2024-02-01T00:00:00Z")],
            "org/b": [],
        },
        diffs={"a1": PY_DIFF},
    )
    cache = RawCache(tmp_path / "raw")
    ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=cache,
        cfg=IngestCfg(use_local_git_for_commits=False),
        target_id=1,
    )

    a = q.get_repo(conn, target_id=1, full_name="org/a")
    b = q.get_repo(conn, target_id=1, full_name="org/b")
    # Both repos advance their cursor — even the empty one, since we
    # successfully observed "nothing new" through that point in time.
    assert a["last_commits_at"] is not None
    assert b["last_commits_at"] is not None


def test_ingest_commits_org_is_idempotent(conn, tmp_path: Path):
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    gh = FakeGH(
        commits_by_repo={
            "org/r": [_commit("aaa1", "alice", "a@x.com", "2024-02-01T00:00:00Z")],
        },
        diffs={"aaa1": PY_DIFF},
    )
    cache = RawCache(tmp_path / "raw")
    ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=cache,
        cfg=IngestCfg(use_local_git_for_commits=False),
        target_id=1,
    )
    n1 = conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    c1 = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=cache,
        cfg=IngestCfg(use_local_git_for_commits=False),
        target_id=1,
    )
    n2 = conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    c2 = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    assert n1 == n2
    assert c1 == c2
