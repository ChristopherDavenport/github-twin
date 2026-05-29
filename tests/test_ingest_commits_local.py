"""Tests for the git-local commits ingest path (default).

These build a tiny real git repo via the `tmp_git_repo` fixture, then point
`commits_clone` at it (via monkeypatch) so the ingest functions walk a real
working tree without cloning from GitHub.

Covers both modes:
- user mode (`ingest_commits`): filters by configured emails, no `author_login`.
- org mode (`ingest_commits_org`): walks everything, resolves `author_login`
  through the cached email→login map.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from github_twin.config import IngestCfg
from github_twin.ingest.cache import RawCache
from github_twin.ingest.clone import ClonedRepo
from github_twin.ingest.commits import ingest_commits, ingest_commits_org
from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    seed_target(db)
    yield db
    db.close()


def _patch_commits_clone(monkeypatch, repo_full: str, fixture_path: Path, head_sha: str):
    """Make `commits_clone(repo_full, ...)` yield the fixture repo."""
    from github_twin.ingest import commits as commits_mod

    @contextmanager
    def fake(full_name, *, cache_dir, token=None, shallow_since=None):
        yield ClonedRepo(
            full_name=full_name,
            path=fixture_path,
            head_sha=head_sha,
            from_cache=False,
        )

    monkeypatch.setattr(commits_mod, "commits_clone", fake)


class FakeGH:
    """Minimal GitHubClient stand-in.

    `paginate(/search/commits, ...)` is used for user-mode repo discovery; it
    returns whatever we hand in. `gh` is also passed to the identity resolver
    in org mode — there `/search/commits` should not be called when the email
    is a noreply (resolved locally) or already cached. `get_json(/repos/...)`
    backs the org-mode fast-skip pre-check; default returns no pushed_at so
    repos always walk (preserving pre-fast-skip test behaviour).
    """

    token = "fake-token"

    def __init__(
        self,
        repo_search_results=None,
        login_search_results=None,
        repo_info: dict[str, dict] | None = None,
    ):
        self.repo_search_results = repo_search_results or []
        self.login_search_results = login_search_results or {}
        self.repo_info = repo_info or {}
        self.calls: list[tuple[str, dict]] = []

    def paginate(self, path: str, *, params: dict | None = None):
        self.calls.append((path, params or {}))
        if path == "/search/commits":
            qstr = (params or {}).get("q", "")
            if "author-email:" in qstr:
                # email→login lookup
                for email, login in self.login_search_results.items():
                    if email in qstr:
                        yield {"author": {"login": login}}
                        return
                return
            yield from self.repo_search_results

    def get_json(self, path: str, *, params: dict | None = None):
        self.calls.append((path, params or {}))
        if path.startswith("/repos/"):
            repo_full = path[len("/repos/") :]
            return self.repo_info.get(repo_full, {})
        return {}

    def get_json_cached(self, path: str, *, params: dict | None = None):
        # `_fetch_repo_pushed_at` uses the conditional variant; this fake
        # doesn't model 304s, so route to the unconditional path.
        return self.get_json(path, params=params)


# ---------- user mode ----------


def test_user_mode_walks_only_matching_emails(conn, monkeypatch, tmp_git_repo):
    # Two commits: one by the target user, one by someone else.
    tmp_git_repo.make_commit(
        path="a.py",
        content="def a():\n    return 1\n",
        message="add a",
        author_email="me@example.com",
        author_name="Me",
    )
    other = tmp_git_repo.make_commit(
        path="b.py",
        content="def b():\n    return 2\n",
        message="add b",
        author_email="other@example.com",
        author_name="Other",
    )
    head = tmp_git_repo.head_sha

    _patch_commits_clone(monkeypatch, "me/proj", tmp_git_repo.path, head)
    gh = FakeGH(
        repo_search_results=[{"repository": {"full_name": "me/proj"}}],
    )

    stats = ingest_commits(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        username="me",
        emails=["me@example.com"],
        cfg=IngestCfg(since="2020-01-01"),
        target_id=1,
    )

    assert stats.fetched == 1
    rows = conn.execute(
        "SELECT external_id, author_email, author_login FROM artifact WHERE kind='commit'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["author_email"] == "me@example.com"
    # user mode never populates author_login
    assert rows[0]["author_login"] is None
    # The other-user commit is not in the DB.
    assert other not in {r["external_id"] for r in rows}


def test_user_mode_persists_walked_sha(conn, monkeypatch, tmp_git_repo):
    tmp_git_repo.make_commit(
        path="a.py",
        content="x=1\n",
        message="first",
        author_email="me@example.com",
    )
    head = tmp_git_repo.head_sha
    _patch_commits_clone(monkeypatch, "me/proj", tmp_git_repo.path, head)
    gh = FakeGH(repo_search_results=[{"repository": {"full_name": "me/proj"}}])

    ingest_commits(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        username="me",
        emails=["me@example.com"],
        cfg=IngestCfg(since="2020-01-01"),
        target_id=1,
    )

    row = q.get_repo(conn, target_id=1, full_name="me/proj")
    assert row is not None
    assert row["last_commits_walked_sha"] == head


def test_user_mode_incremental_walk_skips_already_seen(conn, monkeypatch, tmp_git_repo):
    # First pass: one commit dated 2024 (before the cursor stamp).
    tmp_git_repo.make_commit(
        path="a.py",
        content="x=1\n",
        message="first",
        author_email="me@example.com",
        date="2024-01-02T00:00:00+00:00",
    )
    _patch_commits_clone(monkeypatch, "me/proj", tmp_git_repo.path, tmp_git_repo.head_sha)
    gh = FakeGH(repo_search_results=[{"repository": {"full_name": "me/proj"}}])
    cfg = IngestCfg(since="2020-01-01")
    ingest_commits(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        username="me",
        emails=["me@example.com"],
        cfg=cfg,
        target_id=1,
    )
    first_count = conn.execute("SELECT COUNT(*) FROM artifact WHERE kind='commit'").fetchone()[0]
    assert first_count == 1

    # Second pass: a new commit dated strictly AFTER the cursor that the
    # first sync stamped (`_now_iso()` is the test-execution wall clock).
    # The org-mode-style worker walks `git log --since=<cursor>`, so the
    # new commit must postdate the cursor to be picked up — mirroring the
    # real-world flow where incremental commits are always "now"-dated.
    tmp_git_repo.make_commit(
        path="b.py",
        content="y=2\n",
        message="second",
        author_email="me@example.com",
        date="2099-01-01T00:00:00+00:00",
    )
    _patch_commits_clone(monkeypatch, "me/proj", tmp_git_repo.path, tmp_git_repo.head_sha)
    ingest_commits(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        username="me",
        emails=["me@example.com"],
        cfg=cfg,
        target_id=1,
    )
    assert conn.execute("SELECT COUNT(*) FROM artifact WHERE kind='commit'").fetchone()[0] == 2


def test_user_mode_fast_skips_unchanged_repos(conn, monkeypatch, tmp_git_repo):
    """Second sync skips the clone entirely when `/repos/{r}` reports a
    `pushed_at` no newer than the stored `last_commits_at` cursor.

    Mirrors `_needs_walk` behavior — port from org-mode."""
    tmp_git_repo.make_commit(
        path="a.py",
        content="x=1\n",
        message="first",
        author_email="me@example.com",
        date="2024-01-02T00:00:00+00:00",
    )
    _patch_commits_clone(monkeypatch, "me/proj", tmp_git_repo.path, tmp_git_repo.head_sha)
    # `pushed_at` is in the past relative to the cursor we'll stamp ("now").
    gh = FakeGH(
        repo_search_results=[{"repository": {"full_name": "me/proj"}}],
        repo_info={"me/proj": {"pushed_at": "2020-01-01T00:00:00Z"}},
    )
    cfg = IngestCfg(since="2020-01-01")
    ingest_commits(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        username="me",
        emails=["me@example.com"],
        cfg=cfg,
        target_id=1,
    )
    first_count = conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    assert first_count == 1

    # Second pass: monkeypatch `commits_clone` to explode so any attempted
    # clone would fail the test. With fast-skip wired up, no clone happens.
    from github_twin.ingest import commits as commits_mod

    def boom(*_a, **_kw):
        raise AssertionError("fast-skip failed — repo was cloned")

    monkeypatch.setattr(commits_mod, "commits_clone", boom)
    ingest_commits(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        username="me",
        emails=["me@example.com"],
        cfg=cfg,
        target_id=1,
    )
    assert conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0] == first_count


# ---------- org mode ----------


def test_org_mode_resolves_author_login_via_cache(conn, monkeypatch, tmp_git_repo):
    # Pre-seed cache for both authors so no API call is needed. The fixture's
    # initial commit is authored by test@example.com; we add a second commit
    # authored by alice and verify alice's commit gets `author_login='alice'`.
    q.upsert_email_login(conn, email="alice@example.com", login="alice", source="manual")
    q.upsert_email_login(conn, email="test@example.com", login=None, source="manual")

    alice_sha = tmp_git_repo.make_commit(
        path="a.py",
        content="def a(): return 1\n",
        message="add a",
        author_email="alice@example.com",
    )
    head = tmp_git_repo.head_sha

    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    _patch_commits_clone(monkeypatch, "org/r", tmp_git_repo.path, head)
    gh = FakeGH()  # no API responses needed

    ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        cfg=IngestCfg(since="2020-01-01"),
        target_id=1,
    )

    row = conn.execute(
        "SELECT author_email, author_login FROM artifact WHERE external_id=?",
        (alice_sha,),
    ).fetchone()
    assert row["author_email"] == "alice@example.com"
    assert row["author_login"] == "alice"
    # Resolver was satisfied by the cache, so no /search/commits hits.
    assert all(p != "/search/commits" for p, _ in gh.calls)


def test_org_mode_noreply_email_resolves_without_api(conn, monkeypatch, tmp_git_repo):
    # Pre-seed the fixture's initial-commit author so the resolver doesn't
    # try to hit /search/commits for it during the walk.
    q.upsert_email_login(conn, email="test@example.com", login=None, source="manual")

    bob_sha = tmp_git_repo.make_commit(
        path="a.py",
        content="def a(): return 1\n",
        message="add a",
        author_email="42+bob@users.noreply.github.com",
    )
    head = tmp_git_repo.head_sha

    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    _patch_commits_clone(monkeypatch, "org/r", tmp_git_repo.path, head)
    gh = FakeGH()

    ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        cfg=IngestCfg(since="2020-01-01"),
        target_id=1,
    )
    row = conn.execute(
        "SELECT author_login FROM artifact WHERE external_id=?",
        (bob_sha,),
    ).fetchone()
    assert row["author_login"] == "bob"
    # No API call needed — noreply emails resolve locally.
    assert all(p != "/search/commits" for p, _ in gh.calls)


def test_org_mode_persists_walked_sha(conn, monkeypatch, tmp_git_repo):
    tmp_git_repo.make_commit(
        path="a.py",
        content="x=1\n",
        message="first",
        author_email="alice@example.com",
    )
    head = tmp_git_repo.head_sha
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    _patch_commits_clone(monkeypatch, "org/r", tmp_git_repo.path, head)
    gh = FakeGH(login_search_results={"alice@example.com": "alice"})

    ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        cfg=IngestCfg(since="2020-01-01"),
        target_id=1,
    )
    repo_row = q.get_repo(conn, target_id=1, full_name="org/r")
    assert repo_row["last_commits_walked_sha"] == head


def test_org_mode_is_idempotent(conn, monkeypatch, tmp_git_repo):
    tmp_git_repo.make_commit(
        path="a.py",
        content="x=1\n",
        message="first",
        author_email="alice@example.com",
    )
    head = tmp_git_repo.head_sha
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    q.upsert_email_login(conn, email="alice@example.com", login="alice", source="manual")
    _patch_commits_clone(monkeypatch, "org/r", tmp_git_repo.path, head)
    gh = FakeGH()
    cfg = IngestCfg(since="2020-01-01")

    ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        cfg=cfg,
        target_id=1,
    )
    n1 = conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    c1 = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    # Re-running with the same HEAD walks nothing new (cursor matches).
    ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=RawCache(tmp_git_repo.path / "raw"),
        cfg=cfg,
        target_id=1,
    )
    n2 = conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    c2 = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    assert n1 == n2
    assert c1 == c2
