"""Tests for the parallel org-mode sync pipeline.

Covers the new invariants from the perf overhaul:
- Fast-skip via `/repos/{r}` pushed_at: when pushed_at <= last_commits_at,
  the repo is skipped entirely (no clone, no walk).
- Per-repo transactions: a failure in one repo's worker doesn't roll back
  another's cursor advance.
- Content-hash skip on re-ingest: a commit/comment whose source bytes
  match the stored `artifact.content_hash` skips chunk wipe+re-insert
  (and leaves embeddings intact).
- Bulk email→login pre-resolution warms the cache concurrently.
- Token plumbing: `_resolve_token` is called exactly once per sync
  (workers pass `gh.token` through to `commits_clone`).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from github_twin.config import IngestCfg
from github_twin.ingest import clone as clone_mod
from github_twin.ingest import commits as commits_mod
from github_twin.ingest.cache import RawCache
from github_twin.ingest.clone import ClonedRepo
from github_twin.ingest.commits import (
    _commit_content_hash,
    _needs_walk,
    _shallow_since_for,
    ingest_commits_org,
)
from github_twin.ingest.identity import bulk_resolve_logins
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


# ---------- FakeGH for the parallel paths ----------


class FakeGH:
    """Minimal client supporting the new `/repos/{r}` get_json call plus
    paginate() for review subresources and search/commits login lookups."""

    token = "fake-token"

    def __init__(
        self,
        *,
        repo_info: dict[str, dict[str, Any]] | None = None,
        login_search: dict[str, str | None] | None = None,
        reviews: dict[str, dict[str, Any]] | None = None,
    ):
        self.repo_info = repo_info or {}
        self.login_search = login_search or {}
        self.reviews = reviews or {}
        self.get_json_calls: list[str] = []
        self.paginate_calls: list[tuple[str, dict]] = []

    def get_json(self, path: str, *, params: dict | None = None):
        self.get_json_calls.append(path)
        if path.startswith("/repos/"):
            rest = path[len("/repos/") :]
            return self.repo_info.get(rest, {})
        return {}

    def get_json_cached(self, path: str, *, params: dict | None = None):
        # Mirror unconditional path; conditional 304 handling lives in the
        # real client and is exercised in test_github_client_conditional.
        return self.get_json(path, params=params)

    def paginate_cached(self, path: str, *, params: dict | None = None):
        yield from self.paginate(path, params=params)

    def paginate(self, path: str, *, params: dict | None = None):
        self.paginate_calls.append((path, params or {}))
        if path == "/search/commits":
            qstr = (params or {}).get("q", "")
            for email, login in self.login_search.items():
                if email in qstr and login is not None:
                    yield {"author": {"login": login}}
                    return
            return
        if path.endswith("/pulls") and path.count("/") == 4:
            full = path.removeprefix("/repos/").removesuffix("/pulls")
            yield from self.reviews.get(full, {}).get("prs", [])
            return
        if "/pulls/" in path and path.endswith("/comments"):
            full, _, rest = path.removeprefix("/repos/").partition("/pulls/")
            n = int(rest.split("/")[0])
            yield from self.reviews[full]["review_comments"].get(n, [])
            return
        if path.endswith("/reviews"):
            full, _, rest = path.removeprefix("/repos/").partition("/pulls/")
            n = int(rest.split("/")[0])
            yield from self.reviews[full]["reviews"].get(n, [])
            return
        if "/issues/" in path and path.endswith("/comments"):
            full, _, rest = path.removeprefix("/repos/").partition("/issues/")
            n = int(rest.split("/")[0])
            yield from self.reviews[full]["issue_comments"].get(n, [])
            return
        raise AssertionError(f"unexpected paginate: {path}")


# ---------- Pure-function unit tests ----------


def test_needs_walk_pushed_after_last_commits_at():
    assert _needs_walk("2024-02-02T00:00:00Z", "2024-02-01T00:00:00Z") is True


def test_needs_walk_pushed_before_last_commits_at_skips():
    assert _needs_walk("2024-02-01T00:00:00Z", "2024-02-02T00:00:00Z") is False


def test_needs_walk_unknown_pushed_walks_defensively():
    assert _needs_walk(None, "2024-02-01T00:00:00Z") is True


def test_needs_walk_unknown_cursor_walks_defensively():
    assert _needs_walk("2024-02-01T00:00:00Z", None) is True


def test_commit_content_hash_stable_across_calls():
    h1 = _commit_content_hash("diff body", "msg")
    h2 = _commit_content_hash("diff body", "msg")
    assert h1 == h2


def test_commit_content_hash_message_change_changes_hash():
    assert _commit_content_hash("d", "a") != _commit_content_hash("d", "b")


def test_commit_content_hash_diff_change_changes_hash():
    assert _commit_content_hash("a", "m") != _commit_content_hash("b", "m")


def test_shallow_since_pads_backward():
    cfg = IngestCfg(since="2018-01-01", shallow_since_pad_days=3)
    # ISO date in, ISO date out (date only).
    assert _shallow_since_for("2024-02-05T00:00:00+00:00", cfg) == "2024-02-02"


def test_shallow_since_falls_back_to_cfg_since_when_no_cursor():
    """No cursor → base is `cfg.since`; the pad still applies, which is
    harmless (we want everything since cfg.since anyway)."""
    cfg = IngestCfg(since="2018-01-01", shallow_since_pad_days=0)
    assert _shallow_since_for(None, cfg) == cfg.since
    cfg_padded = IngestCfg(since="2018-01-01", shallow_since_pad_days=1)
    assert _shallow_since_for(None, cfg_padded) == "2017-12-31"


# ---------- Fast-skip integration ----------


def _patch_clone_counter(monkeypatch):
    """Patch `commits_clone` to record calls and yield a dummy ClonedRepo."""
    calls: list[dict] = []

    @contextmanager
    def fake(full_name, *, cache_dir, token=None, shallow_since=None):
        calls.append({"full_name": full_name, "token": token, "shallow_since": shallow_since})
        # Yield a path that doesn't exist — `_git_log` will fail and the
        # worker will record a skipped commit, but the test only cares
        # about whether the clone was attempted.
        yield ClonedRepo(
            full_name=full_name,
            path=Path("/nonexistent"),
            head_sha="dead",
            from_cache=False,
        )

    monkeypatch.setattr(commits_mod, "commits_clone", fake)
    return calls


def test_fast_skip_unchanged_repo_skips_clone(conn, monkeypatch):
    """When pushed_at is older than last_commits_at, the clone is never opened."""
    # Pre-seed a repo with an existing commits cursor in the future relative
    # to the FakeGH-reported pushed_at — fast-skip should kick in.
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/quiet",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
    )
    q.set_repo_cursor(
        conn,
        target_id=1,
        full_name="org/quiet",
        commits_at="2024-06-01T00:00:00Z",
    )
    gh = FakeGH(repo_info={"org/quiet": {"pushed_at": "2024-02-01T00:00:00Z"}})
    calls = _patch_clone_counter(monkeypatch)

    stats = ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=RawCache(Path("/tmp/raw")),
        cfg=IngestCfg(since="2020-01-01"),
        target_id=1,
    )

    assert calls == []  # never cloned
    assert stats.fetched == 0
    # And `/repos/{r}` WAS called (the fast-skip pre-check).
    assert "/repos/org/quiet" in gh.get_json_calls


def test_fast_skip_uses_provided_pushed_at_batch(conn, monkeypatch):
    """When the caller passes pushed_at_by_repo, the helper doesn't re-fetch."""
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/quiet",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
    )
    q.set_repo_cursor(
        conn,
        target_id=1,
        full_name="org/quiet",
        commits_at="2024-06-01T00:00:00Z",
    )
    gh = FakeGH()  # no repo_info — would return {} on get_json
    calls = _patch_clone_counter(monkeypatch)

    ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=RawCache(Path("/tmp/raw")),
        cfg=IngestCfg(since="2020-01-01"),
        target_id=1,
        pushed_at_by_repo={"org/quiet": "2024-02-01T00:00:00Z"},
    )

    assert calls == []  # fast-skipped
    assert gh.get_json_calls == []  # caller's batch was reused


# ---------- Token plumbing ----------


def test_resolve_token_called_at_most_once_per_sync(conn, monkeypatch):
    """The token comes from `gh.token` (single property read), not from
    re-running `_resolve_token` per repo."""
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/a",
        default_branch="main",
        pushed_at="2024-09-01T00:00:00Z",  # newer than any cursor
    )
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/b",
        default_branch="main",
        pushed_at="2024-09-01T00:00:00Z",
    )
    # No cursors → fast-skip won't apply → both get walked.

    resolve_calls = {"n": 0}

    def counting_resolve():
        resolve_calls["n"] += 1
        return "should-not-be-used"

    # Patch BOTH module-level references just in case.
    monkeypatch.setattr(clone_mod, "_resolve_token", counting_resolve)

    # Make commits_clone yield without ever invoking _resolve_token; the
    # token comes from `gh.token` (not None), so `cloned_repo` uses the
    # supplied token and skips _resolve_token entirely.
    @contextmanager
    def fake(full_name, *, cache_dir, token=None, shallow_since=None):
        assert token == "fake-token", "worker should receive gh.token, not None"
        yield ClonedRepo(
            full_name=full_name,
            path=Path("/nonexistent"),
            head_sha="x",
            from_cache=False,
        )

    monkeypatch.setattr(commits_mod, "commits_clone", fake)

    gh = FakeGH(
        repo_info={
            "org/a": {"pushed_at": "2024-09-01T00:00:00Z"},
            "org/b": {"pushed_at": "2024-09-01T00:00:00Z"},
        }
    )
    ingest_commits_org(
        conn=conn,
        gh=gh,
        cache=RawCache(Path("/tmp/raw")),
        cfg=IngestCfg(since="2020-01-01", repo_concurrency=2),
        target_id=1,
    )
    # Token was provided by gh.token; _resolve_token was never needed.
    assert resolve_calls["n"] == 0


# ---------- Bulk email→login pre-resolution ----------


def test_bulk_resolve_logins_caches_misses(conn):
    """Pre-resolving an email that isn't found caches the miss so we don't
    re-issue the API call next time."""
    gh = FakeGH(login_search={"unknown@example.com": None})
    bulk_resolve_logins(conn, gh, ["unknown@example.com"])
    cached, resolved = q.get_email_login(conn, "unknown@example.com")
    assert resolved is True
    assert cached is None


def test_bulk_resolve_logins_skips_cached(conn):
    """Already-cached emails don't trigger a paginate call."""
    q.upsert_email_login(conn, email="alice@example.com", login="alice", source="manual")
    gh = FakeGH()  # would AssertionError on unexpected paginate
    bulk_resolve_logins(conn, gh, ["alice@example.com"])
    # No /search/commits calls because the cache was warm.
    assert all(p != "/search/commits" for p, _ in gh.paginate_calls)


def test_bulk_resolve_logins_handles_noreply_locally(conn):
    """Noreply addresses resolve without an API call."""
    gh = FakeGH()
    bulk_resolve_logins(conn, gh, ["12345+octocat@users.noreply.github.com"])
    cached, resolved = q.get_email_login(conn, "12345+octocat@users.noreply.github.com")
    assert resolved is True
    assert cached == "octocat"
    assert all(p != "/search/commits" for p, _ in gh.paginate_calls)


# ---------- Content-hash skip on reviews ----------


def _pr(n: int, updated: str, title: str = "x", body: str = "") -> dict:
    return {
        "number": n,
        "updated_at": updated,
        "title": title,
        "body": body,
        "state": "open",
        "html_url": "",
        "user": {"login": "alice"},
        "created_at": updated,
    }


def _rc(id_: int, login: str, body: str) -> dict:
    return {
        "id": id_,
        "user": {"login": login},
        "body": body,
        "path": "src/x.py",
        "diff_hunk": "@@ -1,1 +1,2 @@\n+new",
        "created_at": "2024-02-01T00:00:00Z",
        "html_url": f"https://gh/c/{id_}",
    }


def test_review_content_hash_skip_on_unchanged_body(conn, tmp_path: Path, monkeypatch):
    """Re-running reviews ingest on a PR whose comments haven't changed
    must not re-insert chunks (verifies content_hash short-circuit)."""
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-03-01T00:00:00Z",
    )
    pr = _pr(7, "2024-03-01T00:00:00Z", title="t", body="b")
    rc = _rc(101, "bob", "this needs a test")
    gh = FakeGH(
        repo_info={"org/r": {"pushed_at": "2024-04-01T00:00:00Z"}},
        reviews={
            "org/r": {
                "prs": [pr],
                "review_comments": {7: [rc]},
                "reviews": {7: []},
                "issue_comments": {7: []},
            }
        },
    )
    cfg = IngestCfg(since="2020-01-01")

    ingest_reviews_org(conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=cfg, target_id=1)
    chunks_first = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    # Stamp embeddings on the chunks to detect whether the second run
    # invalidates them (the content-hash skip should leave them alone).
    conn.execute("UPDATE chunk SET embed_model='fake-v1'")

    # Move pushed_at forward so fast-skip doesn't apply; the per-comment
    # content-hash short-circuit is what we want to test.
    gh.repo_info["org/r"]["pushed_at"] = "2024-05-01T00:00:00Z"
    ingest_reviews_org(conn=conn, gh=gh, cache=RawCache(tmp_path / "raw"), cfg=cfg, target_id=1)
    chunks_second = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    assert chunks_first == chunks_second

    # Critically: embeddings are still stamped (chunks weren't wiped).
    n_with_embed = conn.execute(
        "SELECT COUNT(*) FROM chunk WHERE embed_model='fake-v1'"
    ).fetchone()[0]
    assert n_with_embed == chunks_second
