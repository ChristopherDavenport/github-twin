"""Tests for repo-mode target discovery and the pipeline + CLI hooks that
let `gt init` operate against a single repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github_twin.config import Config
from github_twin.pipeline import run_ingest
from github_twin.store import queries as q
from github_twin.store.db import open_db, transaction
from github_twin.target import (
    Target,
    _find_git_root,
    _parse_origin_owner_name,
    discover_repo,
    maybe_discover_repo,
    save_target,
)

# ---------- origin URL parsing ----------


def test_parse_origin_owner_name_https():
    cfg = """
[core]
    repositoryformatversion = 0
[remote "origin"]
    url = https://github.com/foo/bar.git
    fetch = +refs/heads/*:refs/remotes/origin/*
"""
    assert _parse_origin_owner_name(cfg) == ("foo", "bar")


def test_parse_origin_owner_name_https_no_dotgit():
    cfg = '[remote "origin"]\n    url = https://github.com/foo/bar\n'
    assert _parse_origin_owner_name(cfg) == ("foo", "bar")


def test_parse_origin_owner_name_ssh():
    cfg = '[remote "origin"]\n    url = git@github.com:foo/bar.git\n'
    assert _parse_origin_owner_name(cfg) == ("foo", "bar")


def test_parse_origin_owner_name_rejects_non_github():
    cfg = '[remote "origin"]\n    url = https://gitlab.com/foo/bar.git\n'
    assert _parse_origin_owner_name(cfg) is None


def test_parse_origin_owner_name_no_origin_section():
    cfg = '[remote "upstream"]\n    url = https://github.com/foo/bar.git\n'
    assert _parse_origin_owner_name(cfg) is None


# ---------- git root walk-up ----------


def test_find_git_root_walks_up(tmp_path: Path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    nested = root / "src" / "a" / "b"
    nested.mkdir(parents=True)
    found = _find_git_root(nested)
    # resolve() canonicalizes both sides so symlinked tmpdirs match.
    assert found is not None and found.resolve() == root.resolve()


def test_find_git_root_returns_none_when_no_git(tmp_path: Path):
    assert _find_git_root(tmp_path) is None


# ---------- discover_repo ----------


class FakeGH:
    """Stubs GitHubClient.get_json with a fixed response or a per-path map."""

    def __init__(
        self, response: dict[str, Any] | dict[str, dict[str, Any]], *, by_path: bool = False
    ):
        self._response = response
        self._by_path = by_path
        self.calls: list[str] = []

    def get_json(self, path: str, *, params: dict[str, Any] | None = None):
        self.calls.append(path)
        if self._by_path:
            return self._response[path]  # type: ignore[index]
        return self._response


def _fake_repo_response(full_name: str = "foo/bar", repo_id: int = 12345) -> dict:
    return {
        "id": repo_id,
        "full_name": full_name,
        "default_branch": "main",
        "pushed_at": "2024-01-01T00:00:00Z",
        "archived": False,
        "fork": False,
        "size": 42,
    }


def _fake_fork_response(
    full_name: str = "me/fork",
    repo_id: int = 555,
    parent_full_name: str = "up/x",
    parent_id: int = 999,
) -> dict:
    """Mirror what GitHub returns for a fork: fork=true + parent block."""
    return {
        "id": repo_id,
        "full_name": full_name,
        "default_branch": "main",
        "pushed_at": "2024-01-01T00:00:00Z",
        "archived": False,
        "fork": True,
        "size": 1,
        "parent": {
            "id": parent_id,
            "full_name": parent_full_name,
        },
    }


def test_discover_repo_with_explicit_arg():
    gh = FakeGH(_fake_repo_response())
    target, meta, parent = discover_repo(gh, repo="foo/bar")
    assert target.kind == "repo"
    assert target.is_repo is True
    assert target.name == "foo/bar"
    assert target.external_id == 12345
    assert target.emails == []
    assert gh.calls == ["/repos/foo/bar"]
    assert meta["full_name"] == "foo/bar"
    assert meta["default_branch"] == "main"
    assert meta["size_kb"] == 42
    assert parent is None


def test_discover_repo_from_git_dir(tmp_path: Path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        '[remote "origin"]\n    url = https://github.com/typelevel/cats-effect.git\n'
    )
    gh = FakeGH(_fake_repo_response("typelevel/cats-effect", repo_id=987))
    target, meta, parent = discover_repo(gh, start_path=tmp_path)
    assert target.name == "typelevel/cats-effect"
    assert target.external_id == 987
    assert gh.calls == ["/repos/typelevel/cats-effect"]
    assert meta["full_name"] == "typelevel/cats-effect"
    assert parent is None


def test_discover_repo_reports_parent_for_forks():
    """`fork=true` with a parent block surfaces the upstream full_name."""
    gh = FakeGH(_fake_fork_response("me/fork", parent_full_name="up/x"))
    target, meta, parent = discover_repo(gh, repo="me/fork")
    assert target.name == "me/fork"
    assert meta["fork"] is True
    assert parent == "up/x"


def test_discover_repo_returns_none_parent_when_fork_lacks_parent_block():
    """Defensive: malformed API response (fork=true, no parent) → parent None."""
    response = _fake_fork_response()
    del response["parent"]
    gh = FakeGH(response)
    target, meta, parent = discover_repo(gh, repo="me/fork")
    assert meta["fork"] is True
    assert parent is None


def test_discover_repo_rejects_bad_owner_name():
    gh = FakeGH(_fake_repo_response())
    with pytest.raises(ValueError, match="owner/name"):
        discover_repo(gh, repo="notvalid")


def test_discover_repo_errors_when_no_git_and_no_arg(tmp_path: Path):
    gh = FakeGH(_fake_repo_response())
    with pytest.raises(ValueError, match="No .git directory"):
        discover_repo(gh, start_path=tmp_path)


def test_discover_repo_errors_on_non_github_origin(tmp_path: Path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text('[remote "origin"]\n    url = https://gitlab.com/foo/bar.git\n')
    gh = FakeGH(_fake_repo_response())
    with pytest.raises(ValueError, match="not a github.com"):
        discover_repo(gh, start_path=tmp_path)


def test_maybe_discover_repo_returns_none_on_failure(tmp_path: Path):
    """Auto-detect must swallow ValueError so `gt init` can fall back to user mode."""
    gh = FakeGH(_fake_repo_response())
    assert maybe_discover_repo(gh, start_path=tmp_path) is None


def test_maybe_discover_repo_returns_target_on_success(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text(
        '[remote "origin"]\n    url = git@github.com:me/x.git\n'
    )
    gh = FakeGH(_fake_repo_response("me/x"))
    result = maybe_discover_repo(gh, start_path=tmp_path)
    assert result is not None
    target, meta, parent = result
    assert target.is_repo and target.name == "me/x"
    assert parent is None


def test_maybe_discover_repo_surfaces_parent_for_forks(tmp_path: Path):
    """Auto-detect inside a fork's working tree must propagate the parent."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text(
        '[remote "origin"]\n    url = git@github.com:me/fork.git\n'
    )
    gh = FakeGH(_fake_fork_response("me/fork", parent_full_name="up/x"))
    result = maybe_discover_repo(gh, start_path=tmp_path)
    assert result is not None
    target, meta, parent = result
    assert target.name == "me/fork"
    assert parent == "up/x"


# ---------- pipeline dispatch ----------


@pytest.fixture
def conn_with_repo_target(tmp_path: Path):
    db = open_db(tmp_path / "repo.sqlite", embed_dim=4)
    target = Target(kind="repo", name="me/x", external_id=1, emails=[])
    with transaction(db):
        target = save_target(db, target)
        assert target.id is not None
        q.upsert_repo(
            db,
            target_id=target.id,
            full_name="me/x",
            default_branch="main",
            pushed_at="2024-01-01T00:00:00Z",
        )
    yield db
    db.close()


def test_run_ingest_dispatches_repo_kind_through_org_path(conn_with_repo_target, monkeypatch):
    """Repo mode must reuse the same three ingest functions org mode uses."""
    called: dict[str, int] = {"files": 0, "commits": 0, "reviews": 0}

    def fake_files(*, conn, cfg, target_id, limit=None):
        called["files"] += 1
        return {"repos_walked": 0}

    def fake_commits(
        *, conn, gh, cache, cfg, target_id, limit_per_repo=None, pushed_at_by_repo=None
    ):
        called["commits"] += 1
        return {"new_commits": 0}

    def fake_reviews(
        *,
        conn,
        gh,
        cache,
        cfg,
        target_id,
        limit_prs_per_repo=None,
        pushed_at_by_repo=None,
    ):
        called["reviews"] += 1
        return {"new_review_comments": 0}

    monkeypatch.setattr("github_twin.pipeline.ingest_files", fake_files)
    monkeypatch.setattr("github_twin.pipeline.ingest_commits_org", fake_commits)
    monkeypatch.setattr("github_twin.pipeline.ingest_reviews_org", fake_reviews)
    # Skip the real GitHubClient context manager.
    monkeypatch.setattr(
        "github_twin.pipeline.GitHubClient",
        lambda: _NullClient(),
    )

    cfg = Config()
    run_ingest(cfg, conn_with_repo_target)

    assert called == {"files": 1, "commits": 1, "reviews": 1}


class _NullClient:
    token = "fake-token"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def get_json(self, path: str, *, params: dict | None = None):
        # Backs the pipeline's `/repos/{r}` fast-skip pre-check. Returns an
        # empty info dict so the test's fake_commits / fake_reviews receive
        # a `pushed_at_by_repo={r: None}` and decide what to do internally.
        return {}
