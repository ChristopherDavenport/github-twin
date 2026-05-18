"""`gt init` auto-swaps to upstream when the resolved repo is a fork.

The shape: when `discover_repo` reports the resolved repo as a fork with
a parent, the CLI silently re-runs discovery against the parent and
persists the upstream as the target. `--keep-fork` opts out and keeps
the fork. Non-fork repos are unchanged.

Verified end-to-end through Typer's `CliRunner` — both the explicit
`--kind repo` and the `--kind auto` (`.git`-detected) branches exercise
the same swap helper.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from github_twin.cli import app


class _FakeGH:
    """Stand-in for `GitHubClient` in `gt init`'s `with GitHubClient() as gh`.

    Routes `/repos/{owner}/{name}` by-path against a dict of canned responses
    so a single CliRunner.invoke can serve both the initial fork lookup AND
    the follow-up upstream lookup performed by `_swap_fork_to_upstream`.
    """

    def __init__(self, responses: dict[str, dict[str, Any]]):
        self._responses = responses
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_json(self, path: str, *, params: dict[str, Any] | None = None):
        self.calls.append(path)
        return self._responses[path]


def _fork_payload(
    full_name: str = "me/fork",
    parent_full_name: str = "up/x",
    repo_id: int = 555,
    parent_id: int = 999,
) -> dict[str, Any]:
    return {
        "id": repo_id,
        "full_name": full_name,
        "default_branch": "main",
        "pushed_at": "2024-01-01T00:00:00Z",
        "archived": False,
        "fork": True,
        "size": 1,
        "parent": {"id": parent_id, "full_name": parent_full_name},
    }


def _plain_payload(full_name: str = "up/x", repo_id: int = 999) -> dict[str, Any]:
    return {
        "id": repo_id,
        "full_name": full_name,
        "default_branch": "main",
        "pushed_at": "2024-02-01T00:00:00Z",
        "archived": False,
        "fork": False,
        "size": 42,
    }


def _read_rows(db_path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, int]]]:
    conn = sqlite3.connect(db_path)
    try:
        targets = conn.execute("SELECT kind, name FROM target ORDER BY id").fetchall()
        repos = conn.execute(
            "SELECT full_name, fork FROM repo ORDER BY target_id, full_name"
        ).fetchall()
    finally:
        conn.close()
    return targets, repos


def _setup_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(data_dir))
    return data_dir


# ---------- explicit --kind repo ----------


def test_init_repo_fork_silently_swaps_to_upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = _setup_data_dir(monkeypatch, tmp_path)
    fake = _FakeGH(
        {
            "/repos/me/fork": _fork_payload("me/fork", parent_full_name="up/x"),
            "/repos/up/x": _plain_payload("up/x", repo_id=999),
        }
    )
    monkeypatch.setattr("github_twin.cli.GitHubClient", lambda: fake)

    result = CliRunner().invoke(app, ["init", "--kind", "repo", "--repo", "me/fork"])
    assert result.exit_code == 0, result.output
    # Both lookups happened, in order: fork first, then upstream after swap.
    assert fake.calls == ["/repos/me/fork", "/repos/up/x"]

    targets, repos = _read_rows(data_dir / "db.sqlite")
    # Target landed as the upstream, NOT the fork.
    assert targets == [("repo", "up/x")]
    # The repo row mirrors the upstream and is NOT marked as a fork.
    assert repos == [("up/x", 0)]
    # User-facing notice that the swap happened.
    assert "is a fork of up/x" in result.output


def test_init_repo_fork_with_keep_fork_preserves_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_dir = _setup_data_dir(monkeypatch, tmp_path)
    fake = _FakeGH({"/repos/me/fork": _fork_payload("me/fork", parent_full_name="up/x")})
    monkeypatch.setattr("github_twin.cli.GitHubClient", lambda: fake)

    result = CliRunner().invoke(app, ["init", "--kind", "repo", "--repo", "me/fork", "--keep-fork"])
    assert result.exit_code == 0, result.output
    # Only the fork was fetched — no follow-up upstream call.
    assert fake.calls == ["/repos/me/fork"]

    targets, repos = _read_rows(data_dir / "db.sqlite")
    assert targets == [("repo", "me/fork")]
    assert repos == [("me/fork", 1)]
    # No swap notice should appear when --keep-fork is set.
    assert "is a fork of" not in result.output


def test_init_repo_non_fork_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = _setup_data_dir(monkeypatch, tmp_path)
    fake = _FakeGH({"/repos/up/x": _plain_payload("up/x", repo_id=999)})
    monkeypatch.setattr("github_twin.cli.GitHubClient", lambda: fake)

    result = CliRunner().invoke(app, ["init", "--kind", "repo", "--repo", "up/x"])
    assert result.exit_code == 0, result.output
    assert fake.calls == ["/repos/up/x"]

    targets, repos = _read_rows(data_dir / "db.sqlite")
    assert targets == [("repo", "up/x")]
    assert repos == [("up/x", 0)]
    assert "is a fork of" not in result.output


# ---------- --kind auto (.git-detected path) ----------


def _seed_git_origin(cwd: Path, origin_url: str) -> None:
    (cwd / ".git").mkdir()
    (cwd / ".git" / "config").write_text(f'[remote "origin"]\n    url = {origin_url}\n')


def test_init_auto_inside_fork_clone_swaps_to_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_dir = _setup_data_dir(monkeypatch, tmp_path)
    _seed_git_origin(tmp_path, "https://github.com/me/fork.git")
    fake = _FakeGH(
        {
            "/repos/me/fork": _fork_payload("me/fork", parent_full_name="up/x"),
            "/repos/up/x": _plain_payload("up/x", repo_id=999),
        }
    )
    monkeypatch.setattr("github_twin.cli.GitHubClient", lambda: fake)

    result = CliRunner().invoke(app, ["init"])  # default --kind=auto
    assert result.exit_code == 0, result.output
    assert fake.calls == ["/repos/me/fork", "/repos/up/x"]

    targets, repos = _read_rows(data_dir / "db.sqlite")
    assert targets == [("repo", "up/x")]
    assert repos == [("up/x", 0)]
    assert "is a fork of up/x" in result.output


def test_init_auto_inside_fork_clone_keep_fork_preserves_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_dir = _setup_data_dir(monkeypatch, tmp_path)
    _seed_git_origin(tmp_path, "https://github.com/me/fork.git")
    fake = _FakeGH({"/repos/me/fork": _fork_payload("me/fork", parent_full_name="up/x")})
    monkeypatch.setattr("github_twin.cli.GitHubClient", lambda: fake)

    result = CliRunner().invoke(app, ["init", "--keep-fork"])
    assert result.exit_code == 0, result.output
    assert fake.calls == ["/repos/me/fork"]

    targets, _ = _read_rows(data_dir / "db.sqlite")
    assert targets == [("repo", "me/fork")]
    assert "is a fork of" not in result.output
