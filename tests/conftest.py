"""Shared pytest fixtures for github-twin.

`tmp_git_repo` builds a tiny real git repo on disk with controlled author
metadata and a few commits.

`seed_target` writes a target row directly so tests don't need to spin
up the discovery layer. Most tests pin to a single user-mode target with
id=1; multi-target tests can seed additional rows.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest


def seed_target(
    conn: sqlite3.Connection,
    *,
    kind: str = "user",
    name: str = "me",
    external_id: int = 1,
    emails: list[str] | None = None,
) -> int:
    """Insert a target row and return its id. Idempotent on (kind, name).

    Tests that need a writable artifact/repo/sync_cursor call this once
    in setup. Default seeds a user-mode target so legacy single-target
    tests keep working with a `target_id=1` everywhere.
    """
    emails_json = json.dumps(emails) if (emails is not None and kind == "user") else None
    cur = conn.execute(
        "INSERT INTO target (kind, name, external_id, emails_json, discovered_at) "
        "VALUES (?, ?, ?, ?, '2024-01-01T00:00:00+00:00') "
        "ON CONFLICT(kind, name) DO UPDATE SET "
        "external_id=excluded.external_id, emails_json=excluded.emails_json "
        "RETURNING id",
        (kind, name, external_id, emails_json),
    )
    row_id: int = cur.fetchone()[0]
    return row_id


def _git(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    base_env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        # Stable dates so any test relying on author-date is deterministic.
        "GIT_AUTHOR_DATE": "2024-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2024-01-01T00:00:00+00:00",
        # Force an empty initial config so the host's settings can't leak in.
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if env:
        base_env.update(env)
    out = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=base_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({out.returncode}): {out.stderr.strip()}")
    return out.stdout.strip()


@dataclass
class GitRepoFixture:
    path: Path
    shas: list[str]  # commits in chronological order
    head_sha: str

    def make_commit(
        self,
        *,
        path: str,
        content: str,
        message: str,
        author_email: str = "test@example.com",
        author_name: str = "Test",
        date: str = "2024-01-02T00:00:00+00:00",
    ) -> str:
        full = self.path / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        _git(["add", path], cwd=self.path)
        _git(
            ["commit", "-m", message],
            cwd=self.path,
            env={
                "GIT_AUTHOR_NAME": author_name,
                "GIT_AUTHOR_EMAIL": author_email,
                "GIT_COMMITTER_NAME": author_name,
                "GIT_COMMITTER_EMAIL": author_email,
                "GIT_AUTHOR_DATE": date,
                "GIT_COMMITTER_DATE": date,
            },
        )
        sha = _git(["rev-parse", "HEAD"], cwd=self.path)
        self.shas.append(sha)
        self.head_sha = sha
        return sha


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> GitRepoFixture:
    """An on-disk git repo with one initial commit.

    The fixture exposes `make_commit(...)` so tests can append commits with
    chosen authors / dates without re-running `git init` boilerplate.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], cwd=repo)
    # Initial commit so HEAD exists.
    (repo / "README.md").write_text("hello\n")
    _git(["add", "README.md"], cwd=repo)
    _git(["commit", "-m", "init"], cwd=repo)
    head = _git(["rev-parse", "HEAD"], cwd=repo)
    return GitRepoFixture(path=repo, shas=[head], head_sha=head)
