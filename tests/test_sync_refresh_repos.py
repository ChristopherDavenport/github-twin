"""`gt sync` refreshes archived + visibility on every org-mode target.

`enumerate_org_repos` only runs at `gt init` time, so a repo archived
after init keeps `archived=0` in the DB and slips through every ingest
read site (which default to `q.list_repos(include_archived=False)`).
The `_refresh_known_repos` helper in `cli.py` closes that gap by
re-enumerating before ingest each sync. This test pins both halves of
the contract: the row flips, and `list_repos` then excludes the repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from github_twin.cli import _refresh_known_repos
from github_twin.config import Config
from github_twin.store import queries as q
from github_twin.store.db import open_db, transaction
from github_twin.target import Target, save_target


class _FakeGH:
    """Stand-in for `GitHubClient()` used as a context manager.

    Yields a different repo payload on each call so the test can simulate
    a repo flipping `archived=False → True` (or its visibility changing)
    between successive syncs.
    """

    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self._pages = pages
        self.calls = 0

    def __enter__(self) -> _FakeGH:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def paginate(self, path: str, *, params: dict[str, Any] | None = None):
        assert path.startswith("/orgs/")
        page = self._pages[self.calls]
        self.calls += 1
        yield from page


def _repo_payload(name: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "full_name": name,
        "default_branch": "main",
        "pushed_at": "2024-01-01T00:00:00Z",
        "archived": False,
        "visibility": "public",
        "fork": False,
        "size": 10,
    }
    base.update(over)
    return base


def _seed_org_target(conn) -> Target:
    target = Target(kind="org", name="acme", external_id=1, emails=[])
    with transaction(conn):
        target = save_target(conn, target)
    assert target.id is not None
    return target


def test_refresh_flips_archived_and_excludes_repo(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite"
    conn = open_db(db_path, embed_dim=4)
    try:
        target = _seed_org_target(conn)
        assert target.id is not None

        # First sync: repo is active. Seed the DB row directly to mirror
        # `gt init` populating the table.
        q.upsert_repo(
            conn,
            target_id=target.id,
            full_name="acme/active",
            default_branch="main",
            pushed_at="2024-01-01T00:00:00Z",
            archived=False,
            visibility="public",
        )

        # Sanity: row is present and visible to ingest.
        assert [r["full_name"] for r in q.list_repos(conn, target_id=target.id)] == ["acme/active"]

        # Second sync: GitHub now reports the repo as internal+archived.
        fake = _FakeGH([[_repo_payload("acme/active", archived=True, visibility="internal")]])
        cfg = Config()
        with patch("github_twin.cli.GitHubClient", lambda: fake):
            _refresh_known_repos(cfg, conn, target_filter=None)

        # The row's archived/visibility flipped...
        row = q.get_repo(conn, target_id=target.id, full_name="acme/active")
        assert row is not None
        assert row["archived"] == 1
        assert row["visibility"] == "internal"

        # ...and the default ingest read excludes it.
        assert q.list_repos(conn, target_id=target.id) == []
        assert [
            r["full_name"] for r in q.list_repos(conn, target_id=target.id, include_archived=True)
        ] == ["acme/active"]
    finally:
        conn.close()


def test_refresh_is_noop_for_user_mode(tmp_path: Path) -> None:
    """User-mode targets have no org repos to enumerate; refresh should
    skip them without hitting GitHub."""
    db_path = tmp_path / "db.sqlite"
    conn = open_db(db_path, embed_dim=4)
    try:
        with transaction(conn):
            save_target(conn, Target(kind="user", name="alice", external_id=1, emails=[]))

        fake = _FakeGH([])  # No pages — any call would IndexError.
        cfg = Config()
        with patch("github_twin.cli.GitHubClient", lambda: fake):
            _refresh_known_repos(cfg, conn, target_filter=None)

        assert fake.calls == 0
    finally:
        conn.close()
