"""`scope` parameter on retrieval tools.

Pure unit tests of `_resolve_scope` — the integration with
`find_review_comments` / `find_style_examples` is covered by the
existing tool tests, which still pass because the default
`scope="all"` is a no-op pass-through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.mcp_server.tools import _resolve_scope
from github_twin.store.db import open_db
from github_twin.target import Target, save_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "scope.sqlite", embed_dim=4)
    yield db
    db.close()


def _seed_target(conn, *, kind: str, name: str) -> None:
    save_target(conn, Target(kind=kind, name=name, external_id=1, emails=[]))


# ---------- "all" is a no-op ----------


def test_scope_all_passes_through(conn):
    _seed_target(conn, kind="user", name="alice")
    assert _resolve_scope(conn, scope="all", repo=None, author_login=None) == (None, None)
    assert _resolve_scope(conn, scope="all", repo="r", author_login="b") == ("r", "b")


# ---------- "personal" ----------


def test_scope_personal_fills_author_in_user_mode(conn):
    _seed_target(conn, kind="user", name="alice")
    repo, author = _resolve_scope(conn, scope="personal", repo=None, author_login=None)
    assert repo is None
    assert author == "alice"


def test_scope_personal_respects_explicit_author(conn):
    """Explicit author_login wins over scope-derived default."""
    _seed_target(conn, kind="user", name="alice")
    repo, author = _resolve_scope(conn, scope="personal", repo=None, author_login="bob")
    assert author == "bob"


def test_scope_personal_no_op_in_org_mode(conn):
    """Without a user-mode target, 'personal' has no name to fill —
    falls through to None and the caller is expected to pass author
    explicitly."""
    _seed_target(conn, kind="org", name="acme")
    repo, author = _resolve_scope(conn, scope="personal", repo=None, author_login=None)
    assert author is None


def test_scope_personal_no_target_at_all(conn):
    """No target row → no resolution; values pass through unchanged."""
    assert _resolve_scope(conn, scope="personal", repo=None, author_login=None) == (None, None)


# ---------- "project" ----------


def test_scope_project_fills_repo_when_repo_mode_target(conn):
    _seed_target(conn, kind="repo", name="me/x")
    repo, author = _resolve_scope(conn, scope="project", repo=None, author_login=None)
    assert repo == "me/x"
    assert author is None


def test_scope_project_respects_explicit_repo(conn):
    _seed_target(conn, kind="repo", name="me/x")
    repo, _ = _resolve_scope(conn, scope="project", repo="custom/y", author_login=None)
    assert repo == "custom/y"


def test_scope_project_no_op_in_user_mode(conn):
    _seed_target(conn, kind="user", name="alice")
    repo, _ = _resolve_scope(conn, scope="project", repo=None, author_login=None)
    assert repo is None


def test_scope_project_no_op_in_org_mode(conn):
    _seed_target(conn, kind="org", name="acme")
    repo, _ = _resolve_scope(conn, scope="project", repo=None, author_login=None)
    assert repo is None
