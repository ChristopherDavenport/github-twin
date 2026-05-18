"""`scope` parameter on retrieval tools.

Pure unit tests of `_resolve_scope`. With multi-target support, the
function returns `(target_id, repo, author_login)`.
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


def _seed_target(conn, *, kind: str, name: str) -> Target:
    return save_target(conn, Target(kind=kind, name=name, external_id=1, emails=[]))


# ---------- "all" is a no-op ----------


def test_scope_all_passes_through(conn):
    _seed_target(conn, kind="user", name="alice")
    assert _resolve_scope(conn, scope="all", target=None, repo=None, author_login=None) == (
        None,
        None,
        None,
    )
    assert _resolve_scope(conn, scope="all", target=None, repo="r", author_login="b") == (
        None,
        "r",
        "b",
    )


# ---------- "personal" ----------


def test_scope_personal_fills_only_target_in_user_mode(conn):
    """User-mode rows have author_login=NULL by design, so personal-scope
    must NOT fill author_login from the target name or every result would
    be zeroed out. target_id alone narrows correctly. Pins issue #13.
    """
    user = _seed_target(conn, kind="user", name="alice")
    tid, repo, author = _resolve_scope(
        conn, scope="personal", target=None, repo=None, author_login=None
    )
    assert repo is None
    assert author is None
    assert tid == user.id


def test_scope_personal_respects_explicit_author(conn):
    """Explicit author_login wins over scope-derived default."""
    _seed_target(conn, kind="user", name="alice")
    _tid, repo, author = _resolve_scope(
        conn, scope="personal", target=None, repo=None, author_login="bob"
    )
    assert author == "bob"


def test_scope_personal_no_op_in_org_mode(conn):
    """Without a user-mode target, 'personal' leaves author None."""
    _seed_target(conn, kind="org", name="acme")
    tid, _repo, author = _resolve_scope(
        conn, scope="personal", target=None, repo=None, author_login=None
    )
    assert author is None
    assert tid is None


def test_scope_personal_no_target_at_all(conn):
    """No target row → no resolution; values pass through unchanged."""
    assert _resolve_scope(conn, scope="personal", target=None, repo=None, author_login=None) == (
        None,
        None,
        None,
    )


# ---------- "project" ----------


def test_scope_project_fills_repo_when_repo_mode_target(conn):
    repo_target = _seed_target(conn, kind="repo", name="me/x")
    tid, repo, author = _resolve_scope(
        conn, scope="project", target=None, repo=None, author_login=None
    )
    assert repo == "me/x"
    assert author is None
    assert tid == repo_target.id


def test_scope_project_respects_explicit_repo(conn):
    _seed_target(conn, kind="repo", name="me/x")
    _tid, repo, _ = _resolve_scope(
        conn, scope="project", target=None, repo="custom/y", author_login=None
    )
    assert repo == "custom/y"


def test_scope_project_no_op_in_user_mode(conn):
    _seed_target(conn, kind="user", name="alice")
    _tid, repo, _ = _resolve_scope(conn, scope="project", target=None, repo=None, author_login=None)
    assert repo is None


def test_scope_project_no_op_in_org_mode(conn):
    _seed_target(conn, kind="org", name="acme")
    _tid, repo, _ = _resolve_scope(conn, scope="project", target=None, repo=None, author_login=None)
    assert repo is None


# ---------- explicit target= overrides ----------


def test_target_parameter_overrides_scope(conn):
    """Passing target=NAME narrows to that target regardless of scope."""
    user = _seed_target(conn, kind="user", name="alice")
    _seed_target(conn, kind="org", name="acme")
    tid, _repo, _author = _resolve_scope(
        conn, scope="all", target="alice", repo=None, author_login=None
    )
    assert tid == user.id
