"""MCP first-run bootstrap.

Two surfaces under test:

1. `bootstrap.init_target` runs the correct discovery branch for each
   kind (user, org, repo, auto) and persists the expected rows.
2. `bootstrap.status_payload` reports the right recommendation for the
   pre-data states (no target / no vectors / cwd repo not indexed /
   ready) — including the cwd-vs-DB probe that closes the
   "user-mode set up but current repo not indexed" blind spot.

The end-to-end `run_bootstrap` (init + ingest + embed) is covered by
the existing pipeline + ingest test suites; this module verifies the
new orchestration + cwd-aware diagnostics that bootstrap adds on top.
The retrieval-side guard (`NeedsInitError`) is covered separately in
`test_needs_init_guard.py`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from github_twin.config import Config, EmbedCfg, IdentityCfg, IngestCfg, VectorStoreCfg
from github_twin.mcp_server import bootstrap as bm
from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.target import load_target, load_targets
from tests.conftest import seed_target


class _FakeGH:
    """Stand-in for `GitHubClient` in discover_* / enumerate_org_repos.

    Routes paths against a dict of canned responses and supports the
    `paginate` shape for `/user/emails` and `/search/commits`. Used by
    every init_target test below.
    """

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        paginate_responses: dict[str, list[dict[str, Any]]] | None = None,
    ):
        self._responses = responses or {}
        self._paginate = paginate_responses or {}
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_json(self, path: str, *, params: dict[str, Any] | None = None):
        self.calls.append(path)
        return self._responses[path]

    def paginate(self, path: str, *, params: dict[str, Any] | None = None):
        self.calls.append(path)
        return iter(self._paginate.get(path, []))


def _cfg(tmp_path: Path) -> Config:
    return Config(
        embed=EmbedCfg(backend="fake", batch_size=10, dim=4),
        vector_store=VectorStoreCfg(backend="sqlite-vec"),
        ingest=IngestCfg(),
        identity=IdentityCfg(),
    )


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "bootstrap.sqlite", embed_dim=4)
    yield db
    db.close()


def _seed_vector(conn: sqlite3.Connection, *, target_id: int) -> int:
    """Insert one artifact + chunk + vec row so status_payload's
    'vectors > 0' branch fires."""
    aid = q.upsert_artifact(
        conn,
        target_id=target_id,
        kind="commit",
        external_id="c-0",
        source_url=None,
        repo="up/x",
        language="python",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    cid = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="code",
        text="def f(): return 1",
        context={"path": "a.py"},
        language="python",
    )
    q.write_embedding(conn, chunk_id=cid, embedding=[1.0, 0.0, 0.0, 0.0], model_id="fake")
    return cid


def _seed_origin(cwd: Path, full_name: str) -> None:
    (cwd / ".git").mkdir()
    (cwd / ".git" / "config").write_text(
        f'[remote "origin"]\n    url = https://github.com/{full_name}.git\n'
    )


# ---------- init_target: kind branches ----------


def test_init_target_user_persists_target_and_emails(conn: sqlite3.Connection, tmp_path: Path):
    gh = _FakeGH(
        responses={
            "/user": {"login": "alice", "id": 42},
            "/user/emails": [{"email": "alice@example.com"}],
        },
        paginate_responses={"/search/commits": []},
    )
    spec = bm.BootstrapSpec(kind="user")

    target = bm.init_target(_cfg(tmp_path), conn, gh, spec)

    assert target.kind == "user"
    assert target.name == "alice"
    assert target.id is not None
    assert "alice@example.com" in target.emails
    # Round-trips through DB.
    persisted = load_target(conn, kind="user")
    assert persisted is not None
    assert persisted.name == "alice"


def test_init_target_repo_persists_target_and_repo_row(conn: sqlite3.Connection, tmp_path: Path):
    gh = _FakeGH(
        responses={
            "/repos/up/x": {
                "id": 999,
                "full_name": "up/x",
                "default_branch": "main",
                "pushed_at": "2024-01-01T00:00:00Z",
                "archived": False,
                "fork": False,
                "size": 7,
            }
        }
    )
    spec = bm.BootstrapSpec(kind="repo", name="up/x")

    target = bm.init_target(_cfg(tmp_path), conn, gh, spec)

    assert (target.kind, target.name) == ("repo", "up/x")
    repos = conn.execute("SELECT full_name FROM repo").fetchall()
    assert [r["full_name"] for r in repos] == ["up/x"]


def test_init_target_repo_fork_swaps_to_upstream_by_default(
    conn: sqlite3.Connection, tmp_path: Path
):
    gh = _FakeGH(
        responses={
            "/repos/me/fork": {
                "id": 1,
                "full_name": "me/fork",
                "default_branch": "main",
                "pushed_at": "2024-01-01T00:00:00Z",
                "archived": False,
                "fork": True,
                "size": 1,
                "parent": {"id": 999, "full_name": "up/x"},
            },
            "/repos/up/x": {
                "id": 999,
                "full_name": "up/x",
                "default_branch": "main",
                "pushed_at": "2024-02-01T00:00:00Z",
                "archived": False,
                "fork": False,
                "size": 42,
            },
        }
    )
    spec = bm.BootstrapSpec(kind="repo", name="me/fork")
    messages: list[str] = []

    target = bm.init_target(_cfg(tmp_path), conn, gh, spec, report=messages.append)

    assert target.name == "up/x"
    assert gh.calls == ["/repos/me/fork", "/repos/up/x"]
    assert any("is a fork of up/x" in m for m in messages)


def test_init_target_repo_keep_fork_skips_swap(conn: sqlite3.Connection, tmp_path: Path):
    gh = _FakeGH(
        responses={
            "/repos/me/fork": {
                "id": 1,
                "full_name": "me/fork",
                "default_branch": "main",
                "pushed_at": "2024-01-01T00:00:00Z",
                "archived": False,
                "fork": True,
                "size": 1,
                "parent": {"id": 999, "full_name": "up/x"},
            }
        }
    )
    spec = bm.BootstrapSpec(kind="repo", name="me/fork", keep_fork=True)

    target = bm.init_target(_cfg(tmp_path), conn, gh, spec)

    assert target.name == "me/fork"
    assert gh.calls == ["/repos/me/fork"]


def test_init_target_auto_detects_repo_from_git_config(conn: sqlite3.Connection, tmp_path: Path):
    _seed_origin(tmp_path, "up/x")
    gh = _FakeGH(
        responses={
            "/repos/up/x": {
                "id": 999,
                "full_name": "up/x",
                "default_branch": "main",
                "pushed_at": "2024-01-01T00:00:00Z",
                "archived": False,
                "fork": False,
                "size": 7,
            }
        }
    )
    spec = bm.BootstrapSpec(path=tmp_path)  # kind=None -> auto

    target = bm.init_target(_cfg(tmp_path), conn, gh, spec)

    assert (target.kind, target.name) == ("repo", "up/x")


def test_init_target_auto_falls_back_to_user_when_no_git(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)  # no .git here
    gh = _FakeGH(
        responses={
            "/user": {"login": "alice", "id": 42},
            "/user/emails": [{"email": "alice@example.com"}],
        },
        paginate_responses={"/search/commits": []},
    )
    spec = bm.BootstrapSpec(path=tmp_path)

    target = bm.init_target(_cfg(tmp_path), conn, gh, spec)

    assert target.kind == "user"
    assert target.name == "alice"


def test_init_target_org_requires_name(conn: sqlite3.Connection, tmp_path: Path):
    with pytest.raises(ValueError, match="requires `name`"):
        bm.init_target(_cfg(tmp_path), conn, _FakeGH(), bm.BootstrapSpec(kind="org"))


def test_init_target_idempotent_on_re_run(conn: sqlite3.Connection, tmp_path: Path):
    gh = _FakeGH(
        responses={
            "/repos/up/x": {
                "id": 999,
                "full_name": "up/x",
                "default_branch": "main",
                "pushed_at": "2024-01-01T00:00:00Z",
                "archived": False,
                "fork": False,
                "size": 7,
            }
        }
    )
    spec = bm.BootstrapSpec(kind="repo", name="up/x")

    bm.init_target(_cfg(tmp_path), conn, gh, spec)
    bm.init_target(_cfg(tmp_path), conn, gh, spec)

    assert len(load_targets(conn)) == 1


# ---------- status_payload ----------


def test_status_payload_no_target(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    payload = bm.status_payload(conn)
    assert payload["targets"] == []
    assert payload["in_progress"] is False
    assert payload["recommendation"] is not None
    assert "bootstrap" in payload["recommendation"]
    assert "No target" in payload["recommendation"]


def test_status_payload_target_but_no_vectors(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    seed_target(conn, kind="repo", name="up/x", external_id=999)
    payload = bm.status_payload(conn)
    assert payload["targets"] == [{"kind": "repo", "name": "up/x"}]
    assert payload["recommendation"] is not None
    assert "no chunks" in payload["recommendation"].lower()


def test_status_payload_ready_when_vectors_present(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """No `.git` in cwd, vectors present → recommendation is None.

    Chdir to an empty tmp_path so the cwd-repo probe in `status_payload`
    finds nothing and can't trigger the 'current repo not indexed' branch.
    """
    monkeypatch.chdir(tmp_path)
    tid = seed_target(conn, kind="repo", name="up/x", external_id=999)
    _seed_vector(conn, target_id=tid)
    payload = bm.status_payload(conn)
    assert payload["recommendation"] is None
    assert payload["stats"]["vectors"] >= 1
    assert payload["current_repo"] is None
    assert payload["current_repo_indexed"] is False


def test_status_payload_flags_current_repo_when_user_indexed_but_repo_missing(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """User-mode set up + vectors present, but cwd is a different repo with
    no matching `repo` row → recommendation calls out the missing repo."""
    monkeypatch.chdir(tmp_path)
    _seed_origin(tmp_path, "other/y")
    tid = seed_target(conn, kind="user", name="alice", external_id=42)
    _seed_vector(conn, target_id=tid)

    payload = bm.status_payload(conn)

    assert payload["current_repo"] == {
        "full_name": "other/y",
        "owner": "other",
        "name": "y",
    }
    assert payload["current_repo_indexed"] is False
    assert payload["recommendation"] is not None
    assert "other/y is not in the index" in payload["recommendation"]


def test_status_payload_no_recommendation_when_current_repo_indexed(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Cwd repo IS in the `repo` table → no extra recommendation."""
    monkeypatch.chdir(tmp_path)
    _seed_origin(tmp_path, "up/x")
    tid = seed_target(conn, kind="repo", name="up/x", external_id=999)
    q.upsert_repo(
        conn,
        target_id=tid,
        full_name="up/x",
        default_branch="main",
        pushed_at=None,
        archived=False,
        fork=False,
        size_kb=1,
    )
    _seed_vector(conn, target_id=tid)

    payload = bm.status_payload(conn)

    assert payload["current_repo_indexed"] is True
    assert payload["recommendation"] is None


def test_status_payload_skips_current_repo_probe_outside_git(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """No `.git` in cwd hierarchy → current_repo is None and the probe
    silently no-ops."""
    monkeypatch.chdir(tmp_path)
    payload = bm.status_payload(conn)
    assert payload["current_repo"] is None
    assert payload["current_repo_indexed"] is False
