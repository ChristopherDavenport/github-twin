"""Tests for `ingest_files` (org-mode file-at-HEAD walk).

We don't hit GitHub or git here. The clone step is monkeypatched to point
at a local tempdir we pre-seed, so the test is hermetic and fast.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

from github_twin.config import IngestCfg
from github_twin.ingest import files as files_mod
from github_twin.ingest.clone import ClonedRepo
from github_twin.store import queries as q
from github_twin.store.db import open_db


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    yield db
    db.close()


def _seed_worktree(root: Path) -> None:
    """A tiny in-memory 'repo' with one chunkable file + several skips."""
    (root / "src").mkdir()
    (root / "src" / "lib.py").write_text("\n".join(f"def fn_{i}(): return {i}" for i in range(10)))
    # Use a nested vendor dir because the default exclude `**/vendor/**`
    # only matches `<some>/vendor/<some>`, not root-level `vendor/`.
    (root / "third_party" / "vendor").mkdir(parents=True)
    (root / "third_party" / "vendor" / "skip.py").write_text("# vendored\n" * 20)
    (root / "README.md").write_text("# read\n" * 5)  # markdown is excluded by language
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("# pretend git metadata")
    # Symlink that points outside the tree — must be skipped.
    with suppress(OSError):  # Windows lacks symlink privileges by default
        (root / "src" / "link.py").symlink_to(root / "src" / "lib.py")


def _patch_clone_to_local(monkeypatch, root: Path, *, head_sha: str = "deadbeef"):
    @contextmanager
    def fake_cloned_repo(full_name, *, cache_dir=None, token=None):
        yield ClonedRepo(
            full_name=full_name,
            path=root,
            head_sha=head_sha,
            from_cache=False,
        )

    monkeypatch.setattr(files_mod, "cloned_repo", fake_cloned_repo)


def test_ingest_files_walks_one_repo(tmp_path: Path, conn, monkeypatch):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    _seed_worktree(worktree)

    q.upsert_repo(
        conn,
        full_name="org/repo",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        archived=False,
        fork=False,
        size_kb=10,
    )
    _patch_clone_to_local(monkeypatch, worktree)

    cfg = IngestCfg()
    stats = files_mod.ingest_files(conn=conn, cfg=cfg)

    assert stats.repos_visited == 1
    assert stats.repos_skipped == 0
    assert stats.files_chunked == 1  # only src/lib.py made it through filters
    assert stats.chunks_written >= 1

    # The artifact + chunks landed in the DB with the right shape.
    arts = conn.execute(
        "SELECT kind, external_id, repo, language, source_url FROM artifact"
    ).fetchall()
    assert len(arts) == 1
    a = arts[0]
    assert a["kind"] == "file"
    assert a["external_id"] == "org/repo:src/lib.py"
    assert a["repo"] == "org/repo"
    assert a["language"] == "python"
    assert "/blob/deadbeef/src/lib.py" in a["source_url"]

    chunks = conn.execute("SELECT kind, language FROM chunk").fetchall()
    assert all(c["kind"] == "file" for c in chunks)
    assert all(c["language"] == "python" for c in chunks)

    # The repo cursor advanced.
    row = q.get_repo(conn, "org/repo")
    assert row["head_sha"] == "deadbeef"
    assert row["last_files_at"] is not None


def test_ingest_files_skips_unchanged_repo(tmp_path: Path, conn, monkeypatch):
    """If pushed_at <= last_files_at we shouldn't even open a clone."""
    q.upsert_repo(
        conn,
        full_name="org/cold",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        archived=False,
        fork=False,
        size_kb=1,
    )
    q.set_repo_cursor(
        conn,
        full_name="org/cold",
        files_at="2024-06-01T00:00:00Z",
    )

    called = {"n": 0}

    @contextmanager
    def boom(*a, **k):
        called["n"] += 1
        yield None  # would never get here

    monkeypatch.setattr(files_mod, "cloned_repo", boom)

    stats = files_mod.ingest_files(conn=conn, cfg=IngestCfg())
    assert stats.repos_visited == 0
    assert stats.repos_skipped == 1
    assert called["n"] == 0


def test_ingest_files_skips_oversize_repo(tmp_path: Path, conn, monkeypatch):
    q.upsert_repo(
        conn,
        full_name="org/huge",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        archived=False,
        fork=False,
        size_kb=10_000_000,
    )
    monkeypatch.setattr(
        files_mod,
        "cloned_repo",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not clone")),
    )
    stats = files_mod.ingest_files(conn=conn, cfg=IngestCfg(max_repo_size_kb=500_000))
    assert stats.repos_visited == 0
    assert stats.repos_skipped == 1


def test_ingest_files_is_idempotent(tmp_path: Path, conn, monkeypatch):
    """Re-running the same head_sha must not duplicate artifacts or chunks."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    _seed_worktree(worktree)

    q.upsert_repo(
        conn,
        full_name="org/repo",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        size_kb=10,
    )
    _patch_clone_to_local(monkeypatch, worktree)

    # First pass.
    files_mod.ingest_files(conn=conn, cfg=IngestCfg())
    n_art1 = conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    n_chunk1 = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]

    # Reset the cursor so the skip-if-unchanged check doesn't short-circuit
    # the second run — we want to verify upsert semantics, not skip behavior.
    conn.execute("UPDATE repo SET last_files_at = NULL WHERE full_name='org/repo'")

    files_mod.ingest_files(conn=conn, cfg=IngestCfg())
    n_art2 = conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    n_chunk2 = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]

    assert n_art1 == n_art2
    assert n_chunk1 == n_chunk2
