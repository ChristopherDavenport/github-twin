"""Tests for `ingest/clone.py` orchestration.

The git subprocess is monkeypatched so we test the context-manager
semantics (tempdir, rmtree on exit, cache path layout) without hitting
GitHub or the local `git` binary. URL helpers are tested directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.ingest import clone as clone_mod


def test_auth_url_embeds_token():
    assert clone_mod._auth_url("org/repo", "tkn-123") == (
        "https://oauth2:tkn-123@github.com/org/repo.git"
    )


def test_public_url_has_no_token():
    assert clone_mod._public_url("org/repo") == "https://github.com/org/repo.git"
    assert "tkn" not in clone_mod._public_url("org/repo")


def _patch_git_to_pretend_clone(monkeypatch, *, head_sha: str = "abc123"):
    """Replace _fresh_clone and _fetch_update so they create a .git marker
    and return a fixed sha without invoking git."""

    calls: dict[str, list[dict]] = {"fresh": [], "fetch": []}

    def fake_fresh(
        path: Path,
        full_name: str,
        token: str,
        *,
        depth=1,
        shallow_since=None,
    ) -> str:
        path.mkdir(parents=True, exist_ok=True)
        (path / ".git").mkdir(exist_ok=True)
        (path / "README.md").write_text("hi")
        calls["fresh"].append({"path": path, "depth": depth, "shallow_since": shallow_since})
        return head_sha

    def fake_fetch(
        path: Path,
        full_name: str,
        token: str,
        *,
        depth=1,
        shallow_since=None,
    ) -> str:
        # Indicate the existing tree was refreshed.
        (path / "REFRESHED").write_text("yes")
        calls["fetch"].append({"path": path, "depth": depth, "shallow_since": shallow_since})
        return head_sha

    monkeypatch.setattr(clone_mod, "_fresh_clone", fake_fresh)
    monkeypatch.setattr(clone_mod, "_fetch_update", fake_fetch)
    monkeypatch.setattr(clone_mod, "_resolve_token", lambda: "fake-token")
    return calls


def test_cloned_repo_purges_tempdir(monkeypatch):
    _patch_git_to_pretend_clone(monkeypatch)
    seen_path: Path | None = None
    with clone_mod.cloned_repo("org/repo") as cr:
        seen_path = cr.path
        assert cr.full_name == "org/repo"
        assert cr.head_sha == "abc123"
        assert cr.from_cache is False
        assert seen_path.exists()
        assert (seen_path / ".git").is_dir()
    assert seen_path is not None
    assert not seen_path.exists()


def test_cloned_repo_purges_tempdir_on_exception(monkeypatch):
    _patch_git_to_pretend_clone(monkeypatch)
    seen_path: Path | None = None
    with (
        pytest.raises(RuntimeError, match="boom"),
        clone_mod.cloned_repo("org/repo") as cr,
    ):
        seen_path = cr.path
        assert seen_path.exists()
        raise RuntimeError("boom")
    assert seen_path is not None
    assert not seen_path.exists()


def test_cloned_repo_cache_layout(monkeypatch, tmp_path: Path):
    _patch_git_to_pretend_clone(monkeypatch)
    cache_dir = tmp_path / "cache"
    with clone_mod.cloned_repo("org/repo", cache_dir=cache_dir) as cr:
        assert cr.path == cache_dir / "org" / "repo"
        assert cr.from_cache is False
        assert cr.path.exists()
    # Cache mode preserves the clone on exit.
    assert (cache_dir / "org" / "repo").exists()


def test_cloned_repo_cache_reuse_calls_fetch(monkeypatch, tmp_path: Path):
    _patch_git_to_pretend_clone(monkeypatch)
    cache_dir = tmp_path / "cache"
    # First call clones fresh.
    with clone_mod.cloned_repo("org/repo", cache_dir=cache_dir) as cr1:
        assert cr1.from_cache is False
        assert not (cr1.path / "REFRESHED").exists()
    # Second call reuses + calls _fetch_update (which writes REFRESHED).
    with clone_mod.cloned_repo("org/repo", cache_dir=cache_dir) as cr2:
        assert cr2.from_cache is True
        assert (cr2.path / "REFRESHED").exists()


def test_commits_clone_uses_full_depth(monkeypatch, tmp_path: Path):
    calls = _patch_git_to_pretend_clone(monkeypatch)
    cache_dir = tmp_path / "cache"
    with clone_mod.commits_clone("org/repo", cache_dir=cache_dir):
        pass
    # Fresh clone for commits ingest must request deep history (depth=None).
    assert calls["fresh"][0]["depth"] is None
    # And the cache layout matches.
    assert (cache_dir / "org" / "repo").exists()


def test_commits_clone_reuse_unshallows(monkeypatch, tmp_path: Path):
    calls = _patch_git_to_pretend_clone(monkeypatch)
    cache_dir = tmp_path / "cache"
    # First call: fresh deep clone.
    with clone_mod.commits_clone("org/repo", cache_dir=cache_dir):
        pass
    # Second call: existing clone, fetch path triggered with depth=None
    # (the unshallow / plain-fetch decision happens inside _fetch_update;
    # here we only verify the depth is plumbed through).
    with clone_mod.commits_clone("org/repo", cache_dir=cache_dir):
        pass
    assert calls["fetch"][0]["depth"] is None


def test_cloned_repo_default_depth_is_one(monkeypatch, tmp_path: Path):
    calls = _patch_git_to_pretend_clone(monkeypatch)
    cache_dir = tmp_path / "cache"
    with clone_mod.cloned_repo("org/repo", cache_dir=cache_dir):
        pass
    assert calls["fresh"][0]["depth"] == 1


def test_is_shallow_helper(monkeypatch):
    """_is_shallow parses git's literal 'true'/'false' output."""
    monkeypatch.setattr(clone_mod, "_git", lambda args, *, cwd=None: "true")
    assert clone_mod._is_shallow(Path("/x")) is True
    monkeypatch.setattr(clone_mod, "_git", lambda args, *, cwd=None: "false")
    assert clone_mod._is_shallow(Path("/x")) is False


def test_head_sha_raises_empty_repo_error_on_unknown_revision(monkeypatch, tmp_path: Path):
    """A successful clone of an empty repo (no commits) fails `git rev-parse HEAD`
    with 'ambiguous argument' / 'unknown revision'. `_head_sha` converts that into
    `EmptyRepoError` so callers can distinguish it from a real clone failure."""

    def fake_git(args, *, cwd=None):
        raise clone_mod.CloneError(
            "git rev-parse HEAD failed (128): fatal: ambiguous argument 'HEAD': "
            "unknown revision or path not in the working tree."
        )

    monkeypatch.setattr(clone_mod, "_git", fake_git)
    with pytest.raises(clone_mod.EmptyRepoError, match="no commits"):
        clone_mod._head_sha(tmp_path, "org/empty-repo")


def test_head_sha_propagates_other_clone_errors(monkeypatch, tmp_path: Path):
    """Non-empty-repo failures (e.g. corrupted .git dir) keep raising plain CloneError."""

    def fake_git(args, *, cwd=None):
        raise clone_mod.CloneError("git rev-parse HEAD failed (128): fatal: not a git repository")

    monkeypatch.setattr(clone_mod, "_git", fake_git)
    with pytest.raises(clone_mod.CloneError, match="not a git repository") as exc:
        clone_mod._head_sha(tmp_path, "org/repo")
    assert not isinstance(exc.value, clone_mod.EmptyRepoError)


def test_empty_repo_error_is_clone_error_subclass():
    """Existing `except CloneError` blocks must continue to catch empty-repo errors
    when callers haven't been updated to special-case them."""
    assert issubclass(clone_mod.EmptyRepoError, clone_mod.CloneError)


def test_fetch_update_quiets_shallow_info_error(monkeypatch, tmp_path: Path):
    """`git fetch --shallow-since=<date>` raises 'error processing shallow info'
    when no commits exist newer than the cutoff (quiet template repos). Treat
    that as a no-op: return current HEAD, never raise, and still scrub the
    remote so the auth URL doesn't linger.
    """
    path = tmp_path / "repo"
    path.mkdir()
    calls: list[list[str]] = []

    def fake_git(args, *, cwd=None):
        calls.append(list(args))
        cmd = args[0]
        if cmd == "fetch":
            raise clone_mod.CloneError(
                "git fetch --shallow-since=2026-05-26 --no-tags origin failed (128): "
                "fatal: error processing shallow info: 4"
            )
        if cmd == "rev-parse":
            return "deadbeef"
        return ""

    monkeypatch.setattr(clone_mod, "_git", fake_git)

    head = clone_mod._fetch_update(
        path, "org/quiet-repo", "tkn", depth=1, shallow_since="2026-05-26"
    )
    assert head == "deadbeef"
    # Remote was set to auth URL up front; scrub must still run via the finally.
    set_url_calls = [c for c in calls if c[:2] == ["remote", "set-url"]]
    assert set_url_calls[0][3] == clone_mod._auth_url("org/quiet-repo", "tkn")
    assert set_url_calls[-1][3] == clone_mod._public_url("org/quiet-repo")


def test_fetch_update_propagates_other_clone_errors(monkeypatch, tmp_path: Path):
    """Only the 'shallow info' case is quieted; other fetch failures still raise."""
    path = tmp_path / "repo"
    path.mkdir()

    def fake_git(args, *, cwd=None):
        if args[0] == "fetch":
            raise clone_mod.CloneError("git fetch failed (128): fatal: remote hung up")
        return ""

    monkeypatch.setattr(clone_mod, "_git", fake_git)

    with pytest.raises(clone_mod.CloneError, match="remote hung up"):
        clone_mod._fetch_update(path, "org/repo", "tkn", depth=1, shallow_since="2026-05-26")


def test_git_decodes_non_utf8_output_with_replacement(monkeypatch, tmp_path: Path):
    """`git show` of a diff containing non-UTF-8 bytes must not crash _git.

    Reproduces the prod traceback: `UnicodeDecodeError: 'utf-8' codec can't
    decode byte 0x95`. We shim a fake `git` executable on PATH that emits
    bytes including the offending 0x95 (Windows-1252 bullet), then assert
    `_git` returns a string with U+FFFD replacements.
    """
    fake_git_dir = tmp_path / "bin"
    fake_git_dir.mkdir()
    fake_git = fake_git_dir / "git"
    # Python is a hard dev dep; using it for the shim avoids shell-quoting issues.
    fake_git.write_text(
        "#!/usr/bin/env python3\nimport sys\nsys.stdout.buffer.write(b'hello \\x95 world')\n"
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_git_dir}:{__import__('os').environ['PATH']}")

    out = clone_mod._git(["show", "deadbeef"])
    assert "hello" in out
    assert "world" in out
    assert "�" in out  # U+FFFD REPLACEMENT CHARACTER
