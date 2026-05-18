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

    def fake_fresh(path: Path, full_name: str, token: str, *, depth=1) -> str:
        path.mkdir(parents=True, exist_ok=True)
        (path / ".git").mkdir(exist_ok=True)
        (path / "README.md").write_text("hi")
        calls["fresh"].append({"path": path, "depth": depth})
        return head_sha

    def fake_fetch(path: Path, full_name: str, token: str, *, depth=1) -> str:
        # Indicate the existing tree was refreshed.
        (path / "REFRESHED").write_text("yes")
        calls["fetch"].append({"path": path, "depth": depth})
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
