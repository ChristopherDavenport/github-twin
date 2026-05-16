"""Default data-dir resolution.

Backward-compat is the load-bearing case: anyone who ran `gt sync`
from the source tree before this change has their corpus at `./data`,
and that has to keep working when they update the package. Behind
that, XDG_DATA_HOME (or the `~/.local/share/<name>` fallback) is the
new default for fresh installs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.config import Config, _default_data_dir


def test_cwd_data_takes_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """If `./data` exists in cwd, it wins regardless of XDG state —
    that's the backward-compat guarantee for early adopters."""
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert _default_data_dir() == tmp_path / "data"


def test_xdg_data_home_when_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """With no `./data` in cwd, `XDG_DATA_HOME` defines the base."""
    monkeypatch.chdir(tmp_path)  # cwd has no `./data`
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert _default_data_dir() == tmp_path / "xdg" / "github-twin"


def test_falls_back_to_local_share(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No `./data`, no XDG_DATA_HOME → `~/.local/share/github-twin`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # `Path.home()` consults HOME on Unix; this works without writing files.
    assert _default_data_dir() == tmp_path / "home" / ".local" / "share" / "github-twin"


def test_gt_paths_data_dir_env_overrides_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The env-driven override is what users actually rely on for
    multi-target setups (one DB per target). It must still win even
    after the XDG default change."""
    monkeypatch.chdir(tmp_path)
    custom = tmp_path / "twin-http4s"
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(custom))
    cfg = Config()
    assert cfg.paths.data_dir == custom


def test_config_uses_default_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`Config()` resolves the default lazily — instances created in
    different cwds pick up different defaults if `./data` is locally
    present in only one of them."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GT_PATHS__DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    cfg = Config()
    # No `./data` in this temp cwd → falls through to XDG path.
    assert "github-twin" in str(cfg.paths.data_dir)
    assert cfg.paths.data_dir != tmp_path / "data"


def test_raw_dir_and_db_path_derive_from_data_dir(tmp_path: Path):
    from github_twin.config import PathsCfg

    cfg = PathsCfg(data_dir=tmp_path / "custom")
    assert cfg.raw_dir == tmp_path / "custom" / "raw"
    assert cfg.db_path == tmp_path / "custom" / "db.sqlite"
