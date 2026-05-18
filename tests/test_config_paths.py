"""Default data-dir resolution.

The resolver is intentionally pure with respect to cwd: it reads
`GT_PATHS__DATA_DIR`, then `XDG_DATA_HOME`, then falls back to
`~/.local/share/github-twin`. No `Path.cwd()` reads. This is the
load-bearing invariant — it lets us look up `config.toml` inside the
data dir without a chicken-and-egg.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.config import (
    Config,
    _default_data_dir,
    config_path_for,
    resolve_data_dir,
)


def test_cwd_data_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A stray `./data` in cwd no longer hijacks the resolver — that
    was a foot-gun (any directory called `data` captured the DB)."""
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GT_PATHS__DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert resolve_data_dir() == tmp_path / "home" / ".local" / "share" / "github-twin"


def test_xdg_data_home_when_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GT_PATHS__DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert resolve_data_dir() == tmp_path / "xdg" / "github-twin"


def test_falls_back_to_local_share(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GT_PATHS__DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert resolve_data_dir() == tmp_path / "home" / ".local" / "share" / "github-twin"


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    custom = tmp_path / "twin-http4s"
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(custom))
    assert resolve_data_dir() == custom


def test_env_override_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GT_PATHS__DATA_DIR", "~/twin-data")
    assert resolve_data_dir() == tmp_path / "home" / "twin-data"


def test_default_factory_matches_resolver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("GT_PATHS__DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert _default_data_dir() == resolve_data_dir()


def test_config_uses_default_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GT_PATHS__DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cfg = Config()
    assert cfg.paths.data_dir == tmp_path / "home" / ".local" / "share" / "github-twin"


def test_raw_dir_and_db_path_derive_from_data_dir(tmp_path: Path):
    from github_twin.config import PathsCfg

    cfg = PathsCfg(data_dir=tmp_path / "custom")
    assert cfg.raw_dir == tmp_path / "custom" / "raw"
    assert cfg.db_path == tmp_path / "custom" / "db.sqlite"


def test_config_path_for_uses_data_dir(tmp_path: Path):
    assert config_path_for(tmp_path / "twin") == tmp_path / "twin" / "config.toml"


def test_config_path_for_defaults_to_resolver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "twin"))
    assert config_path_for() == tmp_path / "twin" / "config.toml"


def test_config_load_reads_from_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / "twin"
    data_dir.mkdir()
    (data_dir / "config.toml").write_text('[embed]\nbackend = "gemini"\ndim = 1536\n')
    monkeypatch.chdir(tmp_path)  # cwd has no config.toml
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(data_dir))
    cfg = Config.load()
    assert cfg.embed.backend == "gemini"
    assert cfg.embed.dim == 1536


def test_config_load_honors_explicit_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    explicit = tmp_path / "elsewhere.toml"
    explicit.write_text('[embed]\nbackend = "gemini"\ndim = 1536\n')
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "other"))
    cfg = Config.load(explicit)
    assert cfg.embed.backend == "gemini"


def test_config_load_ignores_cwd_config_toml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A stray ./config.toml in cwd does NOT get loaded — config now
    lives next to the data, not the cwd."""
    (tmp_path / "config.toml").write_text('[embed]\nbackend = "gemini"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "twin"))
    cfg = Config.load()
    assert cfg.embed.backend == "ollama"  # default, NOT what's in stray cwd file
