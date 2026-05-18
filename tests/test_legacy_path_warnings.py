"""`_warn_legacy_cwd_paths` fires when stray files from the old
(cwd-relative) layout sit in the current directory but the resolved
data_dir is elsewhere. The warning is informational only — no
auto-move (silent moves are scarier than the warning).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from github_twin.cli import _warn_legacy_cwd_paths


def test_stray_cwd_config_triggers_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "twin"))
    (tmp_path / "config.toml").write_text('[embed]\nbackend = "ollama"\n')
    with caplog.at_level(logging.WARNING, logger="github_twin.cli"):
        _warn_legacy_cwd_paths()
    msgs = [r.getMessage() for r in caplog.records]
    assert any("legacy ./config.toml" in m for m in msgs), msgs


def test_no_warning_when_cwd_clean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "twin"))
    with caplog.at_level(logging.WARNING, logger="github_twin.cli"):
        _warn_legacy_cwd_paths()
    assert not [r for r in caplog.records if "legacy" in r.getMessage()]


def test_no_warning_when_data_dir_config_already_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """If config.toml exists in BOTH cwd and data_dir, the user has
    already migrated (the cwd one is just stale) — don't nag."""
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "twin"
    data_dir.mkdir()
    (tmp_path / "config.toml").write_text('[embed]\nbackend = "ollama"\n')
    (data_dir / "config.toml").write_text('[embed]\nbackend = "ollama"\n')
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(data_dir))
    with caplog.at_level(logging.WARNING, logger="github_twin.cli"):
        _warn_legacy_cwd_paths()
    assert not [r for r in caplog.records if "legacy ./config.toml" in r.getMessage()]


def test_stray_cwd_db_triggers_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "twin"))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "db.sqlite").write_bytes(b"")
    with caplog.at_level(logging.WARNING, logger="github_twin.cli"):
        _warn_legacy_cwd_paths()
    msgs = [r.getMessage() for r in caplog.records]
    assert any("legacy ./data/db.sqlite" in m for m in msgs), msgs


def test_no_db_warning_when_data_dir_is_cwd_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """If the user explicitly set GT_PATHS__DATA_DIR=./data, that IS
    the resolved data dir — no migration needed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "db.sqlite").write_bytes(b"")
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "data"))
    with caplog.at_level(logging.WARNING, logger="github_twin.cli"):
        _warn_legacy_cwd_paths()
    assert not [r for r in caplog.records if "legacy ./data" in r.getMessage()]
