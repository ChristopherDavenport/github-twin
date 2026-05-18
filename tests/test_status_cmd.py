"""`gt status` — diagnostic dump of where files live and what's wired up.

Side-effect-free: must not create the DB or any directories. The whole
point of the command is to answer "where IS my data?" before the user
runs anything destructive.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from github_twin.cli import app


def _run(args: list[str]) -> object:
    return CliRunner().invoke(app, args)


def test_status_runs_without_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No data_dir on disk — status should still print paths and the
    'not created yet' marker, without creating the DB."""
    data_dir = tmp_path / "twin"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(data_dir))

    result = _run(["status"])
    assert result.exit_code == 0, result.output

    # Rich may line-wrap long paths in the CliRunner's 80-col terminal;
    # check for the trailing directory name + the marker instead of the
    # full absolute path.
    assert "twin" in result.output
    assert "not created yet" in result.output
    assert "No DB yet" in result.output
    # Confirm side-effect-free: status must NOT create the data_dir.
    assert not data_dir.exists()


def test_status_shows_backends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / "twin"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(data_dir))

    result = _run(["status"])
    assert result.exit_code == 0, result.output
    # Defaults: ollama / nomic-embed-text / 768, sqlite-vec, rule expansion.
    assert "ollama" in result.output
    assert "nomic-embed-text" in result.output
    assert "768" in result.output
    assert "sqlite-vec" in result.output
    assert "rule" in result.output


def test_status_lists_targets_after_init(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """After `gt init`, status should list the target."""
    from github_twin.target import Target

    data_dir = tmp_path / "twin"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(data_dir))

    class _FakeGH:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("github_twin.cli.GitHubClient", _FakeGH)
    monkeypatch.setattr(
        "github_twin.cli.discover_user",
        lambda gh, identity: Target(kind="user", name="alice", external_id=42, emails=[]),
    )
    init_result = _run(["init", "--kind", "user"])
    assert init_result.exit_code == 0, init_result.output

    result = _run(["status"])
    assert result.exit_code == 0, result.output
    assert "alice" in result.output
    assert "user" in result.output
    # DB existed → size marker on the DB line. (config.toml is only written
    # when --embed-backend flags are passed to init, so it can still be
    # "not created yet" here.)
    assert "No DB yet" not in result.output
    # Database is alongside something with a kB/MB size marker.
    assert "KB" in result.output or "MB" in result.output


def test_status_flags_legacy_cwd_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / "twin"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(data_dir))
    (tmp_path / "config.toml").write_text('[embed]\nbackend = "ollama"\n')

    result = _run(["status"])
    assert result.exit_code == 0, result.output
    assert "Legacy paths detected" in result.output
    assert "./config.toml" in result.output


def test_status_clean_cwd_no_legacy_section(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "twin"))
    result = _run(["status"])
    assert result.exit_code == 0, result.output
    assert "Legacy paths detected" not in result.output


def test_status_full_shows_pipeline_section(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """--full adds Pipeline (embed coverage + summarize coverage + cursors)
    when a DB with at least one target exists."""
    from github_twin.target import Target

    data_dir = tmp_path / "twin"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(data_dir))

    class _FakeGH:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("github_twin.cli.GitHubClient", _FakeGH)
    monkeypatch.setattr(
        "github_twin.cli.discover_user",
        lambda gh, identity: Target(kind="user", name="alice", external_id=42, emails=[]),
    )
    assert _run(["init", "--kind", "user"]).exit_code == 0

    # Without --full, no Pipeline section.
    plain = _run(["status"])
    assert plain.exit_code == 0, plain.output
    assert "Pipeline" not in plain.output

    # With --full, Pipeline section appears with the expected subsections.
    full = _run(["status", "--full"])
    assert full.exit_code == 0, full.output
    assert "Pipeline" in full.output
    assert "Embed:" in full.output
    assert "Summarize:" in full.output
    assert "Ingest cursors:" in full.output
    # Fresh DB → text_version cursor hasn't been written yet.
    assert "never run" in full.output


def test_status_full_no_db_skips_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """--full against a missing DB just falls through — no crash, no
    Pipeline section (there's nothing to report)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "twin"))
    result = _run(["status", "--full"])
    assert result.exit_code == 0, result.output
    assert "No DB yet" in result.output
    assert "Pipeline" not in result.output
