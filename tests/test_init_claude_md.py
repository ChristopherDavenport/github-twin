"""`gt init-claude-md` command — writes a CLAUDE.md template.

Uses Typer's CliRunner against the registered app so we exercise the
real argument-parsing path (overwrite guard, --server-name override)
and the target lookup against an in-memory DB.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from github_twin.cli import app
from github_twin.templates.claude_md import render


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the CLI in a clean tmp_path with a private data dir."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "data"))
    # Avoid hitting any global config.toml from the dev tree.
    monkeypatch.delenv("GT_EMBED__MODEL", raising=False)
    monkeypatch.delenv("GT_EMBED__DIM", raising=False)
    return tmp_path


def _seed_target(tmp_path: Path) -> None:
    """Pre-populate the target table so the template renders a real name."""
    from github_twin.config import load_config
    from github_twin.store.db import open_db
    from github_twin.target import Target, save_target

    cfg = load_config()
    conn = open_db(cfg.paths.db_path, cfg.embed.dim)
    try:
        save_target(conn, Target(kind="user", name="alice", external_id=42, emails=[]))
    finally:
        conn.close()


# ---------- happy path ----------


def test_writes_file_with_target_name_substituted(cli_env: Path):
    _seed_target(cli_env)
    runner = CliRunner()
    result = runner.invoke(app, ["init-claude-md"])
    assert result.exit_code == 0, result.output
    out = cli_env / "CLAUDE.md"
    assert out.exists()
    content = out.read_text()
    assert "**alice**" in content  # target_name slot
    assert "**github-twin**" in content  # default server_name
    assert "mcp__github-twin__house_rules" in content
    assert "mcp__github-twin__developer_profile" in content


def test_custom_output_path(cli_env: Path):
    _seed_target(cli_env)
    custom = cli_env / "docs" / "CONTEXT.md"
    runner = CliRunner()
    result = runner.invoke(app, ["init-claude-md", "--output", str(custom)])
    assert result.exit_code == 0, result.output
    assert custom.exists()
    # Parent directory was created.
    assert custom.parent.is_dir()


def test_custom_server_name_threaded_through(cli_env: Path):
    _seed_target(cli_env)
    runner = CliRunner()
    result = runner.invoke(app, ["init-claude-md", "--server-name", "my-twin"])
    assert result.exit_code == 0, result.output
    content = (cli_env / "CLAUDE.md").read_text()
    assert "**my-twin**" in content
    assert "mcp__my-twin__house_rules" in content
    # Default 'github-twin' tool names should not appear under custom server.
    assert "mcp__github-twin__house_rules" not in content


# ---------- guards ----------


def test_refuses_to_overwrite_without_flag(cli_env: Path):
    _seed_target(cli_env)
    out = cli_env / "CLAUDE.md"
    out.write_text("preexisting\n")

    runner = CliRunner()
    result = runner.invoke(app, ["init-claude-md"])
    assert result.exit_code != 0
    # File untouched.
    assert out.read_text() == "preexisting\n"
    assert "already exists" in result.output


def test_overwrite_flag_replaces_existing_file(cli_env: Path):
    _seed_target(cli_env)
    out = cli_env / "CLAUDE.md"
    out.write_text("preexisting\n")

    runner = CliRunner()
    result = runner.invoke(app, ["init-claude-md", "--overwrite"])
    assert result.exit_code == 0, result.output
    content = out.read_text()
    assert "preexisting" not in content
    assert "**alice**" in content


def test_falls_back_when_no_target_set(cli_env: Path):
    """No `gt init` yet → uses a placeholder name and warns the user."""
    runner = CliRunner()
    result = runner.invoke(app, ["init-claude-md"])
    assert result.exit_code == 0, result.output
    content = (cli_env / "CLAUDE.md").read_text()
    assert "run `gt init` first" in content
    assert "placeholder" in result.output.lower()


# ---------- direct render() invariants ----------


def test_render_substitutes_all_placeholders():
    out = render(target_name="bob", server_name="my-srv", date="2026-05-15")
    assert "**bob**" in out
    assert "**my-srv**" in out
    assert "2026-05-15" in out
    # No leftover unrendered curly-brace slots.
    assert "{target_name}" not in out
    assert "{server_name}" not in out
    assert "{date}" not in out


def test_render_does_not_have_dangling_format_specs():
    """Catch the classic 'forgot to escape `{...}` in the template' bug."""
    out = render(target_name="x", server_name="y", date="2026-01-01")
    # No braces should survive rendering since the template has no escaped
    # `{{` / `}}` blocks today. If you add them later, update this test.
    assert "{" not in out
    assert "}" not in out
