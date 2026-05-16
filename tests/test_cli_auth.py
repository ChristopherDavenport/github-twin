"""`gt auth login / status / logout` Typer commands.

Mocks the OAuth network calls + the post-auth identity probe, then
exercises the actual Typer dispatch path including option parsing
and rich-table rendering.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from github_twin.cli import app
from github_twin.ingest import auth_storage, oauth


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    return tmp_path


class _FakeGitHubClient:
    """Stands in for `GitHubClient` so `gt auth login` can capture a login
    without hitting api.github.com."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _FakeGitHubClient:
        return self

    def __exit__(self, *a: object) -> None:
        pass

    def request(self, method: str, path: str) -> _FakeResp:
        assert (method, path) == ("GET", "/user")
        return _FakeResp({"login": "alice"})


class _FakeResp:
    def __init__(self, body: object) -> None:
        self._body = body

    def json(self) -> object:
        return self._body


@pytest.fixture
def fake_oauth(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Mock both halves of the device flow and the identity probe."""
    monkeypatch.setattr(
        oauth,
        "request_device_code",
        lambda client_id, scope: oauth.DeviceCodeResponse(
            device_code="DC",
            user_code="WDJB-MJHT",
            verification_uri="https://github.com/login/device",
            verification_uri_complete="https://github.com/login/device?user_code=WDJB-MJHT",
            expires_in=900,
            interval=1,
        ),
    )
    monkeypatch.setattr(
        oauth,
        "poll_for_token",
        lambda client_id, device_code, *, interval, expires_in: "gho_logged_in_xxx",
    )
    # Prevent the browser from opening during tests.
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda *_a, **_kw: True)
    # Identity probe uses GitHubClient — swap the binding `auth_login` imported.
    import github_twin.cli as cli_mod

    monkeypatch.setattr(cli_mod, "GitHubClient", _FakeGitHubClient)
    return {}


# ---------- login ----------


def test_login_persists_token_and_prints_storage_kind(
    cli_env: Path, fake_oauth: dict[str, object]
) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["auth", "login", "--no-browser"])
    assert result.exit_code == 0, result.output
    assert "WDJB-MJHT" in result.output  # user code shown
    assert "stored" in result.output
    assert "alice" in result.output

    # Token is now retrievable via the storage layer.
    assert auth_storage.load_token(data_dir=cli_env / "data") == "gho_logged_in_xxx"


# ---------- status ----------


def test_status_shows_env_var_when_only_source(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_env_only_value_xxx")
    # Make `gh` lookup fail deterministically (shutil is imported inside
    # the command body, so patching the module-global is sufficient).
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _x: None)

    runner = CliRunner()
    result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 0, result.output
    assert "GITHUB_TOKEN env var" in result.output
    assert "Active:" in result.output


def test_status_after_login_shows_persisted(cli_env: Path, fake_oauth: dict[str, object]) -> None:
    runner = CliRunner()
    runner.invoke(app, ["auth", "login", "--no-browser"])
    result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 0, result.output
    assert "persisted" in result.output
    assert "alice" in result.output


# ---------- logout ----------


def test_logout_removes_persisted_token(cli_env: Path, fake_oauth: dict[str, object]) -> None:
    runner = CliRunner()
    runner.invoke(app, ["auth", "login", "--no-browser"])
    assert auth_storage.load_token(data_dir=cli_env / "data") == "gho_logged_in_xxx"

    result = runner.invoke(app, ["auth", "logout"])
    assert result.exit_code == 0, result.output
    assert "removed" in result.output
    assert auth_storage.load_token(data_dir=cli_env / "data") is None


def test_logout_when_nothing_persisted_is_a_noop(cli_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["auth", "logout"])
    assert result.exit_code == 0
    assert "No persisted token" in result.output


# ---------- login error surface ----------


def test_login_surfaces_oauth_error(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oauth,
        "request_device_code",
        lambda *a, **kw: oauth.DeviceCodeResponse(
            "DC", "CODE-0000", "https://github.com/login/device", None, 900, 1
        ),
    )

    def boom(*_a: object, **_kw: object) -> str:
        raise oauth.OAuthError("Authorization denied in browser.")

    monkeypatch.setattr(oauth, "poll_for_token", boom)

    runner = CliRunner()
    result = runner.invoke(app, ["auth", "login", "--no-browser"])
    assert result.exit_code == 1
    assert "denied" in result.output
