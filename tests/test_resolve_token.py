"""Token-resolution precedence in `_resolve_token`.

Three sources, in order: persisted device-flow token → `gh auth token`
→ GITHUB_TOKEN env. Earlier sources mask later ones.
"""

from __future__ import annotations

import pytest

from github_twin.ingest import auth_storage, github_client


def _patch_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stored: str | None,
    gh_cli: str | None,
    env: str | None,
) -> None:
    monkeypatch.setattr(auth_storage, "load_token", lambda **_kw: stored)
    monkeypatch.setattr(github_client, "_gh_cli_token", lambda: gh_cli)
    if env is None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    else:
        monkeypatch.setenv("GITHUB_TOKEN", env)


def test_persisted_token_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(monkeypatch, stored="stored_tok_xxxx", gh_cli="gh_tok", env="env_tok")
    assert github_client._resolve_token() == "stored_tok_xxxx"


def test_gh_cli_wins_when_no_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(monkeypatch, stored=None, gh_cli="gh_tok_xxxx", env="env_tok_xxxx")
    assert github_client._resolve_token() == "gh_tok_xxxx"


def test_env_used_when_no_persisted_and_no_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(monkeypatch, stored=None, gh_cli=None, env="env_only_xxxx")
    assert github_client._resolve_token() == "env_only_xxxx"


def test_raises_clean_error_when_nothing_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(monkeypatch, stored=None, gh_cli=None, env=None)
    with pytest.raises(github_client.GitHubError, match="gt auth login"):
        github_client._resolve_token()
