"""Unit tests for the `gt feedback` payload assembly.

The CLI command wires this together with a live `sqlite3.Connection` and
`webbrowser.open` — those paths aren't exercised here. We test the pure
data + string assembly: URL building, body rendering, and secret
scrubbing."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from github_twin import feedback


def _env() -> feedback.EnvInfo:
    return feedback.EnvInfo(
        gt_version="0.0.4",
        python_version="3.12.5",
        platform="Linux-6.6-WSL2",
    )


def _snap(kind: str = "user", name: str = "me") -> feedback.TargetSnapshot:
    return feedback.TargetSnapshot(
        kind=kind,
        name=name,
        artifacts={"commit": 12, "review_comment": 3},
        chunks={"code": 30, "review_comment": 3},
        vectors=33,
        pending_embed=0,
    )


# ---------- build_discussion_url ----------


def test_build_discussion_url_body_encoded():
    url = feedback.build_discussion_url(body="hello world & friends", title="t")
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    assert parsed.hostname == "github.com"
    assert parsed.path == "/ChristopherDavenport/github-twin/discussions/new"
    assert qs["body"] == ["hello world & friends"]
    assert qs["title"] == ["t"]


def test_build_discussion_url_omits_optional_fields():
    url = feedback.build_discussion_url(body="x")
    qs = parse_qs(urlparse(url).query)
    assert "title" not in qs
    assert "category" not in qs


def test_build_discussion_url_includes_category():
    url = feedback.build_discussion_url(body="x", category="general")
    qs = parse_qs(urlparse(url).query)
    assert qs["category"] == ["general"]


def test_build_discussion_url_respects_repo_override():
    url = feedback.build_discussion_url(body="x", repo="someorg/somerepo")
    assert urlparse(url).path == "/someorg/somerepo/discussions/new"


# ---------- render_body ----------


def test_render_body_includes_all_sections():
    body = feedback.render_body(
        env=_env(),
        embed_summary="ollama / nomic-embed-text (dim=768)",
        corpus=[_snap()],
        user_note="retrieval is great",
    )
    assert "## Feedback" in body
    assert "retrieval is great" in body
    assert "## Environment" in body
    assert "0.0.4" in body
    assert "3.12.5" in body
    assert "ollama / nomic-embed-text" in body
    assert "## Corpus" in body
    assert "user `me`" in body
    assert "15 artifacts" in body  # 12 + 3
    assert "33 chunks" in body  # 30 + 3
    assert "33 vectors" in body


def test_render_body_handles_empty_note():
    body = feedback.render_body(
        env=_env(),
        embed_summary="ollama / m (dim=4)",
        corpus=[_snap()],
        user_note="   ",
    )
    assert "_(no free-text note provided)_" in body


def test_render_body_handles_no_targets():
    body = feedback.render_body(
        env=_env(),
        embed_summary="ollama / m (dim=4)",
        corpus=[],
        user_note="hi",
    )
    assert "_(no targets" in body


# ---------- scrub_secrets ----------


def test_scrub_secrets_redacts_env_token(monkeypatch: pytest.MonkeyPatch):
    token = "ghp_secrettoken_xxxxxxxxxxxx"
    monkeypatch.setenv("GITHUB_TOKEN", token)

    note = f"my token leaked: {token} oops"
    body = feedback.render_body(
        env=_env(),
        embed_summary="ollama / m (dim=4)",
        corpus=[_snap()],
        user_note=note,
    )
    scrubbed = feedback.scrub_secrets(body)
    assert token not in scrubbed
    assert "REDACTED" in scrubbed


def test_scrub_secrets_redacts_bearer_pattern():
    text = "trace: Authorization: Bearer abc.def-ghi_jklmnop"
    scrubbed = feedback.scrub_secrets(text)
    assert "abc.def-ghi_jklmnop" not in scrubbed
    assert "Bearer ***" in scrubbed


def test_scrub_secrets_is_noop_without_secrets():
    text = "everything fine here\nno tokens\n"
    assert feedback.scrub_secrets(text) == text


# ---------- _embed_summary ----------


def test_embed_summary_renders_fields():
    class Stub:
        backend = "ollama"
        model = "nomic-embed-text"
        dim = 768

    assert feedback._embed_summary(Stub()) == "ollama / nomic-embed-text (dim=768)"


def test_embed_summary_tolerates_missing_attrs():
    class Stub:
        pass

    out = feedback._embed_summary(Stub())
    assert "?" in out
