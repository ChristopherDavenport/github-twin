"""Tests for `make_text_llm` factory precedence.

Mirrors the synthesizer factory tests in `test_distill.py`. The TextLLM
seam is what `process/summarize.py` and the eval harness both ride on,
so its auth-detection logic is load-bearing for three CLI commands.
"""

from __future__ import annotations

from typing import Any

import pytest

from github_twin.eval.llm import (
    ClaudeText,
    GeminiText,
    OllamaText,
    make_text_llm,
)


def test_make_text_llm_picks_claude_when_only_anthropic_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GT_GEMINI_PROJECT", raising=False)
    llm = make_text_llm(claude_model="claude-test")
    assert isinstance(llm, ClaudeText)


def test_make_text_llm_picks_gemini_when_only_api_key_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    monkeypatch.delenv("GT_GEMINI_PROJECT", raising=False)
    llm = make_text_llm(gemini_model="gemini-test")
    assert isinstance(llm, GeminiText)
    assert llm.backend_id == "gemini:gemini-test"


def test_make_text_llm_picks_gemini_when_only_adc_project_set(monkeypatch):
    """ADC fallback parity with the synthesizer factory."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GT_GEMINI_PROJECT", "my-proj")
    from google import genai

    monkeypatch.setattr(genai, "Client", lambda **kw: object())
    llm = make_text_llm(gemini_model="gemini-test")
    assert isinstance(llm, GeminiText)
    assert llm.backend_id == "gemini:gemini-test"


def test_make_text_llm_falls_back_to_ollama_when_no_auth(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GT_GEMINI_PROJECT", raising=False)
    llm = make_text_llm(ollama_model="llama-test")
    assert isinstance(llm, OllamaText)
    assert llm.backend_id == "ollama:llama-test"


def test_make_text_llm_forced_gemini_errors_without_auth(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GT_GEMINI_PROJECT", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        make_text_llm(prefer="gemini")
    msg = str(exc_info.value)
    assert "GEMINI_API_KEY" in msg
    assert "GT_GEMINI_PROJECT" in msg


def test_make_text_llm_claude_beats_gemini_in_auto(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    llm = make_text_llm()
    assert isinstance(llm, ClaudeText)


# --- helper construction: precedence inside make_gemini_client ---------------


def _capture_genai_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def _factory(**kwargs: Any) -> object:
        seen.append(kwargs)
        return object()

    from google import genai

    monkeypatch.setattr(genai, "Client", _factory)
    return seen


def test_make_gemini_client_api_key_wins_over_project(monkeypatch):
    monkeypatch.setenv("GT_GEMINI_PROJECT", "fake-proj")
    seen = _capture_genai_kwargs(monkeypatch)
    from github_twin.gemini_client import make_gemini_client

    make_gemini_client("real-key")
    assert seen == [{"api_key": "real-key"}]


def test_make_gemini_client_vertex_path_when_only_project(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GT_GEMINI_PROJECT", "my-proj")
    monkeypatch.delenv("GT_GEMINI_LOCATION", raising=False)
    seen = _capture_genai_kwargs(monkeypatch)
    from github_twin.gemini_client import make_gemini_client

    make_gemini_client()
    assert seen == [
        {"vertexai": True, "project": "my-proj", "location": "us-central1"},
    ]


def test_make_gemini_client_vertex_honors_custom_location(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GT_GEMINI_PROJECT", "my-proj")
    monkeypatch.setenv("GT_GEMINI_LOCATION", "europe-west4")
    seen = _capture_genai_kwargs(monkeypatch)
    from github_twin.gemini_client import make_gemini_client

    make_gemini_client()
    assert seen == [
        {"vertexai": True, "project": "my-proj", "location": "europe-west4"},
    ]


def test_make_gemini_client_no_auth_calls_bare_client(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GT_GEMINI_PROJECT", raising=False)
    monkeypatch.delenv("GT_GEMINI_LOCATION", raising=False)
    seen = _capture_genai_kwargs(monkeypatch)
    from github_twin.gemini_client import make_gemini_client

    make_gemini_client()
    assert seen == [{}]


def test_has_gemini_auth_detects_all_three_signals(monkeypatch):
    from github_twin.gemini_client import has_gemini_auth

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GT_GEMINI_PROJECT", raising=False)
    assert has_gemini_auth() is False

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert has_gemini_auth() is True
    monkeypatch.delenv("GEMINI_API_KEY")

    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert has_gemini_auth() is True
    monkeypatch.delenv("GOOGLE_API_KEY")

    monkeypatch.setenv("GT_GEMINI_PROJECT", "my-proj")
    assert has_gemini_auth() is True
