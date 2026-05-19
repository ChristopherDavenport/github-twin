"""Thin text-completion wrapper for the eval harness.

The distill `RuleSynthesizer` is structured around a JSON-rule contract that
doesn't fit free-form completions. This module exposes a smaller surface —
`TextLLM.complete(prompt) -> str` — so eval pipelines can prompt the same
underlying models (Claude / Gemini / Ollama) for arbitrary text.

Backend selection mirrors `distill/synth.py:make_synthesizer`: Claude when
`ANTHROPIC_API_KEY` is set, else Gemini when any Gemini auth path is configured
(`GEMINI_API_KEY` / `GOOGLE_API_KEY`, or `GT_GEMINI_PROJECT` + ADC via
`gcloud auth application-default login`), else Ollama. Explicit `prefer=`
overrides.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class TextLLM(Protocol):
    backend_id: str

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str: ...


class ClaudeText:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        import anthropic

        self.model = model
        self.backend_id = f"claude:{model}"
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        from anthropic.types import TextBlock

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # `resp.content` is a union of TextBlock / ThinkingBlock / various
        # tool-use blocks; only TextBlock has `.text`. isinstance narrows
        # the type for both runtime safety and mypy.
        return "".join(b.text for b in resp.content if isinstance(b, TextBlock))


class GeminiText:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        from github_twin.gemini_client import make_gemini_client

        self.model = model
        self.backend_id = f"gemini:{model}"
        self._client = make_gemini_client(api_key)

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        from google.genai import types

        from github_twin.gemini_client import make_thinking_config, with_retry

        resp = with_retry(
            self._client.models.generate_content,
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
                max_output_tokens=max_tokens,
                thinking_config=make_thinking_config(self.model),
            ),
        )
        return resp.text or ""


class OllamaText:
    def __init__(self, *, model: str, host: str | None = None) -> None:
        import ollama

        self.model = model
        self.backend_id = f"ollama:{model}"
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.2, "num_predict": max_tokens},
        )
        # ollama-python returns a typed Mapping but mypy can't see it via
        # `ignore_missing_imports`. The shape is stable; cast to str.
        content: str = resp["message"]["content"]
        return content


def make_text_llm(
    *,
    claude_model: str = "claude-sonnet-4-6",
    gemini_model: str = "gemini-2.5-flash",
    ollama_model: str = "llama3.2",
    prefer: str = "auto",
) -> TextLLM:
    """Mirror of `make_synthesizer`. Same precedence: Claude > Gemini > Ollama."""
    from github_twin.gemini_client import has_gemini_auth

    if prefer == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return ClaudeText(model=claude_model)
    if prefer == "gemini":
        if not has_gemini_auth():
            raise RuntimeError(
                "Gemini requested but no auth available — set GEMINI_API_KEY or "
                "GOOGLE_API_KEY for the API path, or set GT_GEMINI_PROJECT "
                "(and optionally GT_GEMINI_LOCATION) after running "
                "'gcloud auth application-default login' to use Vertex AI."
            )
        return GeminiText(model=gemini_model)
    if prefer == "ollama":
        return OllamaText(model=ollama_model)
    # auto
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeText(model=claude_model)
    if has_gemini_auth():
        return GeminiText(model=gemini_model)
    return OllamaText(model=ollama_model)
