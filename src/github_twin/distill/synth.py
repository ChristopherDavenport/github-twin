"""Rule synthesis: turn a cluster of review comments into a one-sentence rule.

Pluggable backend so we can swap Claude for a local model. Defaults to Claude
when ANTHROPIC_API_KEY is set, falls back to Ollama otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are analyzing code review comments left by a single
reviewer on GitHub pull requests. Your job is to surface the underlying *rule*
or *preference* that recurs across a cluster of related comments — the kind of
guidance the reviewer would give again on the next PR that triggers it.

Each cluster contains several comments. Each comment carries:
- the comment body the reviewer wrote
- the diff hunk it was attached to (the code being reviewed), when available
- the repo and PR title

Rules:
- One sentence, present tense, second-person ("Prefer X over Y", "Add Z when ...").
- Concrete and actionable; avoid vague platitudes.
- If the cluster is incoherent (the comments don't share a real pattern), say so honestly via `incoherent: true`.

Return JSON only, no prose around it:
{
  "rule": "<one-sentence rule>",
  "language": "<dominant language tag if obvious, else null>",
  "example_quotes": ["<short quote 1>", "<short quote 2>"],
  "incoherent": false
}
""".strip()

CODE_SYSTEM_PROMPT = """You are analyzing a cluster of code snippets that an
author has repeatedly written across their commit history. Your job is to
articulate the *pattern* that recurs — the kind of guidance you'd give an
agent writing new code in this codebase so the new code matches the
established style.

Each cluster contains several diff-added code blocks. Each block carries:
- the code text (only added lines from a real diff)
- the file path and language
- the repo

Rules:
- One sentence, present tense, second-person, prescriptive
  ("Prefer X over Y", "Use Z for ...", "Return None on cache miss; raise on
  programmer error").
- Concrete and actionable. Anchor in language- or library-specific identifiers
  where the cluster justifies it ("Use `contextlib.suppress(OSError)` for
  cleanup that may race"). Avoid vague platitudes like "write clean code".
- If the snippets don't share a real pattern (just superficial token overlap),
  say so honestly via `incoherent: true`.
- Capture *positive* patterns — what the author does, not what they avoid.

Return JSON only, no prose around it:
{
  "rule": "<one-sentence rule>",
  "language": "<dominant language tag if obvious, else null>",
  "example_quotes": ["<short code excerpt 1>", "<short code excerpt 2>"],
  "incoherent": false
}
""".strip()


@dataclass
class RuleResult:
    rule: str
    language: str | None
    example_quotes: list[str]
    incoherent: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleResult:
        return cls(
            rule=str(data.get("rule", "")).strip(),
            language=(data.get("language") or None),
            example_quotes=[str(q) for q in (data.get("example_quotes") or [])][:3],
            incoherent=bool(data.get("incoherent", False)),
        )


@runtime_checkable
class RuleSynthesizer(Protocol):
    """Anything that turns a cluster of comments into a rule."""

    backend_id: str

    def synthesize(self, cluster: list[dict[str, Any]]) -> RuleResult: ...


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _parse_json_response(text: str) -> RuleResult:
    """Pull the first JSON object out of a free-text response."""
    m = _JSON_BLOCK.search(text)
    if not m:
        raise ValueError(f"No JSON object in response: {text[:200]!r}")
    data = json.loads(m.group(0))
    return RuleResult.from_dict(data)


def _render_cluster_for_prompt(cluster: list[dict[str, Any]]) -> str:
    """Render a cluster for the LLM. Dispatches on `member_kind`.

    Code clusters carry path/language/source_url instead of pr_title/diff_hunk.
    The orchestrator stamps `member_kind` based on the source chunk_kind.
    """
    lines: list[str] = []
    for i, c in enumerate(cluster, 1):
        if c.get("member_kind") == "code":
            lines.append(f"--- snippet #{i} ---")
            lines.append(f"repo: {c.get('repo', '?')}")
            lines.append(f"path: {c.get('path', '?')}")
            lang = c.get("language")
            if lang:
                lines.append(f"language: {lang}")
            lines.append("code:")
            text = c.get("text", "")
            trimmed = "\n".join(text.splitlines()[:40])
            lines.append(trimmed)
            lines.append("")
            continue
        lines.append(f"--- comment #{i} ---")
        lines.append(f"repo: {c.get('repo', '?')}")
        lines.append(f"pr_title: {c.get('pr_title', '?')}")
        lang = c.get("language")
        if lang:
            lines.append(f"language: {lang}")
        diff = c.get("diff_hunk")
        if diff:
            trimmed = "\n".join(diff.splitlines()[:20])
            lines.append("diff_hunk:")
            lines.append(trimmed)
        lines.append("comment:")
        lines.append(c.get("text", ""))
        lines.append("")
    return "\n".join(lines)


# ---------- Claude impl ----------


class ClaudeSynthesizer:
    """Claude API synthesizer with prompt caching on the system prompt.

    The system prompt is identical across every cluster, so caching it cuts
    input cost ~10x for the typical 20-50 cluster run.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        import anthropic

        self.model = model
        self.backend_id = f"claude:{model}"
        self.system_prompt = system_prompt
        self._client = anthropic.Anthropic(api_key=api_key)

    def synthesize(self, cluster: list[dict[str, Any]]) -> RuleResult:
        user_msg = _render_cluster_for_prompt(cluster)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_msg}],
        )
        from anthropic.types import TextBlock

        # Same union narrowing as eval/llm.py — only TextBlock has `.text`.
        text = "".join(b.text for b in resp.content if isinstance(b, TextBlock))
        return _parse_json_response(text)


# ---------- Gemini impl ----------


class GeminiSynthesizer:
    """Gemini API synthesizer (Google AI Studio).

    Uses `response_mime_type=application/json` for structured output, which is
    a cleaner contract than the free-text-JSON-parse the other backends use,
    but we still pipe through `_parse_json_response` for parity.

    Gemini 2.5 context caching exists but has a minimum-size threshold the
    system prompt doesn't meet, so we skip it here.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        from github_twin.gemini_client import make_gemini_client

        self.model = model
        self.backend_id = f"gemini:{model}"
        self.system_prompt = system_prompt
        # API key wins if set; otherwise falls back to ADC via GT_GEMINI_PROJECT,
        # else the SDK's own env auto-config. See gemini_client.make_gemini_client.
        self._client = make_gemini_client(api_key)

    def synthesize(self, cluster: list[dict[str, Any]]) -> RuleResult:
        from google.genai import types

        from github_twin.gemini_client import make_thinking_config, with_retry

        user_msg = _render_cluster_for_prompt(cluster)
        resp = with_retry(
            self._client.models.generate_content,
            model=self.model,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=512,
                thinking_config=make_thinking_config(self.model),
            ),
        )
        text = resp.text or ""
        return _parse_json_response(text)


# ---------- Ollama impl ----------


class OllamaSynthesizer:
    """Local chat-model synthesizer. Quality depends heavily on the model size."""

    def __init__(
        self,
        *,
        model: str,
        host: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        import ollama

        self.model = model
        self.backend_id = f"ollama:{model}"
        self.system_prompt = system_prompt
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def synthesize(self, cluster: list[dict[str, Any]]) -> RuleResult:
        user_msg = _render_cluster_for_prompt(cluster)
        resp = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            options={"temperature": 0.2},
        )
        text = resp["message"]["content"]
        return _parse_json_response(text)


# ---------- Factory ----------


def make_synthesizer(
    *,
    claude_model: str,
    gemini_model: str = "gemini-3.1-flash-lite",
    ollama_model: str | None = None,
    prefer: str = "auto",
    system_prompt: str = SYSTEM_PROMPT,
) -> RuleSynthesizer:
    """Pick a backend.

    `prefer`:
        - 'auto' (default): Claude if ANTHROPIC_API_KEY is set, else Gemini if
          GEMINI_API_KEY/GOOGLE_API_KEY is set, else Ollama fallback.
        - 'claude' / 'gemini' / 'ollama': force that backend.

    `system_prompt` selects the synthesis flavor — defaults to the review-comment
    prompt; pass `CODE_SYSTEM_PROMPT` for code-pattern rules.
    """
    from github_twin.gemini_client import has_gemini_auth

    prefer = (prefer or "auto").lower()
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_gemini = has_gemini_auth()

    if prefer == "claude":
        if not has_claude:
            raise RuntimeError("Claude requested but ANTHROPIC_API_KEY is not set.")
        log.info("using Claude synthesizer: %s", claude_model)
        return ClaudeSynthesizer(model=claude_model, system_prompt=system_prompt)

    if prefer == "gemini":
        if not has_gemini:
            raise RuntimeError(
                "Gemini requested but no auth available — set GEMINI_API_KEY or "
                "GOOGLE_API_KEY for the API path, or set GT_GEMINI_PROJECT "
                "(and optionally GT_GEMINI_LOCATION) after running "
                "'gcloud auth application-default login' to use Vertex AI."
            )
        log.info("using Gemini synthesizer: %s", gemini_model)
        return GeminiSynthesizer(model=gemini_model, system_prompt=system_prompt)

    if prefer == "ollama":
        model = ollama_model or "llama3.2"
        log.info("using Ollama synthesizer: %s (forced)", model)
        return OllamaSynthesizer(model=model, system_prompt=system_prompt)

    if prefer != "auto":
        raise ValueError(f"Unknown synthesizer backend: {prefer!r}")

    # auto: prefer hosted models in quality order, fall back to local
    if has_claude:
        log.info("using Claude synthesizer: %s", claude_model)
        return ClaudeSynthesizer(model=claude_model, system_prompt=system_prompt)
    if has_gemini:
        log.info("using Gemini synthesizer: %s", gemini_model)
        return GeminiSynthesizer(model=gemini_model, system_prompt=system_prompt)
    model = ollama_model or "llama3.2"
    log.info(
        "using Ollama synthesizer: %s (no ANTHROPIC_API_KEY / Gemini auth found)",
        model,
    )
    return OllamaSynthesizer(model=model, system_prompt=system_prompt)
