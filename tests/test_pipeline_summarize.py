"""run_summarize: backend-aware concurrency default resolution.

The resolution rule (`_resolve_summarize_concurrency`) maps
`cfg.summarize.concurrency=None` to a per-backend default keyed on the
prefix of `llm.backend_id`: 8 for gemini, 4 for claude, 1 for ollama,
fallback 1. An explicit int in config always wins.
"""

from __future__ import annotations

import pytest

from github_twin.pipeline import _DEFAULT_CONCURRENCY, _resolve_summarize_concurrency


@pytest.mark.parametrize(
    "backend_id,expected",
    [
        ("gemini:gemini-2.5-flash", 4),
        ("claude:claude-sonnet-4-6", 4),
        ("ollama:llama3.2", 1),
        ("fake", 1),
        ("openai:gpt-4o", 1),
    ],
)
def test_unset_resolves_to_backend_default(backend_id: str, expected: int):
    assert _resolve_summarize_concurrency(None, backend_id) == expected


@pytest.mark.parametrize("backend_id", ["gemini:foo", "claude:foo", "ollama:foo", "fake"])
def test_explicit_value_wins_regardless_of_backend(backend_id: str):
    assert _resolve_summarize_concurrency(2, backend_id) == 2
    assert _resolve_summarize_concurrency(16, backend_id) == 16


def test_default_table_includes_expected_backends():
    """Pin the contract: any change to the default map should be deliberate
    and visible in this test. Gemini is held at 4 (not 8) so a free-tier
    key doesn't drown in 429s on the first `gt sync`."""
    assert _DEFAULT_CONCURRENCY == {"gemini": 4, "claude": 4, "ollama": 1}
