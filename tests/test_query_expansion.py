"""Asymmetric query expansion for the BM25 leg.

Three concerns covered here:
  1. RuleExpander produces the expected OR-groups for code-shaped input
     (case variants, camelCase splits, hand-curated synonyms).
  2. The FTS5 builder turns those groups into a valid MATCH expression
     that SQLite's FTS5 will actually parse.
  3. End-to-end BM25 path: with an expander, a token's alternates surface
     chunks that would have missed under the bare-token query — without
     pulling vector results sideways.

The full "expansion does not touch the vector leg" contract has its own
test in `test_hybrid_search.py` (added there because that's where the
asymmetry lives in code).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.store.query_expansion import (
    CompositeExpander,
    OllamaExpander,
    RuleExpander,
    _ExpansionCache,
    make_expander,
)

# ---------- RuleExpander shapes ----------


def test_rule_expander_returns_one_group_per_token():
    out = RuleExpander().expand("search function")
    assert len(out) == 2
    assert out[0][0] == "search"
    assert out[1][0] == "function"


def test_rule_expander_includes_synonyms():
    groups = RuleExpander().expand("function")
    assert {"func", "fn", "method"}.issubset({w.lower() for w in groups[0]})


def test_rule_expander_splits_camelcase():
    groups = RuleExpander().expand("getUser")
    flat = {w.lower() for w in groups[0]}
    # The whole token survives, and "user" + "get" surface as compounds.
    assert "getuser" in flat
    assert "user" in flat
    assert "get" in flat


def test_rule_expander_splits_snake_case_into_parts():
    """Snake-case identifier survives as one token (underscore is alnum
    in the tokenizer); compound splitter expands it to its parts."""
    parts = {w.lower() for w in RuleExpander().expand("get_user_id")[0]}
    assert "user" in parts and "id" in parts and "get" in parts


def test_rule_expander_treats_kebab_case_as_separate_tokens():
    """Dashes aren't part of identifier syntax in the tokenizer; the
    user's `get-user-id` query yields three independent OR-groups."""
    groups = RuleExpander().expand("get-user-id")
    assert [g[0] for g in groups] == ["get", "user", "id"]


def test_rule_expander_strips_punctuation():
    # Periods, parens, etc. don't survive tokenization, but the alnum parts do.
    groups = RuleExpander().expand("Tracer.translate()")
    flat = [g[0] for g in groups]
    assert flat == ["Tracer", "translate"]


def test_rule_expander_empty_input_is_empty():
    assert RuleExpander().expand("") == []
    assert RuleExpander().expand("   ") == []


def test_rule_expander_no_duplicate_alternates():
    """Synonyms that overlap with case variants shouldn't double-add."""
    groups = RuleExpander().expand("Find")
    lowered = [w.lower() for w in groups[0]]
    assert len(lowered) == len(set(lowered))
    # Original term is present at index 0.
    assert groups[0][0] == "Find"


# ---------- FTS5 MATCH builder ----------


def test_fts_match_builder_single_token_no_or():
    expr = q._fts_match_from_groups([["search"]])
    assert expr == '"search"'


def test_fts_match_builder_or_group_wraps_in_parens():
    expr = q._fts_match_from_groups([["search", "find", "lookup"]])
    assert expr == '("search" OR "find" OR "lookup")'


def test_fts_match_builder_multi_token_joins_with_explicit_and():
    """Explicit `AND` is required when any group is parenthesized — FTS5
    refuses to chain a phrase next to a `(...)` group with implicit-AND."""
    expr = q._fts_match_from_groups([["search", "find"], ["function", "func"]])
    assert expr == '("search" OR "find") AND ("function" OR "func")'


def test_fts_match_builder_dedupes_case_insensitively():
    expr = q._fts_match_from_groups([["Find", "find", "FIND"]])
    # Only the first survives (case-insensitive dedup keeps the first).
    assert expr == '"Find"'


def test_fts_match_builder_empty_returns_safe_match():
    assert q._fts_match_from_groups([]) == '""'
    assert q._fts_match_from_groups([[]]) == '""'


def test_fts_match_builder_escapes_embedded_quotes():
    """FTS5 doubles internal double-quotes per its quoting rule."""
    expr = q._fts_match_from_groups([['he said "hi"']])
    assert expr == '"he said ""hi"""'


# ---------- End-to-end BM25 with expander ----------


class _StubExpander:
    """Returns a fixed expansion regardless of input — keeps the test
    pinned to the expander -> BM25 wiring, not any specific rule."""

    backend_id = "stub"

    def __init__(self, groups: list[list[str]]) -> None:
        self._groups = groups

    def expand(self, _: str) -> list[list[str]]:
        return self._groups


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "expansion.sqlite", embed_dim=4)
    yield db
    db.close()


def _seed_chunk(conn, *, text: str, language: str = "python") -> int:
    aid = q.upsert_artifact(
        conn,
        kind="commit",
        external_id=f"e-{text}",
        source_url=None,
        repo="me/x",
        language=language,
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    return q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="code",
        text=text,
        context={"language": language},
        language=language,
    )


def test_bm25_search_uses_expander_when_provided(conn):
    """Query `func` alone misses a chunk written with `function`; with
    an expander that turns `func` into the OR-group `{func, function}`
    the chunk surfaces."""
    cid = _seed_chunk(conn, text="def function(): return 1")
    # Without expansion: AND-of-tokens doesn't include "function" in the query.
    bare = q.bm25_search(conn, query_text="func", chunk_kind="code", k=5)
    assert all(h.chunk_id != cid for h in bare)
    # With expansion: the OR-group now includes "function".
    expander = _StubExpander([["func", "function"]])
    expanded = q.bm25_search(conn, query_text="func", chunk_kind="code", k=5, expander=expander)
    assert cid in [h.chunk_id for h in expanded]


def test_bm25_search_without_expander_uses_fts_escape(conn):
    """Confirm the no-expander path matches the legacy `_fts_escape`
    AND-of-tokens behavior — a query of "func function" requires both
    tokens to appear."""
    cid_both = _seed_chunk(conn, text="def function func helper")
    _seed_chunk(conn, text="just function alone")
    hits = q.bm25_search(conn, query_text="func function", chunk_kind="code", k=5)
    # Only the chunk containing both tokens is returned (AND semantics).
    assert [h.chunk_id for h in hits] == [cid_both]


# ---------- CompositeExpander ----------


def test_composite_unions_alternates_and_dedupes():
    a = _StubExpander([["search", "find"]])
    b = _StubExpander([["search", "lookup", "find"]])
    out = CompositeExpander(a, b).expand("search")
    assert out[0][0] == "search"
    assert sorted(out[0][1:]) == ["find", "lookup"]


def test_composite_returns_empty_when_no_expander_yields():
    a = _StubExpander([])
    b = _StubExpander([])
    assert CompositeExpander(a, b).expand("anything") == []


# ---------- OllamaExpander cache ----------


def test_expansion_cache_round_trips(tmp_path: Path):
    cache = _ExpansionCache(tmp_path / "qe-cache.sqlite")
    assert cache.get("model-a", "search") is None
    cache.put("model-a", "search", ["find", "lookup"])
    assert cache.get("model-a", "search") == ["find", "lookup"]
    # Different model = different key.
    assert cache.get("model-b", "search") is None


def test_expansion_cache_is_case_insensitive(tmp_path: Path):
    cache = _ExpansionCache(tmp_path / "qe-cache.sqlite")
    cache.put("m", "Search", ["find"])
    assert cache.get("m", "search") == ["find"]
    assert cache.get("m", "SEARCH") == ["find"]


def test_ollama_expander_consults_cache_first_no_network(tmp_path: Path):
    """If the cache has hits for all tokens, the expander never reaches
    Ollama. Verified by pre-populating and asserting `_fetch` isn't
    invoked (would raise on a missing ollama install)."""
    cache_path = tmp_path / "qe.sqlite"
    cache = _ExpansionCache(cache_path)
    cache.put("qwen3:0.6b", "search", ["find", "lookup"])
    cache.put("qwen3:0.6b", "function", ["func", "method"])

    exp = OllamaExpander(model="qwen3:0.6b", cache_path=cache_path)

    def _no_fetch(_):
        raise AssertionError("cache hit path should not call _fetch")

    exp._fetch = _no_fetch  # type: ignore[method-assign]
    out = exp.expand("search function")
    assert out == [["search", "find", "lookup"], ["function", "func", "method"]]


def test_ollama_expander_writes_to_cache_after_fetch(tmp_path: Path):
    """After a successful fetch, subsequent calls hit the cache."""
    cache_path = tmp_path / "qe.sqlite"
    exp = OllamaExpander(model="qwen3:0.6b", cache_path=cache_path)
    calls = {"count": 0}

    def _fake_fetch(tokens):
        calls["count"] += 1
        return {tok: [tok + "_alt"] for tok in tokens}

    exp._fetch = _fake_fetch  # type: ignore[method-assign]
    first = exp.expand("alpha beta")
    assert first == [["alpha", "alpha_alt"], ["beta", "beta_alt"]]
    assert calls["count"] == 1
    # Second call hits cache for all tokens.
    second = exp.expand("alpha beta")
    assert second == first
    assert calls["count"] == 1


def test_ollama_expander_degrades_on_fetch_failure(tmp_path: Path):
    """When Ollama is down, expander returns the bare tokens (no alts)
    rather than raising — keeps the BM25 path running."""
    exp = OllamaExpander(model="qwen3:0.6b", cache_path=tmp_path / "qe.sqlite")

    def _boom(_):
        raise RuntimeError("ollama is down")

    exp._fetch = _boom  # type: ignore[method-assign]
    out = exp.expand("search")
    assert out == [["search"]]


# ---------- factory ----------


def test_make_expander_off_returns_none():
    assert make_expander("off") is None


def test_make_expander_rule_returns_rule_expander():
    exp = make_expander("rule")
    assert exp is not None
    assert exp.backend_id == "rule"


def test_make_expander_ollama_returns_composite():
    """`ollama` backend wraps Rule + Ollama in a CompositeExpander so
    deterministic wins survive even if the LLM call fails."""
    exp = make_expander("ollama", cache_path=Path("/tmp/qe-test.sqlite"))
    assert exp is not None
    assert exp.backend_id == "composite"
    # Rule layer still produces alternates without the LLM.
    out = exp.expand("function")
    assert "func" in {w.lower() for w in out[0]}


def test_make_expander_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown"):
        make_expander("magic")


# ---------- parse robustness ----------


def test_ollama_parses_partial_json():
    """When the LLM omits some tokens, the parser returns what it can
    and the expander fills in empty alternates for the rest."""
    from github_twin.store.query_expansion import _parse_ollama_json

    raw = json.dumps({"search": ["find", "lookup"], "extra_garbage": "ignored"})
    out = _parse_ollama_json(raw, ["search", "function"])
    assert out["search"] == ["find", "lookup"]
    assert "function" not in out  # parser doesn't fabricate keys


def test_ollama_parses_malformed_json_to_empty():
    from github_twin.store.query_expansion import _parse_ollama_json

    assert _parse_ollama_json("not json at all", ["a", "b"]) == {}
    assert _parse_ollama_json("[1, 2, 3]", ["a"]) == {}  # not a dict
