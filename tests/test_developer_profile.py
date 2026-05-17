"""`developer_profile` MCP tool: synthesis + sample-hash cache.

Uses a `FakeLLM` so the LLM dispatch path stays exercised without
hitting Claude / Gemini / Ollama. The interesting properties are
cache invalidation when a new comment lands and the
explicit-override-wins rule on the cache key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from github_twin.distill.profile import sample_hash, synthesize_profile
from github_twin.mcp_server.tools import _DEFAULT_PROFILE_LOGIN, developer_profile
from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target


@dataclass
class FakeLLM:
    backend_id: str = "fake"
    response: str = "## Voice\n\nBlunt and rigorous; cares about typeclasses."
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        self.calls.append((system, user))
        return self.response


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "profile.sqlite", embed_dim=4)
    seed_target(db)
    yield db
    db.close()


def _seed_review(
    conn,
    *,
    text: str,
    author: str = "alice",
    created_at: str = "2026-05-15T12:00:00+00:00",
    repo: str = "me/x",
    language: str | None = None,
) -> int:
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="review_comment",
        external_id=f"r-{author}-{repo}-{language or 'NL'}-{text[:20]}",
        source_url=None,
        repo=repo,
        language=language,
        author_email=None,
        author_login=author,
        created_at=created_at,
        decision=None,
        meta=None,
    )
    return q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="review_comment",
        text=text,
        context={"repo": repo, "language": language},
        language=language,
    )


# ---------- recent_review_comments ----------


def test_recent_review_comments_orders_by_created_at_desc(conn):
    _seed_review(conn, text="old", author="alice", created_at="2024-01-01T00:00:00+00:00")
    _seed_review(conn, text="new", author="alice", created_at="2026-05-15T00:00:00+00:00")
    _seed_review(conn, text="middle", author="alice", created_at="2025-06-01T00:00:00+00:00")
    rows = q.recent_review_comments(conn, author_login="alice", limit=10)
    assert [r.text for r in rows] == ["new", "middle", "old"]


def test_recent_review_comments_filters_by_author(conn):
    _seed_review(conn, text="alice-1", author="alice")
    _seed_review(conn, text="bob-1", author="bob")
    rows = q.recent_review_comments(conn, author_login="alice", limit=10)
    assert {r.text for r in rows} == {"alice-1"}


def test_recent_review_comments_no_filter_returns_all(conn):
    _seed_review(conn, text="a", author="alice")
    _seed_review(conn, text="b", author="bob")
    rows = q.recent_review_comments(conn, author_login=None, limit=10)
    assert {r.text for r in rows} == {"a", "b"}


# ---------- sample_hash ----------


def test_sample_hash_is_stable_across_orderings(conn):
    _seed_review(conn, text="x")
    _seed_review(conn, text="y")
    rows = q.recent_review_comments(conn, limit=10)
    h1 = sample_hash(rows)
    h2 = sample_hash(list(reversed(rows)))
    assert h1 == h2


def test_sample_hash_changes_when_set_changes(conn):
    _seed_review(conn, text="x")
    h_before = sample_hash(q.recent_review_comments(conn, limit=10))
    _seed_review(conn, text="y")
    h_after = sample_hash(q.recent_review_comments(conn, limit=10))
    assert h_before != h_after


# ---------- synthesize_profile (direct) ----------


def test_synthesize_profile_returns_empty_on_empty_input():
    assert synthesize_profile(FakeLLM(), []) == ""


def test_synthesize_profile_calls_llm_with_recent_comments(conn):
    _seed_review(conn, text="Prefer Async over IO.")
    _seed_review(conn, text="Use Resource for acquire/release.")
    llm = FakeLLM(response="profile body")
    out = synthesize_profile(llm, q.recent_review_comments(conn, limit=10))
    assert out == "profile body"
    assert len(llm.calls) == 1
    system, user = llm.calls[0]
    assert "voice" in system.lower()
    assert "Prefer Async over IO" in user
    assert "Use Resource for acquire/release" in user


# ---------- developer_profile (full tool) ----------


def test_developer_profile_returns_empty_when_no_comments(conn):
    out = developer_profile(conn, FakeLLM(), author_login="ghost")
    assert out["profile_md"] == ""
    assert out["n_samples"] == 0
    assert out["from_cache"] is False
    assert out["generated_at"] is None
    assert out["author_login"] == "ghost"
    # Filter dimensions echo back so callers can confirm what scope ran.
    assert out["language"] is None
    assert out["repo"] is None


def test_developer_profile_caches_after_first_call(conn):
    _seed_review(conn, text="comment 1", author="alice")
    _seed_review(conn, text="comment 2", author="alice")
    llm = FakeLLM(response="alice's profile")

    first = developer_profile(conn, llm, author_login="alice")
    assert first["from_cache"] is False
    assert first["profile_md"] == "alice's profile"
    assert first["n_samples"] == 2
    assert first["generated_at"] is not None
    assert len(llm.calls) == 1

    second = developer_profile(conn, llm, author_login="alice")
    assert second["from_cache"] is True
    assert second["profile_md"] == "alice's profile"
    # LLM was NOT called the second time.
    assert len(llm.calls) == 1


def test_developer_profile_invalidates_on_new_comment(conn):
    _seed_review(conn, text="comment 1", author="alice")
    llm = FakeLLM(response="profile-v1")
    first = developer_profile(conn, llm, author_login="alice")
    assert first["from_cache"] is False

    # Ingest a fresh comment; sample_hash changes.
    _seed_review(
        conn,
        text="comment 2",
        author="alice",
        created_at="2026-06-01T00:00:00+00:00",
    )
    llm.response = "profile-v2"
    second = developer_profile(conn, llm, author_login="alice")
    assert second["from_cache"] is False, "new comment must invalidate cache"
    assert second["profile_md"] == "profile-v2"
    assert second["n_samples"] == 2
    assert len(llm.calls) == 2


def test_developer_profile_force_refresh_bypasses_cache(conn):
    _seed_review(conn, text="c", author="alice")
    llm = FakeLLM(response="v1")
    developer_profile(conn, llm, author_login="alice")
    assert len(llm.calls) == 1

    llm.response = "v2"
    out = developer_profile(conn, llm, author_login="alice", force_refresh=True)
    assert out["from_cache"] is False
    assert out["profile_md"] == "v2"
    assert len(llm.calls) == 2


def test_developer_profile_uses_default_login_when_none(conn):
    """Without `author_login`, the tool profiles the whole corpus and
    caches under the `__target__` sentinel — separate cache key from
    any named author."""
    _seed_review(conn, text="c1", author="alice")
    _seed_review(conn, text="c2", author="bob")
    llm = FakeLLM(response="org-wide profile")

    out = developer_profile(conn, llm, author_login=None)
    assert out["author_login"] is None
    assert out["n_samples"] == 2

    cached = q.get_cached_profile(conn, _DEFAULT_PROFILE_LOGIN)
    assert cached is not None
    assert cached["profile_md"] == "org-wide profile"
    # Per-author cache is distinct.
    assert q.get_cached_profile(conn, "alice") is None


# ---------- recent_review_comments: repo / language filters ----------


def test_recent_review_comments_filters_by_repo(conn):
    _seed_review(conn, text="here", repo="me/x")
    _seed_review(conn, text="there", repo="me/y")
    rows = q.recent_review_comments(conn, repo="me/x", limit=10)
    assert {r.text for r in rows} == {"here"}


def test_recent_review_comments_filters_by_language(conn):
    _seed_review(conn, text="scala-cmt", language="scala")
    _seed_review(conn, text="py-cmt", language="python")
    _seed_review(conn, text="no-lang-cmt", language=None)
    rows = q.recent_review_comments(conn, language="scala", limit=10)
    assert {r.text for r in rows} == {"scala-cmt"}


def test_recent_review_comments_filters_combine(conn):
    _seed_review(conn, text="match", author="alice", repo="me/x", language="scala")
    _seed_review(conn, text="wrong-repo", author="alice", repo="me/y", language="scala")
    _seed_review(conn, text="wrong-lang", author="alice", repo="me/x", language="python")
    _seed_review(conn, text="wrong-author", author="bob", repo="me/x", language="scala")
    rows = q.recent_review_comments(
        conn, author_login="alice", repo="me/x", language="scala", limit=10
    )
    assert {r.text for r in rows} == {"match"}


# ---------- developer_profile: scope-aware filters + cache key isolation ----------


def test_developer_profile_language_filter_narrows_corpus(conn):
    _seed_review(conn, text="scala-1", author="alice", language="scala")
    _seed_review(conn, text="scala-2", author="alice", language="scala")
    _seed_review(conn, text="python-1", author="alice", language="python")
    llm = FakeLLM(response="scala voice")
    out = developer_profile(conn, llm, author_login="alice", language="scala")
    assert out["n_samples"] == 2
    assert out["language"] == "scala"
    assert "python-1" not in llm.calls[0][1], "python comment leaked through language filter"


def test_developer_profile_repo_filter_narrows_corpus(conn):
    _seed_review(conn, text="x-1", author="alice", repo="me/x")
    _seed_review(conn, text="y-1", author="alice", repo="me/y")
    llm = FakeLLM(response="repo-x voice")
    out = developer_profile(conn, llm, author_login="alice", repo="me/x")
    assert out["n_samples"] == 1
    assert out["repo"] == "me/x"


def test_developer_profile_cache_key_distinguishes_scopes(conn):
    """Two different filter combinations must populate two distinct
    cache entries, not clobber each other."""
    _seed_review(conn, text="scala-1", author="alice", language="scala")
    _seed_review(conn, text="python-1", author="alice", language="python")
    llm = FakeLLM(response="scala profile")
    developer_profile(conn, llm, author_login="alice", language="scala")
    llm.response = "python profile"
    developer_profile(conn, llm, author_login="alice", language="python")
    # Two distinct cache rows now exist.
    cached_scala = q.get_cached_profile(conn, "alice|lang=scala")
    cached_python = q.get_cached_profile(conn, "alice|lang=python")
    assert cached_scala is not None
    assert cached_scala["profile_md"] == "scala profile"
    assert cached_python is not None
    assert cached_python["profile_md"] == "python profile"
    # Unfiltered key didn't get written by the scoped calls.
    assert q.get_cached_profile(conn, "alice") is None


def test_developer_profile_cache_unscoped_key_unchanged(conn):
    """Filterless call still keys under the bare login — preserves
    existing cache entries written before scope filters existed."""
    _seed_review(conn, text="c", author="alice")
    llm = FakeLLM(response="profile")
    developer_profile(conn, llm, author_login="alice")
    assert q.get_cached_profile(conn, "alice") is not None
    assert q.get_cached_profile(conn, "alice|lang=python") is None
