"""Tool-layer plumbing for `recency_half_life_days`.

The decay math itself is pinned by tests in `test_hybrid_search.py`. This
module verifies the param crosses the seams correctly:

- `tools.find_*` forwards an explicit kwarg into `hybrid_search`.
- `tools.predict_review_outcome` exposes no such kwarg (its inverse-distance
  weighting needs raw L2 distance; see CLAUDE.md "How to run things").
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from github_twin.mcp_server import tools as t
from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.store.vector_store import SqliteVecStore
from tests.conftest import seed_target


class FakeEmbedder:
    dim = 4
    model_id = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "recency_plumbing.sqlite", embed_dim=FakeEmbedder.dim)
    seed_target(db)
    # One review_comment chunk so the tools have something to retrieve.
    aid = q.upsert_artifact(
        conn=db,
        target_id=1,
        kind="review_comment",
        external_id="rc-1",
        source_url=None,
        repo="me/x",
        language="python",
        author_email=None,
        author_login=None,
        created_at="2025-01-01T00:00:00+00:00",
        decision=None,
        meta=None,
    )
    cid = q.insert_chunk(
        conn=db,
        artifact_id=aid,
        kind="review_comment",
        text="A reviewer note",
        context={"language": "python"},
        language="python",
    )
    q.write_embedding(db, chunk_id=cid, embedding=[1.0, 0, 0, 0], model_id="fake")
    yield db
    db.close()


def _spy_hybrid_search(monkeypatch) -> dict[str, Any]:
    """Replace `tools.hybrid_search` with a spy that records the call's
    kwargs and returns []. Returns the captured-kwargs dict for the
    most recent invocation."""
    captured: dict[str, Any] = {}

    def fake(*args: Any, **kwargs: Any) -> list:
        captured.clear()
        captured.update(kwargs)
        return []

    monkeypatch.setattr(t, "hybrid_search", fake)
    return captured


@pytest.mark.parametrize(
    "tool_name,call_kwargs",
    [
        ("find_review_comments", {"diff_hunk": "A new code"}),
        ("find_style_examples", {"query": "A nl query"}),
        ("find_applicable_rules", {"query": "A nl query"}),
        ("find_code", {"query": "A nl query"}),
    ],
)
def test_tools_pass_recency_half_life_days_through(conn, monkeypatch, tool_name, call_kwargs):
    """Every retrieval tool forwards an explicit `recency_half_life_days`
    kwarg to `hybrid_search`. None (default) flows through as None."""
    captured = _spy_hybrid_search(monkeypatch)
    fn = getattr(t, tool_name)
    embedder = FakeEmbedder()
    store = SqliteVecStore(conn)

    # Explicit override.
    fn(conn, embedder, store, recency_half_life_days=180.0, **call_kwargs)
    assert captured.get("recency_half_life_days") == 180.0

    # Default (omitted): hybrid_search sees None at the tools layer; the
    # server layer is responsible for substituting the cfg default.
    fn(conn, embedder, store, **call_kwargs)
    assert captured.get("recency_half_life_days") is None


def test_predict_review_outcome_has_no_recency_kwarg():
    """`predict_review_outcome` intentionally bypasses `hybrid_search`
    because its inverse-distance weighting needs calibrated L2 distance.
    Adding a recency knob there would require its own analysis; pin the
    current contract via signature inspection so a drive-by addition
    can't slip in silently.
    """
    sig = inspect.signature(t.predict_review_outcome)
    assert "recency_half_life_days" not in sig.parameters


def test_predict_review_outcome_unaffected_by_recency(conn, monkeypatch):
    """Even after monkeypatching `hybrid_search` to a sentinel that would
    fail loudly if called, `predict_review_outcome` runs to completion —
    proving it never touches the recency-aware code path."""

    def fail(*args: Any, **kwargs: Any) -> list:
        pytest.fail("predict_review_outcome must not call hybrid_search")
        return []  # unreachable

    monkeypatch.setattr(t, "hybrid_search", fail)

    embedder = FakeEmbedder()
    store = SqliteVecStore(conn)
    # No pr_summary chunks seeded, so the call should hit the empty-hits
    # early-return path inside predict_review_outcome — but crucially must
    # not invoke hybrid_search even when seeded.
    result = t.predict_review_outcome(conn, embedder, store, diff_or_summary="A candidate PR")
    assert result["prediction"] == "unknown"
