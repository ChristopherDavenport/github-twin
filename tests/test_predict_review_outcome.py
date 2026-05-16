"""Tests for P3: PR-summary chunker, ingest hookup, and the prediction tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github_twin.embed.base import Embedder
from github_twin.mcp_server import tools as t
from github_twin.process.chunkers import MAX_PR_BODY_CHARS, chunk_pr_summary
from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.store.vector_store import SqliteVecStore

# ---------- chunker ----------


def test_chunk_pr_summary_combines_title_and_body():
    out = chunk_pr_summary(
        title="Refactor auth",
        body="Splits middleware into two layers.",
        repo="me/x",
        pr_number=42,
        source_url="https://gh/x/42",
    )
    assert out is not None
    assert "Refactor auth" in out.text
    assert "two layers" in out.text
    assert out.context == {
        "repo": "me/x",
        "pr_number": 42,
        "pr_title": "Refactor auth",
        "url": "https://gh/x/42",
    }


def test_chunk_pr_summary_truncates_long_body():
    long_body = "x" * (MAX_PR_BODY_CHARS * 3)
    out = chunk_pr_summary(title="t", body=long_body, repo="r", pr_number=1, source_url=None)
    assert out is not None
    # Body portion can't exceed cap; title contributes a few chars + "\n\n".
    assert len(out.text) <= MAX_PR_BODY_CHARS + 10


def test_chunk_pr_summary_handles_empty_body():
    out = chunk_pr_summary(title="t", body="", repo="r", pr_number=1, source_url=None)
    assert out is not None
    assert out.text == "t"


def test_chunk_pr_summary_returns_none_when_nothing_to_embed():
    assert chunk_pr_summary(title="", body="", repo="r", pr_number=1, source_url=None) is None


# ---------- prediction aggregation ----------


class FakeEmbedder:
    """Maps the first character of input text to a fixed pattern, so tests
    can produce predictable "similar" / "dissimilar" PRs by construction."""

    dim = 4
    model_id = "fake"
    PATTERNS = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0, 0.0],
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for s in texts:
            for k, v in self.PATTERNS.items():
                if k in s:
                    out.append(list(v))
                    break
            else:
                out.append([0.0, 0.0, 0.0, 1.0])
        return out


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "pred.sqlite", embed_dim=FakeEmbedder.dim)
    yield db
    db.close()


def _seed_pr(
    conn,
    *,
    pr_num: int,
    text: str,
    vec: list[float],
    decision: str | None = None,
    reviewer_decisions: list[dict[str, Any]] | None = None,
) -> int:
    aid = q.upsert_artifact(
        conn,
        kind="pr",
        external_id=f"me/x#{pr_num}",
        source_url=f"https://gh/x/{pr_num}",
        repo="me/x",
        language=None,
        author_email=None,
        created_at=None,
        decision=decision,
        meta={"title": text, "reviewer_decisions": reviewer_decisions or []},
    )
    cid = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="pr_summary",
        text=text,
        context={"pr_number": pr_num, "pr_title": text, "repo": "me/x"},
        language=None,
    )
    q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="fake")
    return aid


def test_predict_returns_unknown_when_no_pr_summaries(conn):
    emb = FakeEmbedder()
    out = t.predict_review_outcome(conn, emb, SqliteVecStore(conn), diff_or_summary="A something")
    assert out["prediction"] == "unknown"
    assert out["n_pulled"] == 0


def test_predict_uses_artifact_level_decision_in_user_mode(conn):
    """User-mode: artifact.decision drives the vote. Three similar PRs all
    approved → prediction is 'approved' with high confidence."""
    emb = FakeEmbedder()
    for i in range(3):
        _seed_pr(
            conn,
            pr_num=i,
            text=f"A-{i}",
            vec=[1.0, 0.0, 0.0, 0.0],
            decision="approved",
        )
    # One outlier with a different vector and a different decision.
    _seed_pr(
        conn,
        pr_num=99,
        text="B-99",
        vec=[0.0, 1.0, 0.0, 0.0],
        decision="changes_requested",
    )

    out = t.predict_review_outcome(
        conn,
        emb,
        SqliteVecStore(conn),
        diff_or_summary="A new candidate",
        k=4,
    )
    assert out["prediction"] == "approved"
    assert out["n_with_decision"] == 4
    assert out["weighted"]["approved"] > out["weighted"]["changes_requested"]
    assert out["confidence"] >= 0.5


def test_predict_with_author_login_uses_reviewer_decisions(conn):
    """Org-mode: artifact.decision is NULL; the right vote lives in
    meta.reviewer_decisions. Filter to one reviewer."""
    emb = FakeEmbedder()
    _seed_pr(
        conn,
        pr_num=1,
        text="A-1",
        vec=[1.0, 0.0, 0.0, 0.0],
        decision=None,
        reviewer_decisions=[
            {"login": "alice", "state": "approved", "submitted_at": "2024-01-01"},
            {"login": "bob", "state": "changes_requested", "submitted_at": "2024-01-02"},
        ],
    )
    _seed_pr(
        conn,
        pr_num=2,
        text="A-2",
        vec=[1.0, 0.0, 0.0, 0.0],
        decision=None,
        reviewer_decisions=[
            {"login": "alice", "state": "approved", "submitted_at": "2024-02-01"},
        ],
    )

    store = SqliteVecStore(conn)
    # Author-scoped: alice approved both -> 'approved'.
    alice = t.predict_review_outcome(
        conn,
        emb,
        store,
        diff_or_summary="A candidate",
        author_login="alice",
        k=5,
    )
    assert alice["prediction"] == "approved"
    assert alice["n_with_decision"] == 2

    # bob only voted on PR 1, requesting changes — sole signal -> changes_requested
    bob = t.predict_review_outcome(
        conn,
        emb,
        store,
        diff_or_summary="A candidate",
        author_login="bob",
        k=5,
    )
    assert bob["prediction"] == "changes_requested"
    assert bob["n_with_decision"] == 1


def test_predict_unknown_when_no_decisions_match(conn):
    """Pulled some similar PRs but none has a usable decision -> unknown
    (still report n_pulled and support)."""
    emb = FakeEmbedder()
    _seed_pr(conn, pr_num=1, text="A-1", vec=[1.0, 0.0, 0.0, 0.0], decision=None)
    out = t.predict_review_outcome(
        conn,
        emb,
        SqliteVecStore(conn),
        diff_or_summary="A candidate",
        k=5,
    )
    assert out["prediction"] == "unknown"
    assert out["n_pulled"] == 1
    assert out["n_with_decision"] == 0
    assert len(out["support"]) == 1


def test_predict_empty_input_returns_unknown(conn):
    emb = FakeEmbedder()
    out = t.predict_review_outcome(
        conn,
        emb,
        SqliteVecStore(conn),
        diff_or_summary="   ",
        k=5,
    )
    assert out["prediction"] == "unknown"
    assert out["n_pulled"] == 0


# Embedder protocol sanity — confirms test FakeEmbedder is structurally valid.
def test_fake_embedder_is_an_embedder():
    assert isinstance(FakeEmbedder(), Embedder)
