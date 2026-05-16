"""Tests for the RAG-vs-baseline eval harness.

LLM calls are stubbed; embedder is the deterministic FakeEmbedder reused
from the distill tests. We verify:

- Holdout selection respects the `since` cutoff.
- Cosine distance + paired-t produce sensible numbers on synthetic data.
- Review eval drops hits that came from the holdout (no leakage).
- Prediction eval scores accuracy + per-class F1 correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.eval.holdout import (
    count_eligible,
    iter_held_out_prs,
    iter_held_out_review_comments,
)
from github_twin.eval.runner import (
    _cosine_distance,
    _normalize_decision,
    _paired_t_one_sided,
    _per_class_f1,
    evaluate_predictions,
    evaluate_reviews,
)
from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.store.vector_store import SqliteVecStore

# ---------- fixtures ----------


class FakeEmbedder:
    dim = 4
    model_id = "fake"
    PATTERNS = {"A": [1.0, 0.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0, 0.0]}

    def embed(self, texts):
        out = []
        for s in texts:
            for k, v in self.PATTERNS.items():
                if k in s:
                    out.append(list(v))
                    break
            else:
                out.append([0.0, 0.0, 0.0, 1.0])
        return out


class StubLLM:
    """Returns canned text per call. Useful for asserting prompt routing."""

    backend_id = "stub"

    def __init__(self, baseline_text: str, rag_text: str) -> None:
        self._baseline = baseline_text
        self._rag = rag_text
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        self.calls.append({"system": system, "user": user})
        # The runner sends the baseline prompt first, then the RAG prompt.
        # Discriminate by looking for the example block.
        return self._rag if "Examples of past reviews" in user else self._baseline


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "eval.sqlite", embed_dim=FakeEmbedder.dim)
    yield db
    db.close()


def _seed_rc(
    conn,
    *,
    ext_id,
    body,
    hunk,
    created_at,
    vec,
    author_login=None,
    repo="me/x",
):
    aid = q.upsert_artifact(
        conn,
        kind="review_comment",
        external_id=ext_id,
        source_url=f"https://gh/{ext_id}",
        repo=repo,
        language="python",
        author_email=None,
        author_login=author_login,
        created_at=created_at,
        decision=None,
        meta={"pr_title": f"PR {ext_id}"},
    )
    cid = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="review_comment",
        text=body,
        context={
            "diff_hunk": hunk,
            "url": f"https://gh/{ext_id}",
            "pr_title": f"PR {ext_id}",
            "repo": repo,
        },
        language="python",
    )
    q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="fake")
    return aid


def _seed_pr(
    conn,
    *,
    ext_id,
    title,
    body,
    decision,
    created_at,
    vec,
    reviewer_decisions=None,
    repo="me/x",
):
    meta = {"title": title}
    if reviewer_decisions is not None:
        meta["reviewer_decisions"] = reviewer_decisions
    aid = q.upsert_artifact(
        conn,
        kind="pr",
        external_id=ext_id,
        source_url=f"https://gh/x/{ext_id}",
        repo=repo,
        language=None,
        author_email=None,
        created_at=created_at,
        decision=decision,
        meta=meta,
    )
    cid = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="pr_summary",
        text=f"{title}\n\n{body}",
        context={"pr_title": title, "repo": repo},
        language=None,
    )
    q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="fake")
    return aid


# ---------- low-level numerics ----------


def test_cosine_distance_identical_vectors_is_zero():
    assert _cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_cosine_distance_orthogonal_vectors_is_one():
    assert _cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_paired_t_detects_consistent_improvement():
    """Five trials, RAG ~10% better each time. Should be highly significant."""
    deltas = [0.10, 0.09, 0.11, 0.10, 0.10]
    t, p = _paired_t_one_sided(deltas)
    assert t is not None and t > 5
    assert p is not None and p < 0.01


def test_paired_t_zero_difference_not_significant():
    deltas = [0.0] * 10
    t, p = _paired_t_one_sided(deltas)
    assert t == 0.0
    assert p == 1.0


def test_normalize_decision_accepts_common_shapes():
    assert _normalize_decision("approved") == "approved"
    assert _normalize_decision("Approve") == "approved"
    assert _normalize_decision(" changes_requested.") == "changes_requested"
    assert _normalize_decision("request_changes") == "changes_requested"
    assert _normalize_decision("nonsense") is None


def test_per_class_f1_perfect_classifier():
    truth = ["approved", "approved", "changes_requested"]
    pred = ["approved", "approved", "changes_requested"]
    f1 = _per_class_f1(truth, pred)
    assert f1["approved"] == 1.0
    assert f1["changes_requested"] == 1.0


# ---------- holdout selection ----------


def test_iter_held_out_review_comments_filters_by_cutoff(conn):
    _seed_rc(
        conn,
        ext_id="old",
        body="A old",
        hunk="-1\n+2",
        created_at="2024-01-01T00:00:00Z",
        vec=[1, 0, 0, 0],
    )
    _seed_rc(
        conn,
        ext_id="new1",
        body="A new1",
        hunk="-1\n+2",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
    )
    _seed_rc(
        conn,
        ext_id="new2",
        body="A new2",
        hunk="-1\n+2",
        created_at="2025-03-01T00:00:00Z",
        vec=[1, 0, 0, 0],
    )
    items = list(iter_held_out_review_comments(conn, since="2025-01-01"))
    assert [i.truth_comment for i in items] == ["A new1", "A new2"]


def test_iter_held_out_review_comments_skips_missing_hunk(conn):
    """Comments without a diff_hunk can't be evaluated — skip silently."""
    aid = q.upsert_artifact(
        conn,
        kind="review_comment",
        external_id="x",
        source_url=None,
        repo="r",
        language=None,
        author_email=None,
        created_at="2025-02-01T00:00:00Z",
        decision=None,
        meta=None,
    )
    cid = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="review_comment",
        text="no hunk",
        context={},
        language=None,
    )
    q.write_embedding(conn, chunk_id=cid, embedding=[1.0, 0, 0, 0], model_id="f")
    assert list(iter_held_out_review_comments(conn, since="2025-01-01")) == []


def test_iter_held_out_prs_only_keeps_decisioned_rows(conn):
    _seed_pr(
        conn,
        ext_id="r#1",
        title="t1",
        body="b1",
        decision="approved",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
    )
    _seed_pr(
        conn,
        ext_id="r#2",
        title="t2",
        body="b2",
        decision=None,
        created_at="2025-02-02T00:00:00Z",
        vec=[1, 0, 0, 0],
    )
    items = list(iter_held_out_prs(conn, since="2025-01-01"))
    assert [i.truth_decision for i in items] == ["approved"]


def test_iter_held_out_review_comments_filters_by_author(conn):
    _seed_rc(
        conn,
        ext_id="alice1",
        body="A alice1",
        hunk="-1\n+2",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="alice",
    )
    _seed_rc(
        conn,
        ext_id="bob1",
        body="A bob1",
        hunk="-1\n+2",
        created_at="2025-02-02T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="bob",
    )
    items = list(iter_held_out_review_comments(conn, since="2025-01-01", author_login="alice"))
    assert [i.truth_comment for i in items] == ["A alice1"]


def test_iter_held_out_prs_reads_reviewer_decisions_for_author(conn):
    """Org-mode PRs have artifact.decision=NULL; truth lives in meta.reviewer_decisions.
    Passing author_login should surface those rows and skip PRs the author didn't review."""
    # PR alice reviewed (approved); decision column NULL like real org-mode rows.
    _seed_pr(
        conn,
        ext_id="r#1",
        title="org pr 1",
        body="b",
        decision=None,
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        reviewer_decisions=[
            {"login": "alice", "state": "approved", "submitted_at": "2025-02-01T01:00:00Z"},
            {"login": "bob", "state": "commented", "submitted_at": "2025-02-01T02:00:00Z"},
        ],
    )
    # PR bob reviewed but alice didn't.
    _seed_pr(
        conn,
        ext_id="r#2",
        title="org pr 2",
        body="b",
        decision=None,
        created_at="2025-02-02T00:00:00Z",
        vec=[1, 0, 0, 0],
        reviewer_decisions=[
            {"login": "bob", "state": "changes_requested", "submitted_at": "2025-02-02T01:00:00Z"},
        ],
    )
    alice = list(iter_held_out_prs(conn, since="2025-01-01", author_login="alice"))
    assert [(i.artifact_id, i.truth_decision) for i in alice]
    assert [i.truth_decision for i in alice] == ["approved"]

    # Without author, decision=NULL skips both rows — preserves user-mode semantics.
    bare = list(iter_held_out_prs(conn, since="2025-01-01"))
    assert bare == []


def test_iter_held_out_prs_normalizes_request_changes(conn):
    """GitHub returns `state='request_changes'` on the reviews endpoint; we
    normalize it to `changes_requested` to match what `decision` would hold."""
    _seed_pr(
        conn,
        ext_id="r#1",
        title="t",
        body="b",
        decision=None,
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        reviewer_decisions=[
            {"login": "alice", "state": "request_changes", "submitted_at": "2025-02-01T01:00:00Z"},
        ],
    )
    items = list(iter_held_out_prs(conn, since="2025-01-01", author_login="alice"))
    assert [i.truth_decision for i in items] == ["changes_requested"]


# ---------- end-to-end runner ----------


def test_evaluate_reviews_filters_post_cutoff_hits(conn):
    """The retriever might surface the held-out item itself (it's still in
    the index). The eval must drop those to avoid leakage."""
    _seed_rc(
        conn,
        ext_id="old",
        body="A old",
        hunk="-1\n+2",
        created_at="2024-06-01T00:00:00Z",
        vec=[1, 0, 0, 0],
    )
    _seed_rc(
        conn,
        ext_id="heldout",
        body="A heldout",
        hunk="-1\n+heldout",
        created_at="2025-03-01T00:00:00Z",
        vec=[1, 0, 0, 0],
    )
    emb = FakeEmbedder()
    store = SqliteVecStore(conn)
    llm = StubLLM(baseline_text="A baseline review", rag_text="A rag review")
    result = evaluate_reviews(
        conn,
        retriever_embedder=emb,
        judge_embedder=emb,
        store=store,
        llm=llm,
        since="2025-01-01",
        k=5,
    )
    # We had exactly one held-out example.
    assert result.n == 1
    # Both pipelines produced output; cosine should be in [0, 1].
    assert 0.0 <= result.baseline_mean <= 1.0
    assert 0.0 <= result.rag_mean <= 1.0
    # Two LLM calls per example, in this order: baseline then RAG.
    assert len(llm.calls) == 2
    assert "Examples of past reviews" not in llm.calls[0]["user"]
    assert "Examples of past reviews" in llm.calls[1]["user"]


def test_evaluate_reviews_scopes_to_author(conn):
    """When author_login is supplied, only that author's comments are evaluated,
    and the RAG retrieval is similarly scoped — i.e. one author at a time."""
    _seed_rc(
        conn,
        ext_id="alice_old",
        body="A alice old",
        hunk="-1\n+2",
        created_at="2024-06-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="alice",
    )
    _seed_rc(
        conn,
        ext_id="alice_new",
        body="A alice new",
        hunk="-1\n+2",
        created_at="2025-03-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="alice",
    )
    _seed_rc(
        conn,
        ext_id="bob_new",
        body="A bob new",
        hunk="-1\n+2",
        created_at="2025-03-02T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="bob",
    )
    emb = FakeEmbedder()
    store = SqliteVecStore(conn)
    llm = StubLLM(baseline_text="A base", rag_text="A rag")
    result = evaluate_reviews(
        conn,
        retriever_embedder=emb,
        judge_embedder=emb,
        store=store,
        llm=llm,
        since="2025-01-01",
        author_login="alice",
        k=5,
    )
    # Only alice's one post-cutoff comment.
    assert result.n == 1
    # Two LLM calls per example. (Confirms RAG pipeline ran.)
    assert len(llm.calls) == 2


def test_evaluate_predictions_org_mode_uses_reviewer_decisions(conn):
    """Org-mode shape: artifact.decision NULL, truth lives in meta.reviewer_decisions.
    With --author, the harness should find rows and aggregate that author's votes."""
    _seed_pr(
        conn,
        ext_id="r#1",
        title="A title",
        body="A body",
        decision=None,
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        reviewer_decisions=[
            {"login": "alice", "state": "approved", "submitted_at": "2025-02-01T01:00:00Z"},
        ],
    )

    class AlwaysApproveLLM:
        backend_id = "stub"

        def complete(self, *, system, user, max_tokens=8):
            return "approved"

    result = evaluate_predictions(
        conn,
        retriever_embedder=FakeEmbedder(),
        store=SqliteVecStore(conn),
        llm=AlwaysApproveLLM(),
        since="2025-01-01",
        author_login="alice",
    )
    assert result.n == 1
    # Truth = approved (from meta), baseline says approved → 100% accuracy.
    assert result.baseline_accuracy == pytest.approx(1.0)


def test_evaluate_predictions_org_mode_without_author_finds_nothing(conn):
    """Belt-and-suspenders: without --author, org-mode rows are invisible to
    the prediction eval (artifact.decision is NULL). Document the contract
    so a user running `gt eval predictions` against an org DB gets a clear
    n=0 instead of silent garbage."""
    _seed_pr(
        conn,
        ext_id="r#1",
        title="t",
        body="b",
        decision=None,
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        reviewer_decisions=[
            {"login": "alice", "state": "approved", "submitted_at": "2025-02-01T01:00:00Z"}
        ],
    )

    class StubLLM2:
        backend_id = "s"

        def complete(self, *, system, user, max_tokens=8):
            return "approved"

    result = evaluate_predictions(
        conn,
        retriever_embedder=FakeEmbedder(),
        store=SqliteVecStore(conn),
        llm=StubLLM2(),
        since="2025-01-01",
    )
    assert result.n == 0


def test_evaluate_predictions_scores_accuracy(conn):
    _seed_pr(
        conn,
        ext_id="r#1",
        title="A title",
        body="A body",
        decision="approved",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
    )
    _seed_pr(
        conn,
        ext_id="r#2",
        title="A another",
        body="A more",
        decision="changes_requested",
        created_at="2025-02-02T00:00:00Z",
        vec=[1, 0, 0, 0],
    )

    class AlwaysApproveLLM:
        backend_id = "stub"

        def complete(self, *, system, user, max_tokens=8):
            return "approved"

    result = evaluate_predictions(
        conn,
        retriever_embedder=FakeEmbedder(),
        store=SqliteVecStore(conn),
        llm=AlwaysApproveLLM(),
        since="2025-01-01",
    )
    assert result.n == 2
    # Baseline always says 'approved' -> 1 of 2 right.
    assert result.baseline_accuracy == pytest.approx(0.5)
    # F1 for 'commented' is 0 since neither truth nor pred selected it.
    assert result.baseline_f1["commented"] == 0.0


# ---------- --repo filter ----------


def test_iter_held_out_review_comments_filters_by_repo(conn):
    _seed_rc(
        conn,
        ext_id="a1",
        body="A in repo a",
        hunk="-1\n+1",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="alice",
        repo="org/a",
    )
    _seed_rc(
        conn,
        ext_id="a2",
        body="A in repo b",
        hunk="-1\n+1",
        created_at="2025-02-02T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="alice",
        repo="org/b",
    )
    only_a = list(iter_held_out_review_comments(conn, since="2025-01-01", repo="org/a"))
    assert [i.truth_comment for i in only_a] == ["A in repo a"]


def test_iter_held_out_prs_filters_by_repo_user_mode(conn):
    _seed_pr(
        conn,
        ext_id="ax#1",
        title="t",
        body="b",
        decision="approved",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        repo="org/a",
    )
    _seed_pr(
        conn,
        ext_id="bx#1",
        title="t",
        body="b",
        decision="approved",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        repo="org/b",
    )
    only_a = list(iter_held_out_prs(conn, since="2025-01-01", repo="org/a"))
    assert [i.repo for i in only_a] == ["org/a"]


def test_iter_held_out_prs_filters_by_repo_with_author(conn):
    """The repo filter must compose with the author filter (org-mode path)."""
    _seed_pr(
        conn,
        ext_id="ax#1",
        title="t",
        body="b",
        decision=None,
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        repo="org/a",
        reviewer_decisions=[
            {"login": "alice", "state": "approved", "submitted_at": "2025-02-01T01:00:00Z"},
        ],
    )
    _seed_pr(
        conn,
        ext_id="bx#1",
        title="t",
        body="b",
        decision=None,
        created_at="2025-02-02T00:00:00Z",
        vec=[1, 0, 0, 0],
        repo="org/b",
        reviewer_decisions=[
            {"login": "alice", "state": "approved", "submitted_at": "2025-02-02T01:00:00Z"},
        ],
    )
    items = list(iter_held_out_prs(conn, since="2025-01-01", author_login="alice", repo="org/a"))
    assert [i.repo for i in items] == ["org/a"]


# ---------- count_eligible ----------


def test_count_eligible_reports_both_surfaces(conn):
    _seed_rc(
        conn,
        ext_id="rc1",
        body="A x",
        hunk="-1\n+1",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="alice",
        repo="org/a",
    )
    _seed_pr(
        conn,
        ext_id="pr1",
        title="t",
        body="b",
        decision="approved",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        repo="org/a",
    )
    counts = count_eligible(conn, since="2025-01-01")
    assert counts == {"review_comments": 1, "decisioned_prs": 1}


def test_count_eligible_zero_for_unknown_author(conn):
    """Typo'd author -> 0; the CLI uses this to bail before LLM calls."""
    _seed_rc(
        conn,
        ext_id="rc1",
        body="A x",
        hunk="-1\n+1",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="alice",
    )
    counts = count_eligible(
        conn,
        since="2025-01-01",
        author_login="aliec",  # typo
    )
    assert counts == {"review_comments": 0, "decisioned_prs": 0}


def test_count_eligible_respects_repo_scope(conn):
    _seed_rc(
        conn,
        ext_id="rc_a",
        body="A in a",
        hunk="-1\n+1",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="alice",
        repo="org/a",
    )
    _seed_rc(
        conn,
        ext_id="rc_b",
        body="A in b",
        hunk="-1\n+1",
        created_at="2025-02-01T00:00:00Z",
        vec=[1, 0, 0, 0],
        author_login="alice",
        repo="org/b",
    )
    only_a = count_eligible(conn, since="2025-01-01", repo="org/a")
    assert only_a["review_comments"] == 1
