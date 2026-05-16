"""End-to-end smoke for MCP tools running through hybrid retrieval.

Stages a review-comment corpus where one comment wins on lexical match only
and another wins on semantic match only — both must surface via
find_review_comments.
"""

from pathlib import Path

import pytest

from github_twin.mcp_server import tools as t
from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.store.vector_store import SqliteVecStore


class FakeEmbedder:
    dim = 4
    model_id = "fake"
    PATTERNS = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0, 0.0],
        "C": [0.0, 0.0, 1.0, 0.0],
    }

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


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "mcp_integration.sqlite", embed_dim=4)
    yield db
    db.close()


def _seed_review_comment(conn, *, text, vec, diff_hunk="def f(): pass"):
    aid = q.upsert_artifact(
        conn,
        kind="review_comment",
        external_id=f"rc-{text}",
        source_url="https://gh/x/1#r1",
        repo="me/x",
        language="python",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    cid = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="review_comment",
        text=text,
        context={
            "language": "python",
            "diff_hunk": diff_hunk,
            "path": "src/a.py",
            "pr_title": "x",
            "url": "https://gh/x/1",
        },
        language="python",
    )
    q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="fake")
    return cid


def test_find_review_comments_surfaces_both_signals(conn):
    """One chunk wins on vector (pattern A), another wins on lexical
    (contains the exact keyword from the query but vector-far). Both
    must surface."""
    semantic_winner = _seed_review_comment(
        conn,
        text="A always handle the empty-list case",
        vec=[1.0, 0, 0, 0],
    )
    lexical_winner = _seed_review_comment(
        conn,
        text="C don't forget to deduplicate_results upstream",
        vec=[0.0, 0.0, 1.0, 0.0],
    )
    # Distractor: not near in vector, not lexically related.
    _seed_review_comment(
        conn,
        text="B unrelated naming nit",
        vec=[0.0, 1.0, 0.0, 0.0],
    )

    store = SqliteVecStore(conn)
    embedder = FakeEmbedder()
    # Query whose embedding is pattern A AND contains the literal token from
    # the lexical winner.
    hits = t.find_review_comments(
        conn,
        embedder,
        store,
        diff_hunk="A deduplicate_results in the post-processing step",
        k=5,
    )
    comment_ids_by_text = {h["comment"]: h for h in hits}
    semantic_text = "A always handle the empty-list case"
    lexical_text = "C don't forget to deduplicate_results upstream"
    assert semantic_text in comment_ids_by_text
    assert lexical_text in comment_ids_by_text
    # Sanity: both have a small positive distance (the 1 - rrf_score mapping).
    assert all(0.0 < h["distance"] < 1.0 for h in hits)
    # Avoid unused-variable warnings for the ids.
    assert isinstance(semantic_winner, int) and isinstance(lexical_winner, int)


def test_find_review_comments_handles_metacharacter_query(conn):
    """A diff hunk full of FTS5 metacharacters must not raise."""
    _seed_review_comment(conn, text="A placeholder", vec=[1.0, 0, 0, 0])
    store = SqliteVecStore(conn)
    embedder = FakeEmbedder()
    # Realistic diff hunk: contains colons, parens, operators.
    hunk = "@@ -1,3 +1,3 @@\n-def foo(x: int) -> bool:\n+def foo(x: str) -> bool:"
    hits = t.find_review_comments(conn, embedder, store, diff_hunk=hunk, k=5)
    # Just asserting no exception; result content can be anything.
    assert isinstance(hits, list)
