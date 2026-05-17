"""RRF fusion of vector + BM25 results in store.vector_store.hybrid_search.

Uses a tiny FakeEmbedder (first-char keyed) so we can stage chunks where
vector and BM25 disagree, then assert the fused ranking matches the RRF
formula `1/(60 + rank + 1)` exactly.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.store.vector_store import (
    RRF_K,
    SqliteVecStore,
    VectorSearchFilters,
    hybrid_search,
)
from tests.conftest import seed_target


class FakeEmbedder:
    dim = 4
    model_id = "fake"
    PATTERNS = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0, 0.0],
        "C": [0.0, 0.0, 1.0, 0.0],
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
    db = open_db(tmp_path / "hybrid.sqlite", embed_dim=FakeEmbedder.dim)
    seed_target(db)
    yield db
    db.close()


def _seed(conn, *, text, vec):
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="commit",
        external_id=f"code-{text}",
        source_url=None,
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
        kind="code",
        text=text,
        context={"language": "python"},
        language="python",
    )
    q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="fake")
    return cid


def test_hybrid_surfaces_bm25_only_winner(conn):
    """A chunk whose embedding is far from the query but whose text contains
    the query token must still appear via the BM25 leg."""
    # Vector-near (pattern A) but no lexical overlap with query "needle_token".
    near_vec = _seed(conn, text="A common phrase", vec=[1.0, 0.0, 0.0, 0.0])
    # Vector-far (pattern C) but contains the exact keyword.
    bm25_hit = _seed(conn, text="C needle_token in code", vec=[0.0, 0.0, 1.0, 0.0])
    _seed(conn, text="B unrelated", vec=[0.0, 1.0, 0.0, 0.0])

    store = SqliteVecStore(conn)
    embedder = FakeEmbedder()
    qvec = embedder.embed(["A query"])[0]  # pattern A
    hits = hybrid_search(
        store,
        conn,
        query_vec=qvec,
        query_text="needle_token",
        filters=VectorSearchFilters(chunk_kind="code"),
        k=5,
    )
    ids = [h.chunk_id for h in hits]
    assert near_vec in ids  # vector wins
    assert bm25_hit in ids  # BM25 wins


def test_hybrid_surfaces_vector_only_winner(conn):
    """A chunk near in vector space but without the literal keyword still
    appears via the vector leg."""
    near = _seed(conn, text="A semantic neighbor", vec=[1.0, 0.0, 0.0, 0.0])
    _seed(conn, text="C totally elsewhere", vec=[0.0, 0.0, 1.0, 0.0])

    store = SqliteVecStore(conn)
    embedder = FakeEmbedder()
    qvec = embedder.embed(["A query"])[0]
    hits = hybrid_search(
        store,
        conn,
        query_vec=qvec,
        query_text="nonexistent_term_xyzzy",
        filters=VectorSearchFilters(chunk_kind="code"),
        k=5,
    )
    assert near in [h.chunk_id for h in hits]


def test_hybrid_rrf_ranking_matches_formula(conn):
    """Chunk present in both lists outranks chunk in only one list.

    Build a scenario where:
      - chunk X: rank 0 in vector, rank 0 in BM25  -> score = 2/(60+1)
      - chunk Y: rank 0 in vector only            -> score = 1/(60+1)
    Expect X first.
    """
    x = _seed(conn, text="A shared_keyword here", vec=[1.0, 0.0, 0.0, 0.0])
    y = _seed(conn, text="A different content", vec=[0.99, 0.01, 0.0, 0.0])
    # Make sure there's at least one BM25-only candidate so the BM25 leg
    # returns rows in a sensible order: x at rank 0 because "shared_keyword"
    # is exactly the query text.
    _seed(conn, text="C noise", vec=[0.0, 0.0, 1.0, 0.0])

    store = SqliteVecStore(conn)
    embedder = FakeEmbedder()
    qvec = embedder.embed(["A query"])[0]
    hits = hybrid_search(
        store,
        conn,
        query_vec=qvec,
        query_text="shared_keyword",
        filters=VectorSearchFilters(chunk_kind="code"),
        k=5,
    )
    ids = [h.chunk_id for h in hits]
    assert ids.index(x) < ids.index(y)

    # Verify the score-to-distance mapping: top hit has distance close to
    # 1 - 2/(60+1) ≈ 0.9672.
    expected = 1.0 - 2.0 / (RRF_K + 1)
    assert abs(hits[0].distance - expected) < 1e-9


def test_hybrid_passes_expander_only_to_bm25_leg(conn):
    """Asymmetry contract: when `expander` is set, the BM25 leg sees the
    expanded MATCH expression while the vector leg's `query_vec` is
    untouched. This pins amanmcp's "expand BM25 only" finding in code —
    both-backends-expanded measured -25pp on their corpus; vector
    expansion dilutes the dense match.

    Implemented via a spy VectorStore that records every search call,
    plus a stub expander that emits a distinctive alternate token.
    """

    class SpyStore:
        backend_id = "spy"

        def __init__(self, inner: SqliteVecStore) -> None:
            self._inner = inner
            self.calls: list[list[float]] = []

        def search(self, query_vec, *, filters, k=5):
            self.calls.append(list(query_vec))
            return self._inner.search(query_vec, filters=filters, k=k)

    class StubExpander:
        backend_id = "stub"
        captured: list[str] = []

        def expand(self, text: str):
            self.captured.append(text)
            # Force a non-trivial OR-group with a token that would never
            # appear in `query_vec` if the vector leg saw it.
            return [["alpha", "expanded_only_token_xxxxx"]]

    cid = _seed(conn, text="alpha keyword", vec=[1.0, 0, 0, 0])

    inner = SqliteVecStore(conn)
    spy = SpyStore(inner)
    embedder = FakeEmbedder()
    qvec = embedder.embed(["A query"])[0]  # pattern A
    expander = StubExpander()

    hits = hybrid_search(
        spy,
        conn,
        query_vec=qvec,
        query_text="alpha",
        filters=VectorSearchFilters(chunk_kind="code"),
        k=5,
        expander=expander,
    )

    # Vector leg ran exactly once, with the original embedding.
    assert spy.calls == [list(qvec)]
    # BM25 leg ran exactly once, with the original query text — the
    # expander is invoked there.
    assert expander.captured == ["alpha"]
    # Sanity: the hit still surfaces (via BM25 on the alternate or via
    # vector on the original — either way the chunk is present).
    assert cid in [h.chunk_id for h in hits]


def test_hybrid_respects_filters(conn):
    """Filters must apply to BOTH retriever legs."""
    aid_py = q.upsert_artifact(
        conn,
        target_id=1,
        kind="commit",
        external_id="py",
        source_url=None,
        repo="me/x",
        language="python",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    cid_py = q.insert_chunk(
        conn,
        artifact_id=aid_py,
        kind="code",
        text="A shared_word python",
        context={"language": "python"},
        language="python",
    )
    q.write_embedding(conn, chunk_id=cid_py, embedding=[1.0, 0, 0, 0], model_id="fake")

    aid_go = q.upsert_artifact(
        conn,
        target_id=1,
        kind="commit",
        external_id="go",
        source_url=None,
        repo="me/x",
        language="go",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    cid_go = q.insert_chunk(
        conn,
        artifact_id=aid_go,
        kind="code",
        text="A shared_word golang",
        context={"language": "go"},
        language="go",
    )
    q.write_embedding(conn, chunk_id=cid_go, embedding=[1.0, 0, 0, 0], model_id="fake")

    store = SqliteVecStore(conn)
    embedder = FakeEmbedder()
    qvec = embedder.embed(["A"])[0]
    hits = hybrid_search(
        store,
        conn,
        query_vec=qvec,
        query_text="shared_word",
        filters=VectorSearchFilters(chunk_kind="code", language="python"),
        k=5,
    )
    assert [h.chunk_id for h in hits] == [cid_py]


# ---------- recency decay ----------


def _seed_with_age(
    conn, *, text: str, vec: list[float], created_at: str | None, artifact_kind: str = "commit"
) -> int:
    """Like _seed, but lets the test pin the artifact's created_at and kind.

    `commit` is the default kind so the chunk participates in decay; pass
    `file` to verify the kind-exclusion path.
    """
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind=artifact_kind,
        external_id=f"age-{text}-{created_at}-{artifact_kind}",
        source_url=None,
        repo="me/x",
        language="python",
        author_email=None,
        author_login=None,
        created_at=created_at,
        decision=None,
        meta=None,
    )
    chunk_kind = "file" if artifact_kind == "file" else "code"
    cid = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind=chunk_kind,
        text=text,
        context={"language": "python"},
        language="python",
    )
    q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="fake")
    return cid


def test_hybrid_recency_reranks_old_below_new(conn):
    """Two chunks with identical RRF input — the older one drops below
    the newer once decay is applied with a meaningful half-life."""
    ref = datetime(2026, 5, 16, tzinfo=UTC)
    old_ts = (ref - timedelta(days=730)).isoformat()  # ~2 years
    new_ts = (ref - timedelta(days=1)).isoformat()  # yesterday

    # Both contain the same keyword (BM25 tie) and both embed to pattern A
    # (vector tie). Their only differentiator is created_at.
    old_id = _seed_with_age(conn, text="A shared_word old", vec=[1.0, 0, 0, 0], created_at=old_ts)
    new_id = _seed_with_age(conn, text="A shared_word new", vec=[1.0, 0, 0, 0], created_at=new_ts)

    store = SqliteVecStore(conn)
    embedder = FakeEmbedder()
    qvec = embedder.embed(["A query"])[0]

    # Baseline (no decay): both surface; ordering is determined by FTS5 /
    # vector tie-break order, not by age.
    baseline = hybrid_search(
        store,
        conn,
        query_vec=qvec,
        query_text="shared_word",
        filters=VectorSearchFilters(chunk_kind="code"),
        k=5,
        now=ref,
    )
    baseline_ids = {h.chunk_id for h in baseline}
    assert {old_id, new_id} <= baseline_ids

    # With a 1-year half-life: weight(old)=0.5**2=0.25, weight(new)≈1.0.
    decayed = hybrid_search(
        store,
        conn,
        query_vec=qvec,
        query_text="shared_word",
        filters=VectorSearchFilters(chunk_kind="code"),
        k=5,
        recency_half_life_days=365.0,
        now=ref,
    )
    ids = [h.chunk_id for h in decayed]
    assert ids.index(new_id) < ids.index(old_id)


def test_hybrid_recency_skips_file_kind(conn):
    """File-at-HEAD chunks (artifact.kind='file') are excluded from decay
    so an ancient file-snapshot keeps its original rank even with a tiny
    half-life that would otherwise crush it."""
    ref = datetime(2026, 5, 16, tzinfo=UTC)
    ancient = (ref - timedelta(days=3650)).isoformat()  # 10 years
    new_ts = (ref - timedelta(days=1)).isoformat()

    # `file` chunks live in chunk_kind='file'; we need the filter to match.
    file_old = _seed_with_age(
        conn,
        text="A shared_word ancient_file",
        vec=[1.0, 0, 0, 0],
        created_at=ancient,
        artifact_kind="file",
    )
    file_new = _seed_with_age(
        conn,
        text="A shared_word recent_file",
        vec=[0.99, 0.01, 0, 0],
        created_at=new_ts,
        artifact_kind="file",
    )

    store = SqliteVecStore(conn)
    embedder = FakeEmbedder()
    qvec = embedder.embed(["A query"])[0]

    # Capture order without decay…
    baseline = hybrid_search(
        store,
        conn,
        query_vec=qvec,
        query_text="shared_word",
        filters=VectorSearchFilters(chunk_kind="file"),
        k=5,
        now=ref,
    )
    baseline_ids = [h.chunk_id for h in baseline]

    # …and verify a 30-day half-life leaves it intact (no decay applied
    # because artifact.kind='file' is excluded).
    decayed = hybrid_search(
        store,
        conn,
        query_vec=qvec,
        query_text="shared_word",
        filters=VectorSearchFilters(chunk_kind="file"),
        k=5,
        recency_half_life_days=30.0,
        now=ref,
    )
    assert [h.chunk_id for h in decayed] == baseline_ids
    assert {file_old, file_new} <= set(baseline_ids)


def test_hybrid_recency_missing_created_at_is_no_penalty(conn):
    """A chunk whose artifact has NULL created_at must not be silently
    penalized — current policy treats undatable chunks as weight=1.0."""
    ref = datetime(2026, 5, 16, tzinfo=UTC)
    old_ts = (ref - timedelta(days=3650)).isoformat()

    no_date = _seed_with_age(
        conn, text="A shared_word undated", vec=[1.0, 0, 0, 0], created_at=None
    )
    old_id = _seed_with_age(
        conn, text="A shared_word ancient", vec=[1.0, 0, 0, 0], created_at=old_ts
    )

    store = SqliteVecStore(conn)
    embedder = FakeEmbedder()
    qvec = embedder.embed(["A query"])[0]

    hits = hybrid_search(
        store,
        conn,
        query_vec=qvec,
        query_text="shared_word",
        filters=VectorSearchFilters(chunk_kind="code"),
        k=5,
        recency_half_life_days=365.0,
        now=ref,
    )
    ids = [h.chunk_id for h in hits]
    # The undated chunk should outrank the 10-year-old chunk (weight 1.0
    # vs. 0.5**10 ≈ 0.001).
    assert ids.index(no_date) < ids.index(old_id)


def test_hybrid_recency_none_is_identical_to_omitting(conn):
    """Passing `recency_half_life_days=None` must produce the same hits +
    distances as omitting the parameter entirely (defensive: bit-identical
    no-op when off)."""
    ref = datetime(2026, 5, 16, tzinfo=UTC)
    _seed_with_age(
        conn,
        text="A shared_word a",
        vec=[1.0, 0, 0, 0],
        created_at=(ref - timedelta(days=100)).isoformat(),
    )
    _seed_with_age(
        conn,
        text="A shared_word b",
        vec=[0.99, 0.01, 0, 0],
        created_at=(ref - timedelta(days=1000)).isoformat(),
    )

    store = SqliteVecStore(conn)
    embedder = FakeEmbedder()
    qvec = embedder.embed(["A query"])[0]
    common = dict(
        query_vec=qvec,
        query_text="shared_word",
        filters=VectorSearchFilters(chunk_kind="code"),
        k=5,
    )
    omitted = hybrid_search(store, conn, **common)
    passed_none = hybrid_search(store, conn, recency_half_life_days=None, **common)
    zero = hybrid_search(store, conn, recency_half_life_days=0, **common)

    def _shape(hs):
        return [(h.chunk_id, round(h.distance, 9)) for h in hs]

    assert _shape(omitted) == _shape(passed_none)
    # Zero / negative is also treated as off (avoids divide-by-zero).
    assert _shape(omitted) == _shape(zero)
