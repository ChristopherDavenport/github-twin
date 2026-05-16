"""Tests for the pluggable VectorStore.

Two implementations:
- SqliteVecStore — always present, brute-force KNN via sqlite-vec.
- FaissVectorStore — opt-in (`pip install github-twin[faiss]`), skipped otherwise.

Both must agree on result ordering for a given query + filter, modulo float
drift in the distance metric. We assert ID equality (not exact distances) so
the L2 vs. precomputed difference doesn't cause spurious failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.store.vector_store import (
    SqliteVecStore,
    VectorSearchFilters,
    make_vector_store,
)


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "vs.sqlite", embed_dim=4)
    yield db
    db.close()


def _seed(
    conn,
    *,
    kind,
    lang,
    text,
    vec,
    repo="me/x",
    author=None,
    node_kind=None,
    symbol_name=None,
):
    aid = q.upsert_artifact(
        conn,
        kind="commit" if kind == "code" else "review_comment",
        external_id=f"{kind}-{text}",
        source_url=None,
        repo=repo,
        language=lang,
        author_email=None,
        author_login=author,
        created_at=None,
        decision=None,
        meta=None,
    )
    context: dict = {"language": lang}
    if node_kind is not None:
        context["node_kind"] = node_kind
    if symbol_name is not None:
        context["symbol_name"] = symbol_name
    cid = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind=kind,
        text=text,
        context=context,
        language=lang,
    )
    q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="t")
    return cid


def test_sqlite_vec_store_matches_queries_vector_search(conn):
    """SqliteVecStore is a thin wrapper — its results must be identical to
    calling q.vector_search directly."""
    _seed(conn, kind="code", lang="python", text="a", vec=[1.0, 0.0, 0.0, 0.0])
    _seed(conn, kind="code", lang="python", text="b", vec=[0.0, 1.0, 0.0, 0.0])
    _seed(conn, kind="code", lang="python", text="c", vec=[0.0, 0.0, 1.0, 0.0])

    store = SqliteVecStore(conn)
    via_store = store.search(
        [0.95, 0.05, 0.0, 0.0],
        filters=VectorSearchFilters(chunk_kind="code"),
        k=2,
    )
    via_q = q.vector_search(conn, query_vec=[0.95, 0.05, 0.0, 0.0], chunk_kind="code", k=2)
    assert [h.chunk_id for h in via_store] == [h.chunk_id for h in via_q]


def test_sqlite_vec_store_honors_filters(conn):
    _seed(conn, kind="code", lang="python", text="py", vec=[1.0, 0.0, 0.0, 0.0])
    _seed(conn, kind="code", lang="go", text="go", vec=[1.0, 0.0, 0.0, 0.0])
    store = SqliteVecStore(conn)
    hits = store.search(
        [1.0, 0.0, 0.0, 0.0],
        filters=VectorSearchFilters(chunk_kind="code", language="python"),
        k=5,
    )
    assert {h.text for h in hits} == {"py"}


def test_sqlite_vec_store_honors_node_kind_filter(conn):
    """`VectorSearchFilters.node_kind` should narrow the candidate set to
    chunks whose stored node_kind matches."""
    _seed(
        conn,
        kind="code",
        lang="python",
        text="fn",
        vec=[1.0, 0.0, 0.0, 0.0],
        node_kind="function_definition",
        symbol_name="run",
    )
    _seed(
        conn,
        kind="code",
        lang="python",
        text="cls",
        vec=[1.0, 0.0, 0.0, 0.0],
        node_kind="class_definition",
        symbol_name="Run",
    )
    _seed(
        conn,
        kind="code",
        lang="python",
        text="window",
        vec=[1.0, 0.0, 0.0, 0.0],
    )
    store = SqliteVecStore(conn)
    hits = store.search(
        [1.0, 0.0, 0.0, 0.0],
        filters=VectorSearchFilters(chunk_kind="code", node_kind="function_definition"),
        k=10,
    )
    assert {h.text for h in hits} == {"fn"}


def test_make_vector_store_default_is_sqlite_vec(conn):
    store = make_vector_store(conn, backend="sqlite-vec", dim=4)
    assert isinstance(store, SqliteVecStore)
    assert store.backend_id == "sqlite-vec"


def test_make_vector_store_unknown_backend_raises(conn):
    with pytest.raises(ValueError, match="Unknown vector store backend"):
        make_vector_store(conn, backend="bogus", dim=4)


# ---------- FAISS path (skipped if dep missing) ----------


def _require_faiss():
    pytest.importorskip("faiss")


def test_faiss_store_agrees_with_sqlite_vec_on_ranking(conn):
    """FAISS L2 over the same vectors should pick the same top-k chunk IDs
    as sqlite-vec's L2 (the default metric for vec0)."""
    _require_faiss()
    _seed(conn, kind="code", lang="python", text="a", vec=[1.0, 0.0, 0.0, 0.0])
    _seed(conn, kind="code", lang="python", text="b", vec=[0.0, 1.0, 0.0, 0.0])
    _seed(conn, kind="code", lang="python", text="c", vec=[0.0, 0.0, 1.0, 0.0])

    faiss_store = make_vector_store(conn, backend="faiss", dim=4)
    sqlite_store = SqliteVecStore(conn)
    filt = VectorSearchFilters(chunk_kind="code")
    q_vec = [0.95, 0.05, 0.0, 0.0]
    a = [h.chunk_id for h in faiss_store.search(q_vec, filters=filt, k=2)]
    b = [h.chunk_id for h in sqlite_store.search(q_vec, filters=filt, k=2)]
    assert a == b


def test_faiss_store_respects_filters(conn):
    _require_faiss()
    _seed(conn, kind="code", lang="python", text="py", vec=[1.0, 0.0, 0.0, 0.0])
    _seed(conn, kind="code", lang="go", text="go", vec=[1.0, 0.0, 0.0, 0.0])
    store = make_vector_store(conn, backend="faiss", dim=4)
    hits = store.search(
        [1.0, 0.0, 0.0, 0.0],
        filters=VectorSearchFilters(chunk_kind="code", language="python"),
        k=5,
    )
    assert {h.text for h in hits} == {"py"}


def test_faiss_store_empty_index_returns_empty(conn):
    _require_faiss()
    store = make_vector_store(conn, backend="faiss", dim=4)
    hits = store.search(
        [1.0, 0.0, 0.0, 0.0],
        filters=VectorSearchFilters(chunk_kind="code"),
        k=5,
    )
    assert hits == []
