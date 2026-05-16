from pathlib import Path

import pytest

from github_twin.store import queries as q
from github_twin.store.db import open_db


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    yield db
    db.close()


def _seed_chunk(
    conn,
    *,
    kind,
    language,
    text,
    vec,
    decision=None,
    repo="me/x",
    author_login=None,
    node_kind=None,
    symbol_name=None,
):
    aid = q.upsert_artifact(
        conn,
        kind="commit" if kind == "code" else "review_comment",
        external_id=f"{kind}-{text}",
        source_url=None,
        repo=repo,
        language=language,
        author_email="me@e.com",
        author_login=author_login,
        created_at="2024-01-01",
        decision=decision,
        meta=None,
    )
    context: dict = {"language": language}
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
        language=language,
    )
    q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="test-embedder")
    return aid, cid


def test_vector_search_returns_nearest_first(conn):
    _seed_chunk(conn, kind="code", language="python", text="aaa", vec=[1.0, 0.0, 0.0, 0.0])
    _seed_chunk(conn, kind="code", language="python", text="bbb", vec=[0.0, 1.0, 0.0, 0.0])
    _seed_chunk(conn, kind="code", language="python", text="ccc", vec=[0.0, 0.0, 1.0, 0.0])

    hits = q.vector_search(conn, query_vec=[0.95, 0.05, 0.0, 0.0], chunk_kind="code", k=2)
    assert [h.text for h in hits] == ["aaa", "bbb"]
    assert hits[0].distance < hits[1].distance


def test_vector_search_language_filter(conn):
    _seed_chunk(conn, kind="code", language="python", text="py-near", vec=[1.0, 0.0, 0.0, 0.0])
    _seed_chunk(conn, kind="code", language="go", text="go-nearer", vec=[0.99, 0.01, 0.0, 0.0])

    hits_py = q.vector_search(
        conn, query_vec=[1.0, 0.0, 0.0, 0.0], chunk_kind="code", language="python", k=5
    )
    assert {h.text for h in hits_py} == {"py-near"}

    hits_any = q.vector_search(conn, query_vec=[1.0, 0.0, 0.0, 0.0], chunk_kind="code", k=5)
    assert {h.text for h in hits_any} == {"py-near", "go-nearer"}


def test_vector_search_repo_filter(conn):
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="repo-a",
        vec=[1.0, 0.0, 0.0, 0.0],
        repo="org/a",
    )
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="repo-b",
        vec=[1.0, 0.0, 0.0, 0.0],
        repo="org/b",
    )
    hits = q.vector_search(
        conn, query_vec=[1.0, 0.0, 0.0, 0.0], chunk_kind="code", repo="org/a", k=5
    )
    assert {h.text for h in hits} == {"repo-a"}


def test_vector_search_author_login_filter(conn):
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="alice-code",
        vec=[1.0, 0.0, 0.0, 0.0],
        author_login="alice",
    )
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="bob-code",
        vec=[1.0, 0.0, 0.0, 0.0],
        author_login="bob",
    )
    hits = q.vector_search(
        conn,
        query_vec=[1.0, 0.0, 0.0, 0.0],
        chunk_kind="code",
        author_login="alice",
        k=5,
    )
    assert {h.text for h in hits} == {"alice-code"}


def test_insert_chunk_persists_node_kind_and_symbol_name(conn):
    """`insert_chunk` should pull `node_kind` and `symbol_name` off the
    context dict and write them into the dedicated columns."""
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="def x(): ...",
        vec=[1.0, 0.0, 0.0, 0.0],
        node_kind="function_definition",
        symbol_name="x",
    )
    row = conn.execute(
        "SELECT node_kind, symbol_name FROM chunk WHERE text=?",
        ("def x(): ...",),
    ).fetchone()
    assert row["node_kind"] == "function_definition"
    assert row["symbol_name"] == "x"


def test_insert_chunk_leaves_node_kind_null_for_non_ast_chunks(conn):
    """Chunks without node_kind in context (line-window fallback, review
    comments, commit messages) should land with NULL columns."""
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="windowed-chunk",
        vec=[1.0, 0.0, 0.0, 0.0],
    )
    row = conn.execute(
        "SELECT node_kind, symbol_name FROM chunk WHERE text=?",
        ("windowed-chunk",),
    ).fetchone()
    assert row["node_kind"] is None
    assert row["symbol_name"] is None


def test_vector_search_node_kind_filter(conn):
    """node_kind filter narrows to AST-chunked rows of the requested type."""
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="fn-chunk",
        vec=[1.0, 0.0, 0.0, 0.0],
        node_kind="function_definition",
        symbol_name="run",
    )
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="class-chunk",
        vec=[1.0, 0.0, 0.0, 0.0],
        node_kind="class_definition",
        symbol_name="Run",
    )
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="window-chunk",
        vec=[1.0, 0.0, 0.0, 0.0],
    )
    hits = q.vector_search(
        conn,
        query_vec=[1.0, 0.0, 0.0, 0.0],
        chunk_kind="code",
        node_kind="function_definition",
        k=10,
    )
    assert {h.text for h in hits} == {"fn-chunk"}


def test_bm25_search_node_kind_filter(conn):
    """Same filter wired through bm25_search."""
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="needle function body here",
        vec=[1.0, 0.0, 0.0, 0.0],
        node_kind="function_definition",
        symbol_name="needle",
    )
    _seed_chunk(
        conn,
        kind="code",
        language="python",
        text="needle class body here",
        vec=[0.0, 1.0, 0.0, 0.0],
        node_kind="class_definition",
        symbol_name="Needle",
    )
    hits = q.bm25_search(
        conn,
        query_text="needle",
        chunk_kind="code",
        node_kind="function_definition",
        k=10,
    )
    assert all("function body" in h.text for h in hits)
    assert len(hits) == 1


def test_vector_search_kind_filter(conn):
    _seed_chunk(conn, kind="code", language="python", text="codey", vec=[1.0, 0.0, 0.0, 0.0])
    _seed_chunk(
        conn, kind="review_comment", language="python", text="reviewy", vec=[1.0, 0.0, 0.0, 0.0]
    )

    hits = q.vector_search(conn, query_vec=[1.0, 0.0, 0.0, 0.0], chunk_kind="review_comment", k=5)
    assert {h.text for h in hits} == {"reviewy"}


def test_vector_search_finds_rare_kind_among_many(conn):
    """Regression: when one kind dominates the index (e.g. lots of code, few
    review_comments), the rare kind must still be retrievable. Earlier the
    KNN ran unfiltered and the post-filter dropped everything."""
    # 50 'code' chunks clustered near the query
    for i in range(50):
        _seed_chunk(
            conn,
            kind="code",
            language="python",
            text=f"code-{i}",
            vec=[1.0, 0.01 * i, 0.0, 0.0],
        )
    # 2 'review_comment' chunks, slightly farther from the query
    _seed_chunk(
        conn,
        kind="review_comment",
        language="python",
        text="rare-1",
        vec=[0.6, 0.6, 0.0, 0.0],
    )
    _seed_chunk(
        conn,
        kind="review_comment",
        language="python",
        text="rare-2",
        vec=[0.5, 0.7, 0.0, 0.0],
    )

    hits = q.vector_search(conn, query_vec=[1.0, 0.0, 0.0, 0.0], chunk_kind="review_comment", k=5)
    assert {h.text for h in hits} == {"rare-1", "rare-2"}


def test_upsert_artifact_is_idempotent(conn):
    aid1 = q.upsert_artifact(
        conn,
        kind="commit",
        external_id="sha1",
        source_url=None,
        repo="r",
        language="python",
        author_email=None,
        created_at=None,
        decision=None,
        meta={"a": 1},
    )
    aid2 = q.upsert_artifact(
        conn,
        kind="commit",
        external_id="sha1",
        source_url="new",
        repo="r",
        language="python",
        author_email=None,
        created_at=None,
        decision=None,
        meta={"a": 2},
    )
    assert aid1 == aid2
    rows = conn.execute("SELECT source_url, meta_json FROM artifact WHERE id=?", (aid1,)).fetchone()
    assert rows["source_url"] == "new"
    assert '"a":2' in rows["meta_json"]


def test_delete_chunks_for_artifact_clears_vectors(conn):
    aid, cid = _seed_chunk(conn, kind="code", language="python", text="t", vec=[1.0, 0.0, 0.0, 0.0])
    assert conn.execute("SELECT COUNT(*) AS n FROM vec_chunk").fetchone()["n"] == 1
    q.delete_chunks_for_artifact(conn, aid)
    assert conn.execute("SELECT COUNT(*) AS n FROM chunk").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM vec_chunk").fetchone()["n"] == 0


def test_pending_embed_chunks_skips_already_embedded(conn):
    aid = q.upsert_artifact(
        conn,
        kind="commit",
        external_id="x",
        source_url=None,
        repo=None,
        language=None,
        author_email=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    cid_embedded = q.insert_chunk(conn, artifact_id=aid, kind="code", text="done", context=None)
    q.write_embedding(conn, chunk_id=cid_embedded, embedding=[1.0, 0, 0, 0], model_id="m")
    cid_pending = q.insert_chunk(conn, artifact_id=aid, kind="code", text="todo", context=None)

    pending = list(q.pending_embed_chunks(conn))
    assert [c.id for c in pending] == [cid_pending]


def test_cursor_roundtrip(conn):
    assert q.get_cursor(conn, "commits") is None
    q.set_cursor(conn, "commits", "2024-01-01T00:00:00+00:00")
    assert q.get_cursor(conn, "commits") == "2024-01-01T00:00:00+00:00"
    q.set_cursor(conn, "commits", "2024-06-01T00:00:00+00:00")
    assert q.get_cursor(conn, "commits") == "2024-06-01T00:00:00+00:00"


def test_target_roundtrip_user(conn):
    assert q.get_target(conn) is None
    q.upsert_target(conn, kind="user", name="me", external_id=42, emails=["a@b", "c@d"])
    t = q.get_target(conn)
    assert t["kind"] == "user"
    assert t["name"] == "me"
    assert t["external_id"] == 42
    assert t["emails"] == ["a@b", "c@d"]


def test_target_roundtrip_org(conn):
    q.upsert_target(conn, kind="org", name="typelevel", external_id=1234, emails=None)
    t = q.get_target(conn)
    assert t["kind"] == "org"
    assert t["name"] == "typelevel"
    assert t["external_id"] == 1234
    assert t["emails"] is None


def test_repo_upsert_and_list(conn):
    assert q.list_repos(conn) == []
    q.upsert_repo(
        conn,
        full_name="org/r1",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        archived=False,
        fork=False,
        size_kb=10,
    )
    q.upsert_repo(
        conn,
        full_name="org/r2",
        default_branch="main",
        pushed_at="2024-02-01T00:00:00Z",
        archived=True,
        fork=False,
        size_kb=20,
    )
    q.upsert_repo(
        conn,
        full_name="org/r3",
        default_branch="main",
        pushed_at="2024-03-01T00:00:00Z",
        archived=False,
        fork=True,
        size_kb=30,
    )
    # Default filters: no archived, no forks.
    repos = q.list_repos(conn)
    assert [r["full_name"] for r in repos] == ["org/r1"]
    # With archived included.
    repos = q.list_repos(conn, include_archived=True)
    assert {r["full_name"] for r in repos} == {"org/r1", "org/r2"}
    # Everything.
    repos = q.list_repos(conn, include_archived=True, include_forks=True)
    assert {r["full_name"] for r in repos} == {"org/r1", "org/r2", "org/r3"}


def test_set_repo_cursor_is_partial(conn):
    q.upsert_repo(
        conn,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
    )
    q.set_repo_cursor(conn, full_name="org/r", files_at="2024-01-02T00:00:00Z")
    r = q.get_repo(conn, "org/r")
    assert r["last_files_at"] == "2024-01-02T00:00:00Z"
    assert r["last_commits_at"] is None  # unchanged
    # Subsequent partial update doesn't clobber files_at
    q.set_repo_cursor(conn, full_name="org/r", commits_at="2024-01-03T00:00:00Z")
    r = q.get_repo(conn, "org/r")
    assert r["last_files_at"] == "2024-01-02T00:00:00Z"
    assert r["last_commits_at"] == "2024-01-03T00:00:00Z"


def test_open_db_rejects_wrong_dim(tmp_path: Path):
    db_path = tmp_path / "dim.sqlite"
    open_db(db_path, embed_dim=4).close()
    with pytest.raises(RuntimeError, match="different embedding dimension"):
        open_db(db_path, embed_dim=8)
