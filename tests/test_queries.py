from pathlib import Path

import pytest

from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    seed_target(db)
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
    target_id=1,
):
    aid = q.upsert_artifact(
        conn,
        target_id=target_id,
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
    """Regression: when one kind dominates the index, the rare kind must still
    be retrievable."""
    for i in range(50):
        _seed_chunk(
            conn,
            kind="code",
            language="python",
            text=f"code-{i}",
            vec=[1.0, 0.01 * i, 0.0, 0.0],
        )
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
        target_id=1,
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
        target_id=1,
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
        target_id=1,
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
    # Global cursor (default target_id=0).
    assert q.get_cursor(conn, "commits") is None
    q.set_cursor(conn, "commits", "2024-01-01T00:00:00+00:00")
    assert q.get_cursor(conn, "commits") == "2024-01-01T00:00:00+00:00"
    # Per-target cursor doesn't collide with the global one.
    assert q.get_cursor(conn, "commits", target_id=1) is None
    q.set_cursor(conn, "commits", "2024-06-01T00:00:00+00:00", target_id=1)
    assert q.get_cursor(conn, "commits", target_id=1) == "2024-06-01T00:00:00+00:00"
    assert q.get_cursor(conn, "commits") == "2024-01-01T00:00:00+00:00"


def test_target_roundtrip_user(conn):
    # The conftest fixture already seeded a user target (id=1).
    rows = q.get_all_targets(conn)
    assert len(rows) == 1
    assert rows[0]["kind"] == "user"
    # Adding a second org target appends, doesn't overwrite.
    q.upsert_target(conn, kind="org", name="typelevel", external_id=1234, emails=None)
    rows = q.get_all_targets(conn)
    assert {r["kind"] for r in rows} == {"user", "org"}


def test_target_roundtrip_org(tmp_path: Path):
    # Fresh DB (no autoseeded user).
    db = open_db(tmp_path / "org.sqlite", embed_dim=4)
    try:
        tid = q.upsert_target(db, kind="org", name="typelevel", external_id=1234, emails=None)
        row = q.get_target_by_id(db, tid)
        assert row is not None
        assert row["kind"] == "org"
        assert row["name"] == "typelevel"
        assert row["external_id"] == 1234
        assert row["emails_json"] is None
    finally:
        db.close()


def test_delete_target_handles_more_chunks_than_sqlite_var_limit(tmp_path: Path):
    """`delete_target` used to bind every chunk_id as a SQL parameter, which
    blew the SQLITE_MAX_VARIABLE_NUMBER limit on real org corpora. Lower the
    per-statement variable cap via `setlimit` so we can reproduce the bug
    without seeding tens of thousands of chunks."""
    import sqlite3 as _sqlite3

    db = open_db(tmp_path / "many.sqlite", embed_dim=4)
    try:
        # SQLITE_LIMIT_VARIABLE_NUMBER = 9. Cap at 999 — SQLite's pre-3.32
        # default — so the unbatched DELETE blows up at a tractable size and
        # the batched fix (chunks of 500) still fits safely under it.
        db.setlimit(_sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        tid = q.upsert_target(db, kind="org", name="bigorg", external_id=42, emails=None)
        aid = q.upsert_artifact(
            db,
            target_id=tid,
            kind="commit",
            external_id="sha-many",
            source_url=None,
            repo="bigorg/lib",
            language="python",
            author_email=None,
            created_at="2024-01-01",
            decision=None,
            meta=None,
        )
        n = 1200
        for i in range(n):
            cid = q.insert_chunk(
                db,
                artifact_id=aid,
                kind="code",
                text=f"chunk-{i}",
                context={"language": "python"},
                language="python",
            )
            q.write_embedding(db, chunk_id=cid, embedding=[1.0, 0.0, 0.0, 0.0], model_id="m")
        assert db.execute("SELECT COUNT(*) AS n FROM vec_chunk").fetchone()["n"] == n

        q.delete_target(db, tid)

        assert q.get_target_by_id(db, tid) is None
        assert db.execute("SELECT COUNT(*) AS n FROM chunk").fetchone()["n"] == 0
        assert db.execute("SELECT COUNT(*) AS n FROM vec_chunk").fetchone()["n"] == 0
    finally:
        db.close()


def test_repo_upsert_and_list(conn):
    assert q.list_repos(conn) == []
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r1",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        archived=False,
        fork=False,
        size_kb=10,
    )
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r2",
        default_branch="main",
        pushed_at="2024-02-01T00:00:00Z",
        archived=True,
        fork=False,
        size_kb=20,
    )
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r3",
        default_branch="main",
        pushed_at="2024-03-01T00:00:00Z",
        archived=False,
        fork=True,
        size_kb=30,
    )
    repos = q.list_repos(conn)
    assert [r["full_name"] for r in repos] == ["org/r1"]
    repos = q.list_repos(conn, include_archived=True)
    assert {r["full_name"] for r in repos} == {"org/r1", "org/r2"}
    repos = q.list_repos(conn, include_archived=True, include_forks=True)
    assert {r["full_name"] for r in repos} == {"org/r1", "org/r2", "org/r3"}


def test_repo_visibility_round_trips(conn):
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/internal",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        visibility="internal",
    )
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/legacy",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        # visibility omitted — defaults to None / NULL (older GHE responses).
    )
    row_internal = q.get_repo(conn, target_id=1, full_name="org/internal")
    assert row_internal is not None
    assert row_internal["visibility"] == "internal"
    row_legacy = q.get_repo(conn, target_id=1, full_name="org/legacy")
    assert row_legacy is not None
    assert row_legacy["visibility"] is None


def test_repo_upsert_refreshes_archived_and_visibility(conn):
    """Sync re-upserts with fresh GitHub state; existing row's columns flip."""
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
        archived=False,
        visibility="public",
    )
    # `gt sync` refresh sees the repo is now archived and internal.
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-05-01T00:00:00Z",
        archived=True,
        visibility="internal",
    )
    row = q.get_repo(conn, target_id=1, full_name="org/r")
    assert row is not None
    assert row["archived"] == 1
    assert row["visibility"] == "internal"
    # `list_repos` default now excludes it.
    assert [r["full_name"] for r in q.list_repos(conn)] == []
    assert [r["full_name"] for r in q.list_repos(conn, include_archived=True)] == ["org/r"]


def test_set_repo_cursor_is_partial(conn):
    q.upsert_repo(
        conn,
        target_id=1,
        full_name="org/r",
        default_branch="main",
        pushed_at="2024-01-01T00:00:00Z",
    )
    q.set_repo_cursor(conn, target_id=1, full_name="org/r", files_at="2024-01-02T00:00:00Z")
    r = q.get_repo(conn, target_id=1, full_name="org/r")
    assert r["last_files_at"] == "2024-01-02T00:00:00Z"
    assert r["last_commits_at"] is None
    q.set_repo_cursor(conn, target_id=1, full_name="org/r", commits_at="2024-01-03T00:00:00Z")
    r = q.get_repo(conn, target_id=1, full_name="org/r")
    assert r["last_files_at"] == "2024-01-02T00:00:00Z"
    assert r["last_commits_at"] == "2024-01-03T00:00:00Z"


def test_open_db_migrates_repo_visibility_column(tmp_path: Path):
    """An old DB created before the `visibility` column gets it back-filled
    as NULL on the next open, with existing rows untouched."""
    db_path = tmp_path / "legacy.sqlite"
    db = open_db(db_path, embed_dim=4)
    try:
        # Drop visibility to simulate a pre-migration DB, then re-open.
        db.execute(
            "CREATE TABLE repo_legacy AS "
            "SELECT target_id, full_name, default_branch, head_sha, pushed_at, "
            "archived, fork, size_kb, last_files_at, last_commits_at, "
            "last_commits_walked_sha, last_reviews_at FROM repo"
        )
        db.execute("DROP TABLE repo")
        db.execute(
            "CREATE TABLE repo ("
            "target_id INTEGER NOT NULL REFERENCES target(id) ON DELETE CASCADE,"
            "full_name TEXT NOT NULL,"
            "default_branch TEXT,"
            "head_sha TEXT,"
            "pushed_at TEXT,"
            "archived INTEGER NOT NULL DEFAULT 0,"
            "fork INTEGER NOT NULL DEFAULT 0,"
            "size_kb INTEGER,"
            "last_files_at TEXT,"
            "last_commits_at TEXT,"
            "last_commits_walked_sha TEXT,"
            "last_reviews_at TEXT,"
            "PRIMARY KEY (target_id, full_name))"
        )
        seed_target(db)
        db.execute(
            "INSERT INTO repo (target_id, full_name, default_branch, pushed_at) "
            "VALUES (1, 'org/legacy', 'main', '2024-01-01T00:00:00Z')"
        )
    finally:
        db.close()

    db = open_db(db_path, embed_dim=4)
    try:
        cols = {row["name"] for row in db.execute("PRAGMA table_info(repo)").fetchall()}
        assert "visibility" in cols
        row = db.execute("SELECT visibility FROM repo WHERE full_name = 'org/legacy'").fetchone()
        assert row["visibility"] is None
    finally:
        db.close()


def test_open_db_rejects_wrong_dim(tmp_path: Path):
    db_path = tmp_path / "dim.sqlite"
    open_db(db_path, embed_dim=4).close()
    with pytest.raises(RuntimeError, match="different embedding dimension"):
        open_db(db_path, embed_dim=8)


def test_coalesce_dedupes_same_external_id_across_targets(tmp_path: Path):
    """A commit ingested under two targets surfaces once in coalesce mode."""
    db = open_db(tmp_path / "multi.sqlite", embed_dim=4)
    try:
        t1 = seed_target(db, kind="user", name="me", external_id=1)
        t2 = seed_target(db, kind="org", name="acme", external_id=2)
        for tid in (t1, t2):
            aid = q.upsert_artifact(
                db,
                target_id=tid,
                kind="commit",
                external_id="shared-sha",
                source_url=None,
                repo="acme/lib",
                language="python",
                author_email=None,
                author_login="me",
                created_at="2024-01-01",
                decision=None,
                meta=None,
            )
            cid = q.insert_chunk(
                db,
                artifact_id=aid,
                kind="code",
                text="shared body",
                context={"language": "python"},
                language="python",
            )
            q.write_embedding(db, chunk_id=cid, embedding=[1.0, 0.0, 0.0, 0.0], model_id="m")
        # Coalesce mode (no target_id) dedupes.
        hits = q.vector_search(db, query_vec=[1.0, 0.0, 0.0, 0.0], chunk_kind="code", k=5)
        assert len(hits) == 1
        # Narrow modes return only that target's row.
        hits_t1 = q.vector_search(
            db, query_vec=[1.0, 0.0, 0.0, 0.0], chunk_kind="code", target_id=t1, k=5
        )
        hits_t2 = q.vector_search(
            db, query_vec=[1.0, 0.0, 0.0, 0.0], chunk_kind="code", target_id=t2, k=5
        )
        assert len(hits_t1) == 1 and hits_t1[0].target_id == t1
        assert len(hits_t2) == 1 and hits_t2[0].target_id == t2
    finally:
        db.close()
