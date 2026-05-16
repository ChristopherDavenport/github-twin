"""BM25 keyword search over chunk_fts.

Parallel to the vector_search tests in test_queries.py: same _seed_chunk
helper shape, same filter-parity assertions, plus adversarial coverage of
_fts_escape so user queries with FTS5 metacharacters don't blow up.
"""

from pathlib import Path

import pytest

from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.store.queries import _fts_escape


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    yield db
    db.close()


_SEED_COUNTER = {"n": 0}


def _seed_chunk(
    conn,
    *,
    kind,
    language,
    text,
    repo="me/x",
    author_login=None,
):
    # Unique external_id per seed so identical text in different repos/authors
    # produces distinct artifacts (upsert_artifact keys on (kind, external_id)).
    _SEED_COUNTER["n"] += 1
    aid = q.upsert_artifact(
        conn,
        kind="commit" if kind == "code" else "review_comment",
        external_id=f"{kind}-{_SEED_COUNTER['n']}",
        source_url=None,
        repo=repo,
        language=language,
        author_email=None,
        author_login=author_login,
        created_at=None,
        decision=None,
        meta=None,
    )
    cid = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind=kind,
        text=text,
        context={"language": language},
        language=language,
    )
    return aid, cid


def test_bm25_returns_lexical_match(conn):
    _seed_chunk(
        conn, kind="code", language="python", text="def getUserById(uid): return db.find(uid)"
    )
    _seed_chunk(conn, kind="code", language="python", text="def listProducts(): return store.all()")
    hits = q.bm25_search(conn, query_text="getUserById", chunk_kind="code", k=5)
    assert [h.text.startswith("def getUserById") for h in hits] == [True]


def test_bm25_keeps_snake_case_as_one_token(conn):
    """tokenchars '_' is the load-bearing part of the tokenizer config."""
    _seed_chunk(conn, kind="code", language="python", text="user_session_id = generate_token()")
    _seed_chunk(conn, kind="code", language="python", text="path = os.environ.get('HOME')")
    hits = q.bm25_search(conn, query_text="user_session_id", chunk_kind="code", k=5)
    assert {h.text for h in hits} == {"user_session_id = generate_token()"}


def test_bm25_language_filter(conn):
    _seed_chunk(conn, kind="code", language="python", text="def foo_widget(): pass")
    _seed_chunk(conn, kind="code", language="go", text="func fooWidget() {}")
    hits = q.bm25_search(conn, query_text="foo_widget", chunk_kind="code", language="python", k=5)
    assert {h.text for h in hits} == {"def foo_widget(): pass"}


def test_bm25_repo_filter(conn):
    _seed_chunk(conn, kind="code", language="python", text="widget_a token", repo="org/a")
    _seed_chunk(conn, kind="code", language="python", text="widget_a token", repo="org/b")
    hits = q.bm25_search(conn, query_text="widget_a", chunk_kind="code", repo="org/a", k=5)
    assert [h.artifact_repo for h in hits] == ["org/a"]


def test_bm25_author_filter(conn):
    _seed_chunk(conn, kind="code", language="python", text="alpha beta gamma", author_login="alice")
    _seed_chunk(conn, kind="code", language="python", text="alpha beta gamma", author_login="bob")
    hits = q.bm25_search(conn, query_text="alpha", chunk_kind="code", author_login="alice", k=5)
    assert len(hits) == 1


def test_bm25_kind_filter(conn):
    _seed_chunk(conn, kind="code", language="python", text="shared keyword foo")
    _seed_chunk(conn, kind="review_comment", language="python", text="shared keyword foo")
    hits = q.bm25_search(conn, query_text="shared", chunk_kind="review_comment", k=5)
    assert [h.artifact_kind for h in hits] == ["review_comment"]


def test_bm25_empty_query_returns_no_results(conn):
    _seed_chunk(conn, kind="code", language="python", text="anything at all")
    hits = q.bm25_search(conn, query_text="   ", chunk_kind="code", k=5)
    assert hits == []


@pytest.mark.parametrize(
    "raw",
    [
        "foo: bar",  # colon is FTS5 column-filter syntax
        "x AND y",  # reserved operator
        "x OR y",
        "NEAR(a b)",
        "((( unbalanced",
        'say "hello" world',  # embedded double quotes
        "*wildcard*",
        "ends-with-dash-",
    ],
)
def test_fts_escape_defangs_adversarial_input(conn, raw):
    """Without _fts_escape these would raise OperationalError."""
    _seed_chunk(conn, kind="code", language="python", text="placeholder content")
    # Just assert it doesn't raise; result set can be empty.
    q.bm25_search(conn, query_text=raw, chunk_kind="code", k=5)


def test_fts_escape_quotes_each_token():
    assert _fts_escape("hello world") == '"hello" "world"'
    assert _fts_escape('say "hi"') == '"say" """hi"""'
    assert _fts_escape("") == '""'
    assert _fts_escape("   ") == '""'
