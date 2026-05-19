"""Retrieval tools raise NeedsInitError when the DB has nothing to search.

The five `find_*` / `predict_*` retrieval tools previously returned `[]`
(or the empty-shape predict dict) against an uninitialized DB, leaving
the MCP client to guess whether the empty result meant "no match" or
"this repo isn't set up". `_require_data` distinguishes the two cases
and `NeedsInitError` surfaces the structured hint into the protocol so
the client can take the obvious next action (`bootstrap` or the CLI
fallback).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from github_twin.mcp_server import tools as mtools
from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "needs_init.sqlite", embed_dim=4)
    yield db
    db.close()


def _seed_vector(conn: sqlite3.Connection, *, target_id: int) -> int:
    """Insert one artifact + chunk + vec row so `_require_data` accepts."""
    aid = q.upsert_artifact(
        conn,
        target_id=target_id,
        kind="commit",
        external_id="c-0",
        source_url=None,
        repo="up/x",
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
        text="def f(): return 1",
        context={"path": "a.py"},
        language="python",
    )
    q.write_embedding(conn, chunk_id=cid, embedding=[1.0, 0.0, 0.0, 0.0], model_id="fake")
    return cid


# ---------- _require_data ----------


def test_require_data_raises_when_no_target(conn: sqlite3.Connection):
    with pytest.raises(mtools.NeedsInitError, match="no target configured"):
        mtools._require_data(conn)


def test_require_data_raises_when_target_but_no_vectors(conn: sqlite3.Connection):
    seed_target(conn, kind="repo", name="up/x", external_id=999)
    with pytest.raises(mtools.NeedsInitError, match="no chunks are embedded"):
        mtools._require_data(conn)


def test_require_data_passes_when_vectors_present(conn: sqlite3.Connection):
    tid = seed_target(conn, kind="repo", name="up/x", external_id=999)
    _seed_vector(conn, target_id=tid)
    mtools._require_data(conn)  # no raise


# ---------- retrieval tools propagate NeedsInitError ----------


class _StubEmbedder:
    dim = 4
    model_id = "stub"

    def embed(self, texts):
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


class _StubStore:
    """Never actually searched — the precheck raises before we get here."""

    def search(self, *_args, **_kw):  # pragma: no cover
        raise AssertionError("retrieval should not be reached when DB is empty")


@pytest.mark.parametrize(
    "call",
    [
        lambda c: mtools.find_review_comments(
            c, _StubEmbedder(), _StubStore(), diff_hunk="def f(): pass"
        ),
        lambda c: mtools.find_style_examples(c, _StubEmbedder(), _StubStore(), query="hello"),
        lambda c: mtools.find_code(c, _StubEmbedder(), _StubStore(), query="hello"),
        lambda c: mtools.find_applicable_rules(c, _StubEmbedder(), _StubStore(), query="hello"),
        lambda c: mtools.predict_review_outcome(
            c, _StubEmbedder(), _StubStore(), diff_or_summary="def f(): pass"
        ),
    ],
)
def test_retrieval_tools_raise_needs_init_on_empty_db(conn: sqlite3.Connection, call):
    with pytest.raises(mtools.NeedsInitError):
        call(conn)
