"""chunk_fts stays in sync with chunk via triggers, and open_db backfills
pre-existing chunk rows on first open after this migration lands.
"""

import sqlite3
from pathlib import Path

import pytest

from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "fts.sqlite", embed_dim=4)
    seed_target(db)
    yield db
    db.close()


def _insert(conn, *, text):
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="commit",
        external_id=f"x-{text}",
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
        context=None,
        language="python",
    )
    return aid, cid


def _fts_text(conn, cid):
    row = conn.execute("SELECT text FROM chunk_fts WHERE rowid=?", (cid,)).fetchone()
    return row["text"] if row else None


def test_insert_trigger_populates_fts(conn):
    _, cid = _insert(conn, text="hello world identifier")
    assert _fts_text(conn, cid) == "hello world identifier"


def test_update_trigger_refreshes_fts(conn):
    _, cid = _insert(conn, text="old_token here")
    conn.execute("UPDATE chunk SET text=? WHERE id=?", ("new_token there", cid))
    assert _fts_text(conn, cid) == "new_token there"


def test_delete_trigger_clears_fts(conn):
    _, cid = _insert(conn, text="will_be_deleted")
    conn.execute("DELETE FROM chunk WHERE id=?", (cid,))
    assert _fts_text(conn, cid) is None


def test_delete_chunks_for_artifact_clears_fts(conn):
    aid, cid = _insert(conn, text="will_be_purged")
    q.delete_chunks_for_artifact(conn, aid)
    assert _fts_text(conn, cid) is None
    hits = q.bm25_search(conn, query_text="will_be_purged", chunk_kind="code", k=5)
    assert hits == []


def test_open_db_backfills_fts_from_existing_chunks(tmp_path: Path):
    """Simulate a DB written before the FTS5 migration: chunks present but
    chunk_fts empty (or absent). open_db must backfill."""
    db_path = tmp_path / "preexisting.sqlite"

    # Seed with the normal pipeline.
    conn = open_db(db_path, embed_dim=4)
    seed_target(conn)
    _insert(conn, text="alpha bravo charlie")
    _insert(conn, text="delta echo foxtrot")
    conn.close()

    # Simulate pre-migration state: drop the triggers and the FTS5 table.
    raw = sqlite3.connect(db_path)
    raw.executescript(
        "DROP TRIGGER IF EXISTS chunk_ai; "
        "DROP TRIGGER IF EXISTS chunk_au; "
        "DROP TRIGGER IF EXISTS chunk_ad; "
        "DROP TABLE IF EXISTS chunk_fts;"
    )
    raw.close()

    # Re-open: schema.sql re-creates chunk_fts (empty), then _backfill_fts
    # populates it from existing chunk rows.
    conn = open_db(db_path, embed_dim=4)
    try:
        n_fts = conn.execute("SELECT COUNT(*) AS n FROM chunk_fts").fetchone()["n"]
        n_chunk = conn.execute("SELECT COUNT(*) AS n FROM chunk").fetchone()["n"]
        assert n_fts == n_chunk == 2

        hits = q.bm25_search(conn, query_text="bravo", chunk_kind="code", k=5)
        assert len(hits) == 1
    finally:
        conn.close()


def test_open_db_backfill_is_idempotent(tmp_path: Path):
    """Reopening a DB that already has chunk_fts populated does nothing."""
    db_path = tmp_path / "idempotent.sqlite"
    conn = open_db(db_path, embed_dim=4)
    seed_target(conn)
    _insert(conn, text="some content")
    n_before = conn.execute("SELECT COUNT(*) AS n FROM chunk_fts").fetchone()["n"]
    conn.close()

    conn = open_db(db_path, embed_dim=4)
    try:
        n_after = conn.execute("SELECT COUNT(*) AS n FROM chunk_fts").fetchone()["n"]
        assert n_after == n_before == 1
    finally:
        conn.close()
