"""Migration from the legacy `identity` singleton to the kind-discriminated
`target` row. The pre-schema migration in `db._run_pre_schema_migrations` must:
  1. Copy the lone identity row into target with kind='user'.
  2. Drop the identity table.
  3. Be a no-op on second open.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

from github_twin.store.db import open_db


def _seed_legacy_identity(db_path: Path) -> None:
    """Mimic the pre-O-A schema: an `identity` table with one row."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(
        """
        CREATE TABLE identity (
          id            INTEGER PRIMARY KEY CHECK (id = 1),
          username      TEXT NOT NULL,
          user_id       INTEGER NOT NULL,
          emails_json   TEXT NOT NULL,
          discovered_at TEXT NOT NULL
        );
        INSERT INTO identity (id, username, user_id, emails_json, discovered_at)
        VALUES (1, 'me', 42, '["a@b.com", "c@d.com"]', '2024-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()


def test_identity_migrates_to_target(tmp_path: Path):
    db_path = tmp_path / "legacy.sqlite"
    _seed_legacy_identity(db_path)

    conn = open_db(db_path, embed_dim=4)
    try:
        # Identity table is gone.
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='identity'"
        ).fetchone()
        assert row is None

        # Target row carries the legacy data over with kind='user'.
        t = conn.execute(
            "SELECT kind, name, external_id, emails_json, discovered_at FROM target"
        ).fetchone()
        assert t["kind"] == "user"
        assert t["name"] == "me"
        assert t["external_id"] == 42
        assert t["emails_json"] == '["a@b.com", "c@d.com"]'
        assert t["discovered_at"] == "2024-01-01T00:00:00+00:00"
    finally:
        conn.close()


def test_migration_is_idempotent_on_reopen(tmp_path: Path):
    db_path = tmp_path / "legacy.sqlite"
    _seed_legacy_identity(db_path)
    conn = open_db(db_path, embed_dim=4)
    conn.close()
    # Second open must not raise and must leave target unchanged.
    conn = open_db(db_path, embed_dim=4)
    try:
        assert conn.execute("SELECT COUNT(*) FROM target").fetchone()[0] == 1
    finally:
        conn.close()


def test_artifact_author_login_column_added(tmp_path: Path):
    """A pre-O-A DB also lacks artifact.author_login. The migration adds it."""
    db_path = tmp_path / "pre_oa.sqlite"
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # Old artifact shape — no author_login column.
    conn.executescript(
        """
        CREATE TABLE artifact (
          id INTEGER PRIMARY KEY,
          kind TEXT NOT NULL,
          external_id TEXT,
          source_url TEXT,
          repo TEXT,
          language TEXT,
          author_email TEXT,
          created_at TEXT,
          decision TEXT,
          meta_json TEXT,
          UNIQUE(kind, external_id)
        );
        """
    )
    conn.commit()
    conn.close()

    conn = open_db(db_path, embed_dim=4)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(artifact)").fetchall()}
        assert "author_login" in cols
    finally:
        conn.close()


def test_fresh_db_creates_target_table_without_legacy(tmp_path: Path):
    """When there's no legacy identity table, schema.sql still creates target."""
    db_path = tmp_path / "fresh.sqlite"
    conn = open_db(db_path, embed_dim=4)
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "target" in tables
        assert "repo" in tables
        assert "identity" not in tables
        # No rows in target on a fresh DB.
        assert conn.execute("SELECT COUNT(*) FROM target").fetchone()[0] == 0
    finally:
        conn.close()
