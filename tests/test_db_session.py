"""`db_session` context manager — used by the MCP server.

The CLI commands deliberately don't use this — they're short-lived
processes and Python's sqlite3 finalizer handles cleanup at exit. But
the MCP server holds its connection for hours and needs a guaranteed
clean close when FastMCP returns (Claude Code disconnects, etc.).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.store.db import db_session


def test_db_session_yields_open_connection(tmp_path: Path):
    with db_session(tmp_path / "live.sqlite", embed_dim=4) as conn:
        cur = conn.execute("SELECT 1 AS v")
        assert cur.fetchone()["v"] == 1


def test_db_session_closes_on_normal_exit(tmp_path: Path):
    """After the `with` block exits, the connection is closed —
    subsequent operations on it raise `ProgrammingError`."""
    with db_session(tmp_path / "live.sqlite", embed_dim=4) as conn:
        pass  # exits normally
    import sqlite3

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_db_session_closes_on_exception(tmp_path: Path):
    """`db_session` must close even if the wrapped block raises —
    otherwise the SQLite file stays locked under WAL until process exit."""
    import sqlite3

    conn_ref = None
    with (
        pytest.raises(RuntimeError, match="boom"),
        db_session(tmp_path / "live.sqlite", embed_dim=4) as conn,
    ):
        conn_ref = conn
        raise RuntimeError("boom")
    assert conn_ref is not None
    with pytest.raises(sqlite3.ProgrammingError):
        conn_ref.execute("SELECT 1")


def test_db_session_swallows_close_errors(monkeypatch: pytest.MonkeyPatch):
    """An error during `close()` must NOT propagate up — the surrounding
    code already finished its work and we don't want a teardown failure
    to mask a successful run.

    `sqlite3.Connection` is a C type whose `close` is read-only, so we
    swap `open_db` for a stub that returns a fake whose `close()` raises."""

    class _FakeConn:
        def close(self) -> None:
            raise RuntimeError("close blew up")

    from github_twin.store import db as db_mod

    monkeypatch.setattr(db_mod, "open_db", lambda *a, **kw: _FakeConn())

    with db_session(Path("/unused"), embed_dim=4):
        pass
    # If we reached this line, the close-time error was suppressed.
