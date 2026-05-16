"""Token persistence with keyring + file fallback.

The keyring layer is exercised via a fake in-memory keyring backend
(set via `keyring.set_keyring`). The fallback path is exercised by
patching the keyring module functions to raise.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import keyring
import keyring.backend
import keyring.errors
import pytest

from github_twin.ingest import auth_storage


class _MemKeyring(keyring.backend.KeyringBackend):
    """Trivial in-memory backend used to make keyring deterministic.

    Real backends touch the OS keychain / D-Bus; we never want that
    side effect in tests."""

    priority = 1.0  # type: ignore[assignment]

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        key = (service, username)
        if key not in self._store:
            raise keyring.errors.PasswordDeleteError(f"{service}/{username}")
        del self._store[key]


class _BrokenKeyring(keyring.backend.KeyringBackend):
    """Backend that simulates D-Bus unavailability."""

    priority = 1.0  # type: ignore[assignment]

    def get_password(self, service: str, username: str) -> str | None:
        raise RuntimeError("no D-Bus")

    def set_password(self, service: str, username: str, password: str) -> None:
        raise RuntimeError("no D-Bus")

    def delete_password(self, service: str, username: str) -> None:
        raise RuntimeError("no D-Bus")


@pytest.fixture
def mem_keyring() -> _MemKeyring:
    """Install the in-memory backend and restore the original afterwards."""
    original = keyring.get_keyring()
    backend = _MemKeyring()
    keyring.set_keyring(backend)
    yield backend
    keyring.set_keyring(original)


@pytest.fixture
def broken_keyring() -> _BrokenKeyring:
    original = keyring.get_keyring()
    backend = _BrokenKeyring()
    keyring.set_keyring(backend)
    yield backend
    keyring.set_keyring(original)


# ---------- keyring happy path ----------


def test_store_and_load_via_keyring(mem_keyring: _MemKeyring, tmp_path: Path) -> None:
    kind = auth_storage.store_token(
        "gho_token_abcdefghij", login="alice", scopes="repo", data_dir=tmp_path
    )
    assert kind == "keyring"
    # Token landed in the in-mem backend.
    assert (
        mem_keyring.get_password(auth_storage.KEYRING_SERVICE, auth_storage.KEYRING_USER)
        == "gho_token_abcdefghij"
    )
    # File was NOT written when keyring succeeded.
    assert not (tmp_path / "auth" / "token.json").exists()
    # load_token returns the keyring value.
    assert auth_storage.load_token(data_dir=tmp_path) == "gho_token_abcdefghij"


def test_describe_source_returns_meta(mem_keyring: _MemKeyring, tmp_path: Path) -> None:
    auth_storage.store_token("tok_xxxxxxxxxx", login="alice", scopes="repo", data_dir=tmp_path)
    src = auth_storage.describe_source(data_dir=tmp_path)
    assert src is not None
    assert src.kind == "keyring"
    assert src.login == "alice"
    assert src.scopes == "repo"


# ---------- file fallback ----------


def test_falls_back_to_file_when_keyring_set_raises(
    broken_keyring: _BrokenKeyring, tmp_path: Path
) -> None:
    kind = auth_storage.store_token("gho_file_token12345", data_dir=tmp_path)
    assert kind == "file"
    p = tmp_path / "auth" / "token.json"
    assert p.exists()
    body = json.loads(p.read_text())
    assert body["token"] == "gho_file_token12345"


def test_file_is_written_with_mode_0600(broken_keyring: _BrokenKeyring, tmp_path: Path) -> None:
    auth_storage.store_token("tok_xxxxxxxxxx", data_dir=tmp_path)
    p = tmp_path / "auth" / "token.json"
    mode = stat.S_IMODE(p.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {mode:o}"


def test_load_token_reads_file_when_keyring_empty(mem_keyring: _MemKeyring, tmp_path: Path) -> None:
    """Keyring is reachable but empty; file has the token. Keyring still wins
    when *populated*, but an empty keyring should not mask a file value."""
    p = tmp_path / "auth" / "token.json"
    p.parent.mkdir(mode=0o700)
    p.write_text(json.dumps({"token": "tok_in_file_1234", "login": None, "scopes": None}))
    p.chmod(0o600)
    assert auth_storage.load_token(data_dir=tmp_path) == "tok_in_file_1234"


def test_world_readable_file_is_refused(
    broken_keyring: _BrokenKeyring, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "auth" / "token.json"
    p.parent.mkdir(mode=0o700)
    p.write_text(json.dumps({"token": "tok_xxxxxxxxxx"}))
    p.chmod(0o644)  # group/other-readable
    with caplog.at_level("WARNING"):
        loaded = auth_storage.load_token(data_dir=tmp_path)
    assert loaded is None
    assert any("group/other readable" in rec.message for rec in caplog.records)


# ---------- precedence ----------


def test_keyring_wins_when_both_present(mem_keyring: _MemKeyring, tmp_path: Path) -> None:
    # File present
    p = tmp_path / "auth" / "token.json"
    p.parent.mkdir(mode=0o700)
    p.write_text(json.dumps({"token": "from_file_xxxxxx"}))
    p.chmod(0o600)
    # Keyring present
    auth_storage.store_token("from_keyring_xxx", data_dir=tmp_path)
    assert auth_storage.load_token(data_dir=tmp_path) == "from_keyring_xxx"


# ---------- delete ----------


def test_delete_clears_both_backends(mem_keyring: _MemKeyring, tmp_path: Path) -> None:
    auth_storage.store_token("tok_xxxxxxxxxx", data_dir=tmp_path)
    # Also drop a file copy to confirm both get cleaned.
    p = tmp_path / "auth" / "token.json"
    p.parent.mkdir(mode=0o700, exist_ok=True)
    p.write_text(json.dumps({"token": "tok_yyyyyyyyyy"}))
    p.chmod(0o600)

    auth_storage.delete_token(data_dir=tmp_path)
    assert auth_storage.load_token(data_dir=tmp_path) is None
    assert not p.exists()


def test_delete_is_safe_when_nothing_persisted(tmp_path: Path) -> None:
    # No fixtures used — exercises the "everything missing" path.
    auth_storage.delete_token(data_dir=tmp_path)  # must not raise
