"""Persist the OAuth device-flow access token across `gt` invocations.

Two backends, transparent to callers:

1. **OS keyring** (`keyring.set_password`) — macOS Keychain, Linux
   Secret Service (gnome-keyring / kwallet via D-Bus), Windows
   Credential Manager. Tried first.
2. **0600 file** at `<data_dir>/auth/token.json` — pure fallback for
   headless WSL/SSH/docker boxes where keyring is unavailable. Same
   posture as `gh auth login --insecure-storage`.

The fallback decision is per-call: if `keyring.set_password` raises,
we write the file. If `keyring.get_password` returns None, we read
the file. Tests can force either backend by monkeypatching the
keyring module.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import keyring
import keyring.errors

from github_twin.config import load_config

log = logging.getLogger(__name__)

KEYRING_SERVICE = "github-twin"
KEYRING_USER = "oauth"
KEYRING_META_USER = "oauth.meta"

StorageKind = Literal["keyring", "file"]


@dataclass(frozen=True)
class AuthSource:
    kind: StorageKind
    login: str | None
    scopes: str | None
    location: str


def _auth_file(data_dir: Path | None = None) -> Path:
    """Resolve the on-disk token path against the configured data_dir.

    A `None` override means: ask the configured `paths.data_dir`. Tests
    pass an explicit `data_dir` to avoid touching the user's real
    XDG dir.
    """
    if data_dir is None:
        data_dir = load_config().paths.data_dir
    return data_dir / "auth" / "token.json"


def _try_keyring_set(token: str, login: str | None, scopes: str | None) -> bool:
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, token)
        keyring.set_password(
            KEYRING_SERVICE,
            KEYRING_META_USER,
            json.dumps({"login": login, "scopes": scopes, "stored_at": int(time.time())}),
        )
    except (keyring.errors.KeyringError, Exception) as exc:  # noqa: BLE001
        log.debug("keyring set_password failed, falling back to file: %s", exc)
        return False
    return True


def _try_keyring_get() -> tuple[str, dict[str, object]] | None:
    try:
        tok = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    except (keyring.errors.KeyringError, Exception) as exc:  # noqa: BLE001
        log.debug("keyring get_password failed: %s", exc)
        return None
    if not tok:
        return None
    meta: dict[str, object] = {}
    try:
        raw = keyring.get_password(KEYRING_SERVICE, KEYRING_META_USER)
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                meta = parsed
    except (keyring.errors.KeyringError, Exception) as exc:  # noqa: BLE001
        log.debug("keyring meta read failed (ignored): %s", exc)
    return tok, meta


def _try_keyring_delete() -> bool:
    """Best-effort delete; returns True if something was removed."""
    removed = False
    for user in (KEYRING_USER, KEYRING_META_USER):
        try:
            keyring.delete_password(KEYRING_SERVICE, user)
            removed = True
        except keyring.errors.PasswordDeleteError:
            pass  # entry didn't exist
        except (keyring.errors.KeyringError, Exception) as exc:  # noqa: BLE001
            log.debug("keyring delete_password(%s) failed (ignored): %s", user, exc)
    return removed


def _write_file(path: Path, token: str, login: str | None, scopes: str | None) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # umask-proof write: 0600 from the start, no widening window.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(
                {
                    "token": token,
                    "login": login,
                    "scopes": scopes,
                    "stored_at": int(time.time()),
                },
                f,
            )
    except Exception:
        # If fdopen raised before taking ownership of fd, close it ourselves.
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


def _read_file(path: Path) -> tuple[str, dict[str, object]] | None:
    if not path.exists():
        return None
    st = path.stat()
    if st.st_mode & 0o077:
        log.warning(
            "Refusing to read %s: file is group/other readable (mode %o). "
            "Run `chmod 600 %s` to restore secure perms.",
            path,
            stat.S_IMODE(st.st_mode),
            path,
        )
        return None
    try:
        raw = path.read_text()
        body = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to read %s: %s", path, exc)
        return None
    if not isinstance(body, dict):
        return None
    tok = body.get("token")
    if not isinstance(tok, str) or not tok:
        return None
    return tok, body


def store_token(
    token: str,
    *,
    login: str | None = None,
    scopes: str | None = None,
    data_dir: Path | None = None,
) -> StorageKind:
    """Persist the token. Returns which backend won (for status display)."""
    if _try_keyring_set(token, login, scopes):
        return "keyring"
    _write_file(_auth_file(data_dir), token, login, scopes)
    return "file"


def load_token(data_dir: Path | None = None) -> str | None:
    """Return the persisted token, or None. Keyring wins over file."""
    kr = _try_keyring_get()
    if kr is not None:
        return kr[0]
    f = _read_file(_auth_file(data_dir))
    if f is not None:
        return f[0]
    return None


def delete_token(data_dir: Path | None = None) -> None:
    """Clear both backends. Never raises."""
    _try_keyring_delete()
    path = _auth_file(data_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.debug("unlink(%s) failed (ignored): %s", path, exc)


def describe_source(data_dir: Path | None = None) -> AuthSource | None:
    """Diagnostic snapshot of what `load_token` would return + where it lives."""
    kr = _try_keyring_get()
    if kr is not None:
        _, meta = kr
        login = meta.get("login") if isinstance(meta.get("login"), str) else None
        scopes = meta.get("scopes") if isinstance(meta.get("scopes"), str) else None
        return AuthSource(
            kind="keyring",
            login=login,  # type: ignore[arg-type]
            scopes=scopes,  # type: ignore[arg-type]
            location=f"{KEYRING_SERVICE}/{KEYRING_USER}",
        )
    path = _auth_file(data_dir)
    f = _read_file(path)
    if f is not None:
        _, meta = f
        login = meta.get("login") if isinstance(meta.get("login"), str) else None
        scopes = meta.get("scopes") if isinstance(meta.get("scopes"), str) else None
        return AuthSource(
            kind="file",
            login=login,  # type: ignore[arg-type]
            scopes=scopes,  # type: ignore[arg-type]
            location=str(path),
        )
    return None
