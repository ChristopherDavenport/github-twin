"""Target: who or what a target row tracks.

Three kinds:
  - user: a single GitHub user. Carries the discovered email set used to widen
          commit-search recall (people commit from work / personal / noreply).
  - org:  a whole GitHub organization. Has no emails; ingest is repo-scoped via
          the `repo` table populated by `gt init`.
  - repo: a single repository. Same pipeline as org with one repo-table row,
          but the target name is `'owner/name'`. Designed for the
          "cd into a cloned repo; just work" workflow.

A single DB can hold many targets — one user-mode + N org-mode + M repo-mode
co-exist. Per-target rows in `artifact` / `repo` / `sync_cursor` carry the
parent `target.id`. `(kind, name)` is unique so `gt init` is idempotent
per target.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from github_twin.config import IdentityCfg
from github_twin.ingest.github_client import GitHubClient
from github_twin.store import queries as q

log = logging.getLogger(__name__)


@dataclass
class Target:
    kind: str  # 'user' | 'org' | 'repo'
    name: str  # username, org login, or 'owner/name'
    external_id: int  # numeric GitHub id (user/org/repo id depending on kind)
    emails: list[str]  # user-mode only; empty list for org/repo
    id: int | None = None  # set after upsert / load; None on freshly discovered

    @property
    def is_user(self) -> bool:
        return self.kind == "user"

    @property
    def is_org(self) -> bool:
        return self.kind == "org"

    @property
    def is_repo(self) -> bool:
        return self.kind == "repo"


class AmbiguousTargetError(RuntimeError):
    """Raised when callers need a unique target but multiple match."""


def discover_user(gh: GitHubClient, cfg: IdentityCfg, *, sweep_pages: int = 5) -> Target:
    """User-mode discovery: username + author-email sweep."""
    user = gh.get_json("/user")
    username: str = user["login"]
    user_id: int = user["id"]

    discovered: set[str] = set()

    try:
        for entry in gh.get_json("/user/emails") or []:
            email = entry.get("email")
            if email:
                discovered.add(email.lower())
    except Exception as exc:  # noqa: BLE001
        log.warning("Skipping /user/emails (missing scope?): %s", exc)

    discovered.add(f"{user_id}+{username}@users.noreply.github.com".lower())
    discovered.add(f"{username}@users.noreply.github.com".lower())

    seen_pages = 0
    try:
        items = gh.paginate(
            "/search/commits",
            params={"q": f"author:{username}", "per_page": 100},
        )
        for item in items:
            email = (item.get("commit", {}).get("author", {}) or {}).get("email")
            if email:
                discovered.add(email.lower())
            if len(discovered) >= 25 or seen_pages >= sweep_pages * 100:
                break
            seen_pages += 1
    except Exception as exc:  # noqa: BLE001
        log.warning("Historical email sweep failed: %s", exc)

    for e in cfg.extra_emails:
        discovered.add(e.lower())
    for e in cfg.ignore_emails:
        discovered.discard(e.lower())

    return Target(
        kind="user",
        name=username,
        external_id=user_id,
        emails=sorted(discovered),
    )


def discover_org(gh: GitHubClient, org: str) -> Target:
    """Org-mode discovery: just the org login + numeric id.

    Repo enumeration is a separate step (O-B). Keeping discovery minimal here
    so `gt init --kind org` is cheap and idempotent.
    """
    data = gh.get_json(f"/orgs/{org}")
    return Target(
        kind="org",
        name=data["login"],
        external_id=data["id"],
        emails=[],
    )


# ---------- repo-mode discovery ----------


_HTTPS_RE = re.compile(r"^https://github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/?$")
_SSH_RE = re.compile(r"^git@github\.com:([^/\s]+)/([^/\s]+?)(?:\.git)?$")
_GIT_CONFIG_ORIGIN_RE = re.compile(
    r'\[remote "origin"\][^\[]*?url\s*=\s*([^\s\n]+)',
    re.MULTILINE | re.DOTALL,
)


def _parse_origin_owner_name(config_text: str) -> tuple[str, str] | None:
    """Pull (owner, name) from a `.git/config` text, or None if no github.com origin."""
    m = _GIT_CONFIG_ORIGIN_RE.search(config_text)
    if not m:
        return None
    url = m.group(1).strip()
    for regex in (_HTTPS_RE, _SSH_RE):
        hit = regex.match(url)
        if hit:
            return hit.group(1), hit.group(2)
    return None


def _find_git_root(start: Path) -> Path | None:
    """Walk up from `start` looking for a directory containing `.git`."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def discover_repo(
    gh: GitHubClient,
    *,
    repo: str | None = None,
    start_path: Path | None = None,
) -> tuple[Target, dict[str, Any]]:
    """Discover a repo target.

    - If `repo='owner/name'` is given, use it directly.
    - Else walk up from `start_path or Path.cwd()` to find `.git/config`,
      parse its `origin` URL, and use the resulting `'owner/name'`.

    Returns `(Target, repo_metadata_dict)` where the second value is shaped
    for `q.upsert_repo(**metadata)` (without target_id; the caller stamps it).
    """
    if repo is None:
        root = _find_git_root(start_path or Path.cwd())
        if root is None:
            raise ValueError(
                "No .git directory found from cwd. Pass --repo owner/name, "
                "or use --kind user / --kind org instead."
            )
        config_path = root / ".git" / "config"
        if not config_path.is_file():
            raise ValueError(
                f"Found .git at {root} but no .git/config inside it. "
                "Pass --repo owner/name explicitly."
            )
        parsed = _parse_origin_owner_name(config_path.read_text(encoding="utf-8", errors="replace"))
        if parsed is None:
            raise ValueError(
                f"Remote 'origin' in {config_path} is not a github.com URL. "
                "Pass --repo owner/name, or use --kind user / --kind org instead."
            )
        owner, name = parsed
        full_name = f"{owner}/{name}"
    else:
        if "/" not in repo:
            raise ValueError(f"--repo must be 'owner/name', got {repo!r}")
        full_name = repo
        owner, name = repo.split("/", 1)

    data = gh.get_json(f"/repos/{owner}/{name}")
    metadata = {
        "full_name": data.get("full_name") or full_name,
        "default_branch": data.get("default_branch"),
        "pushed_at": data.get("pushed_at"),
        "archived": bool(data.get("archived", False)),
        "fork": bool(data.get("fork", False)),
        "size_kb": data.get("size"),
    }
    target = Target(
        kind="repo",
        name=metadata["full_name"],
        external_id=int(data["id"]),
        emails=[],
    )
    return target, metadata


# ---------- persistence ----------


def _row_to_target(row: dict[str, Any]) -> Target:
    import json

    emails_raw = row["emails_json"]
    emails = json.loads(emails_raw) if emails_raw else []
    return Target(
        id=row["id"],
        kind=row["kind"],
        name=row["name"],
        external_id=row["external_id"],
        emails=list(emails),
    )


def load_targets(conn: sqlite3.Connection) -> list[Target]:
    """Return every target in the DB, ordered by id."""
    return [_row_to_target(row) for row in q.get_all_targets(conn)]


def load_target(
    conn: sqlite3.Connection,
    *,
    target_id: int | None = None,
    name: str | None = None,
    kind: str | None = None,
) -> Target | None:
    """Look up a single target.

    Exactly one of `target_id`, `name`, or `kind` should narrow the
    search:
      - `target_id=N` → exact id match.
      - `name="X"` → exact name match (raises `AmbiguousTargetError` if
        two kinds share the name, which shouldn't happen in practice).
      - `kind="user"` → return the sole target of that kind, or None
        if absent; raises `AmbiguousTargetError` if >1 exist.

    With no arguments, returns the lone target if exactly one exists,
    None if zero, raises `AmbiguousTargetError` if >1. This is the
    backward-compatible single-target convenience.
    """
    if target_id is not None:
        row = q.get_target_by_id(conn, target_id)
        return _row_to_target(row) if row else None
    if name is not None:
        rows = q.get_targets_by_name(conn, name)
        if not rows:
            return None
        if len(rows) > 1:
            kinds = sorted({r["kind"] for r in rows})
            raise AmbiguousTargetError(
                f"Multiple targets named {name!r} (kinds: {kinds}); pass kind="
            )
        return _row_to_target(rows[0])
    if kind is not None:
        rows = q.get_targets_by_kind(conn, kind)
        if not rows:
            return None
        if len(rows) > 1:
            names = sorted(r["name"] for r in rows)
            raise AmbiguousTargetError(
                f"Multiple {kind!r} targets: {names}; pass name= or target_id="
            )
        return _row_to_target(rows[0])
    rows = q.get_all_targets(conn)
    if not rows:
        return None
    if len(rows) > 1:
        names = sorted(f"{r['kind']}:{r['name']}" for r in rows)
        raise AmbiguousTargetError(
            f"Multiple targets in DB ({names}); pass target_id=, name=, or kind="
        )
    return _row_to_target(rows[0])


def save_target(conn: sqlite3.Connection, target: Target) -> Target:
    """Insert-or-update a target row keyed by (kind, name). Returns a
    Target with `id` populated. Idempotent: re-running with the same
    (kind, name) updates the existing row in place."""
    target_id = q.upsert_target(
        conn,
        kind=target.kind,
        name=target.name,
        external_id=target.external_id,
        emails=target.emails if target.is_user else None,
    )
    target.id = target_id
    return target


def maybe_discover_repo(
    gh: GitHubClient, *, start_path: Path | None = None
) -> tuple[Target, dict[str, Any]] | None:
    """Best-effort repo discovery for `gt init` auto-detect mode.

    Returns (target, metadata) on success, None on any failure (no .git,
    non-github origin, etc.). Suppresses the discovery errors so the caller
    can quietly fall back to user-mode.
    """
    try:
        return discover_repo(gh, start_path=start_path)
    except ValueError as exc:
        log.debug("repo auto-detect skipped: %s", exc)
        return None
