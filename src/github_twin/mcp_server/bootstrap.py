"""MCP bootstrap: surface 'this repo needs init' into the protocol.

When `gt serve` starts in an uninitialized repo, query tools raise
NeedsInitError and the client (Claude) can call `bootstrap` to drive
setup end-to-end with streamed progress notifications.

Mirrors the pure-Python tool pattern in `tools.py`: no MCP framework deps
here; `server.py` wires the async progress bridge around `run_bootstrap`.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from github_twin.config import Config
from github_twin.ingest.github_client import GitHubClient
from github_twin.ingest.repos import enumerate_org_repos
from github_twin.pipeline import run_embed, run_ingest
from github_twin.store import queries as q
from github_twin.store.db import db_session, transaction
from github_twin.target import (
    Target,
    discover_org,
    discover_repo,
    discover_user,
    load_targets,
    maybe_discover_repo,
    read_origin_owner_name,
    save_target,
    swap_fork_to_upstream,
)

log = logging.getLogger(__name__)

Reporter = Callable[[str], None]


def _noop(_: str) -> None:
    return None


# Single in-process flag: concurrent bootstrap calls refuse rather than
# share a worker connection. Reset on success or failure via try/finally.
_in_progress = False


def is_in_progress() -> bool:
    return _in_progress


def status_payload(conn: sqlite3.Connection) -> dict[str, Any]:
    """Pure helper for the `bootstrap_status` MCP tool. Independent of the
    MCP framework so it can be unit-tested directly.

    Also walks the MCP server's cwd for a github.com `.git` origin so
    "I have user-mode set up but the current repo isn't indexed" surfaces
    as a distinct recommendation — without that signal the tool would
    silently return ready against an indexed user corpus while
    `find_code` for the current repo turns up nothing.
    """
    targets = [{"kind": t.kind, "name": t.name} for t in load_targets(conn)]
    stats = q.stats(conn)
    in_progress = is_in_progress()

    # Compare cwd to the `repo` table so the recommendation can call out a
    # missing current-repo even when other targets are healthy.
    origin = read_origin_owner_name()
    current_repo: dict[str, str] | None = None
    current_repo_indexed = False
    if origin is not None:
        owner, name = origin
        full_name = f"{owner}/{name}"
        current_repo = {"full_name": full_name, "owner": owner, "name": name}
        row = conn.execute(
            "SELECT 1 FROM repo WHERE full_name = ? LIMIT 1", (full_name,)
        ).fetchone()
        current_repo_indexed = row is not None

    recommendation: str | None = None
    if in_progress:
        recommendation = "bootstrap is currently running; poll bootstrap_status again."
    elif not targets:
        recommendation = (
            "No target configured. Call `bootstrap` to auto-detect from the "
            "MCP server's working directory, or pass kind='user' / 'org' / "
            "'repo' with name='...' to override."
        )
    elif stats.get("vectors", 0) == 0:
        recommendation = (
            "Target exists but no chunks are embedded. Call `bootstrap` "
            "(without skip_sync) or `sync` to ingest + embed."
        )
    elif current_repo is not None and not current_repo_indexed:
        # Fork caveat: `bootstrap` auto-swaps to upstream by default, so the
        # row that lands may not literally match `current_repo.full_name`.
        # A re-run is idempotent, so the worst case is the user calls
        # bootstrap and discovers the upstream was already indexed.
        recommendation = (
            f"Current repo {current_repo['full_name']} is not in the index. "
            "Call `bootstrap` to add it (existing targets stay intact; forks "
            "auto-swap to upstream unless keep_fork=true)."
        )
    return {
        "db_initialized": True,
        "targets": targets,
        "stats": stats,
        "in_progress": in_progress,
        "current_repo": current_repo,
        "current_repo_indexed": current_repo_indexed,
        "recommendation": recommendation,
    }


@dataclass
class BootstrapSpec:
    """Inputs to `bootstrap`. Mirrors `gt init` CLI flags.

    - `kind`: 'user' | 'org' | 'repo' | None (auto-detect from `path`).
    - `name`: user login (kind='user'), org login (kind='org'), or
      'owner/name' (kind='repo'). Optional for repo when `path` resolves.
    - `path`: filesystem path to walk for a `.git/config`. Used by 'repo'
      and 'auto' kinds. None means caller's cwd at the call site.
    - `keep_fork`: when the discovered repo is a fork, keep it instead of
      swapping to upstream (default: swap, matching `gt init`).
    - `skip_sync`: stop after writing target/repo rows. Lets the client
      kick off a fast init and invoke `sync` later.
    - `include_archived`: include archived repos in org enumeration
      (default: skip them, catching internal-archived too). Overrides
      `cfg.ingest.include_archived`.
    """

    kind: str | None = None
    name: str | None = None
    path: Path | None = None
    keep_fork: bool = False
    skip_sync: bool = False
    include_archived: bool = False


def init_target(
    cfg: Config,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    spec: BootstrapSpec,
    *,
    report: Reporter = _noop,
) -> Target:
    """Discover + persist a target row + (for repo/org) its repo metadata.

    Mirrors the three branches of `cli.py:init`. Returns the saved
    Target with `id` populated. Idempotent: re-running with the same
    (kind, name) refreshes the existing row.
    """
    kind = (spec.kind or "auto").lower()

    if kind == "auto":
        auto = maybe_discover_repo(gh, start_path=spec.path)
        if auto is not None:
            target, metadata, parent_full_name = auto
            target, metadata = swap_fork_to_upstream(
                gh,
                target,
                metadata,
                parent_full_name,
                keep_fork=spec.keep_fork,
                report=report,
            )
            with transaction(conn):
                target = save_target(conn, target)
                assert target.id is not None
                q.upsert_repo(conn, target_id=target.id, **metadata)
            report(f"added repo target {target.name} (auto-detected)")
            return target
        report("no github.com .git found; falling back to user mode")
        kind = "user"

    if kind == "user":
        target = discover_user(gh, cfg.identity)
        with transaction(conn):
            target = save_target(conn, target)
        report(f"added user target {target.name} (emails: {len(target.emails)})")
        return target

    if kind == "org":
        if not spec.name:
            raise ValueError("kind='org' requires `name` (the org login)")
        keep_archived = spec.include_archived or cfg.ingest.include_archived
        target = discover_org(gh, spec.name)
        with transaction(conn):
            target = save_target(conn, target)
        assert target.id is not None
        report(
            f"added org target {target.name}; enumerating repos "
            f"(include_archived={keep_archived})..."
        )
        n_kept = 0
        with transaction(conn):
            for r in enumerate_org_repos(
                gh,
                target.name,
                include=cfg.ingest.include_repos,
                exclude=cfg.ingest.exclude_repos,
                include_archived=keep_archived,
            ):
                q.upsert_repo(conn, target_id=target.id, **r)
                n_kept += 1
        report(f"saved {n_kept} repos for org {target.name}")
        return target

    if kind == "repo":
        target, metadata, parent_full_name = discover_repo(gh, repo=spec.name, start_path=spec.path)
        target, metadata = swap_fork_to_upstream(
            gh,
            target,
            metadata,
            parent_full_name,
            keep_fork=spec.keep_fork,
            report=report,
        )
        with transaction(conn):
            target = save_target(conn, target)
            assert target.id is not None
            q.upsert_repo(conn, target_id=target.id, **metadata)
        report(f"added repo target {target.name}")
        return target

    raise ValueError(f"Unknown kind: {kind!r}. Expected 'user', 'org', or 'repo'.")


def run_bootstrap(
    cfg: Config,
    spec: BootstrapSpec,
    *,
    report: Reporter = _noop,
) -> dict[str, Any]:
    """Open a fresh DB connection, run init + ingest + embed.

    Opens its own connection so it can run on a worker thread while the
    MCP server's main-thread connection stays available for other tools.
    Both share the same WAL-mode SQLite file safely.
    """
    global _in_progress
    if _in_progress:
        raise RuntimeError("bootstrap already in progress; wait for it to finish")
    _in_progress = True
    try:
        with db_session(cfg.paths.db_path, cfg.embed.dim) as conn:
            with GitHubClient() as gh:
                target = init_target(cfg, conn, gh, spec, report=report)

            if spec.skip_sync:
                report("init complete (skip_sync=true; call `sync` to ingest)")
                return {
                    "target": {"kind": target.kind, "name": target.name},
                    "ingested": False,
                    "stats": q.stats(conn),
                }

            assert target.id is not None
            report(f"ingesting {target.kind}:{target.name}...")
            run_ingest(cfg, conn, target=target.id, report=report)
            report("embedding...")
            run_embed(cfg, conn, report=report)
            stats = q.stats(conn)
            report(f"bootstrap complete: {stats['chunks']} chunks, {stats['vectors']} vectors")
            return {
                "target": {"kind": target.kind, "name": target.name},
                "ingested": True,
                "stats": stats,
            }
    finally:
        _in_progress = False
