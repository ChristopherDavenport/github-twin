"""Org repo enumeration.

`gt init --kind org --org <name>` walks `/orgs/{org}/repos` to populate the
`repo` table. Subsequent phases (file walk, commits, reviews) iterate that
table instead of hitting GitHub's listing endpoint each time.

include/exclude filters use fnmatch on the `owner/name` string. An empty
`include_repos` means "all repos pass include"; `exclude_repos` is always
applied. Filters live in `IngestCfg` so they're shared across phases.
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterable, Iterator
from typing import Any

from github_twin.ingest.github_client import GitHubClient

log = logging.getLogger(__name__)


def matches_any(full_name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(full_name, pat) for pat in patterns)


def repo_passes_filters(
    full_name: str,
    *,
    include: list[str],
    exclude: list[str],
) -> bool:
    if include and not matches_any(full_name, include):
        return False
    return not (exclude and matches_any(full_name, exclude))


def enumerate_org_repos(
    gh: GitHubClient,
    org: str,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    include_archived: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield repo dicts (one per matching repo) shaped for `upsert_repo`.

    Pagination is handled by `GitHubClient.paginate`. `type=all` returns
    sources + forks + private repos the token can see; the `fork` and
    `visibility` columns on each row preserve that info for later filtering
    at query time.

    `include_archived=False` (the default) drops `archived=true` repos at
    enumeration so they never enter the DB. Pass `True` from sync to
    refresh archive state for already-known repos — downstream
    `q.list_repos(include_archived=False)` then naturally excludes them
    from ingest.
    """
    include = include or []
    exclude = exclude or []

    items = gh.paginate(
        f"/orgs/{org}/repos",
        params={"per_page": 100, "type": "all", "sort": "full_name"},
    )
    for item in items:
        full_name = item.get("full_name")
        if not full_name:
            continue
        if not repo_passes_filters(full_name, include=include, exclude=exclude):
            continue
        is_archived = bool(item.get("archived", False))
        if is_archived and not include_archived:
            continue
        yield {
            "full_name": full_name,
            "default_branch": item.get("default_branch"),
            "pushed_at": item.get("pushed_at"),
            "archived": is_archived,
            "visibility": item.get("visibility"),
            "fork": bool(item.get("fork", False)),
            "size_kb": item.get("size"),
        }
