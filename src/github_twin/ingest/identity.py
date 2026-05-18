"""Email → GitHub login resolver.

The git-local commits walk reads `author.email` from commit objects, but
`artifact.author_login` (used by org-mode filtering) is server-side state
GitHub holds. We resolve unknown emails lazily via `/search/commits` with an
`author-email:` qualifier and cache the result — including misses — so each
unique email costs at most one API call per corpus.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from github_twin.ingest.github_client import GitHubClient, GitHubError
from github_twin.store import queries as q

log = logging.getLogger(__name__)

# `12345+octocat@users.noreply.github.com` or `octocat@users.noreply.github.com`
_NOREPLY = re.compile(r"^(?:(?P<id>\d+)\+)?(?P<login>[A-Za-z0-9-]+)@users\.noreply\.github\.com$")


def _parse_noreply(email: str) -> str | None:
    m = _NOREPLY.match(email)
    return m.group("login") if m else None


def resolve_login(
    conn: sqlite3.Connection,
    gh: GitHubClient | None,
    email: str | None,
) -> str | None:
    """Return the GitHub login for `email`, or None if no link is known.

    Lookup order: DB cache → `*@users.noreply.github.com` parse → GitHub
    `/search/commits` (cached, including misses). Pass `gh=None` to disable
    the API fallback (cache-only mode, useful in tests).
    """
    if not email:
        return None
    e = email.lower()

    cached, resolved = q.get_email_login(conn, e)
    if resolved:
        return cached

    noreply_login = _parse_noreply(e)
    if noreply_login is not None:
        q.upsert_email_login(conn, email=e, login=noreply_login, source="noreply")
        return noreply_login

    if gh is None:
        return None

    login: str | None = None
    try:
        for item in gh.paginate(
            "/search/commits",
            params={"q": f'author-email:"{e}"', "per_page": 1},
        ):
            cand = (item.get("author") or {}).get("login")
            if cand:
                login = cand
            break
    except GitHubError as ex:
        log.warning("email→login lookup failed for %s: %s", e, ex)
        return None

    q.upsert_email_login(conn, email=e, login=login, source="search_commits")
    return login


def bulk_resolve_logins(
    conn: sqlite3.Connection,
    gh: GitHubClient | None,
    emails: Iterable[str],
    *,
    max_workers: int = 4,
) -> None:
    """Pre-warm `email_login_map` for a batch of distinct emails.

    Each email is either: already cached (no-op), parseable as a noreply
    address (resolved synchronously, no API), or needs a `/search/commits`
    lookup. Only the last group benefits from concurrency, so we fan out
    the HTTP calls; DB writes happen one at a time on the calling thread
    via the existing single-writer model.
    """
    distinct: list[str] = []
    seen: set[str] = set()
    for email in emails:
        if not email:
            continue
        norm = email.lower()
        if norm in seen:
            continue
        seen.add(norm)
        distinct.append(norm)

    needs_lookup: list[str] = []
    for e in distinct:
        _, resolved = q.get_email_login(conn, e)
        if resolved:
            continue
        if _parse_noreply(e) is not None:
            # Cheap, synchronous.
            resolve_login(conn, gh=None, email=e)
            continue
        needs_lookup.append(e)

    if not needs_lookup or gh is None:
        return

    # Two-phase: do the HTTP lookups in parallel (read-only, cacheable),
    # then write the cache entries serially on this thread.
    def _lookup(email: str) -> tuple[str, str | None]:
        try:
            for item in gh.paginate(
                "/search/commits",
                params={"q": f'author-email:"{email}"', "per_page": 1},
            ):
                cand = (item.get("author") or {}).get("login")
                return email, cand
        except GitHubError as ex:
            log.warning("email→login lookup failed for %s: %s", email, ex)
            return email, None
        return email, None

    workers = min(max_workers, len(needs_lookup))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for email, login in pool.map(_lookup, needs_lookup):
            q.upsert_email_login(conn, email=email, login=login, source="search_commits")
