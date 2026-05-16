"""Commits ingest.

Two paths share this module, dispatched by `cfg.use_local_git_for_commits`:

- **git-local** (default): walk a deep persistent clone with `git log` +
  `git show` and resolve `author_login` via the `email_login_map` cache. No
  per-sha GitHub API calls; not bounded by the `/search/commits` 1000-result
  cap.
- **API fallback** (`use_local_git_for_commits=False`): the legacy path that
  paginates `/search/commits` (user) or `/repos/{r}/commits` (org) and fetches
  each diff via `/repos/{r}/commits/{sha}` with `Accept: application/vnd.github.diff`.

Re-ingest is idempotent in both paths: commits keyed on SHA, chunks are wiped
and rewritten on re-ingest.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from github_twin.config import IngestCfg
from github_twin.ingest.cache import RawCache
from github_twin.ingest.clone import CloneError, _git, commits_clone
from github_twin.ingest.github_client import GitHubClient, GitHubError
from github_twin.ingest.identity import resolve_login
from github_twin.process.chunkers import chunk_commit_message, chunk_diff
from github_twin.process.language import language_for_path
from github_twin.store import queries as q

log = logging.getLogger(__name__)


@dataclass
class CommitStats:
    fetched: int = 0
    code_chunks: int = 0
    message_chunks: int = 0
    skipped: int = 0


# ---------- shared helpers ----------


def _primary_language(diff: str) -> str | None:
    """Pick a representative language from a diff (first chunkable file)."""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip().removeprefix("b/")
            if path == "/dev/null":
                continue
            lang = language_for_path(path)
            if lang:
                return lang
    return None


def _match(repo_full: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(repo_full, pattern)


def _allowed_repo(repo_full: str, cfg: IngestCfg) -> bool:
    if cfg.exclude_repos and any(_match(repo_full, p) for p in cfg.exclude_repos):
        return False
    return not (cfg.include_repos and not any(_match(repo_full, p) for p in cfg.include_repos))


def _write_commit_artifact(
    *,
    conn: sqlite3.Connection,
    cfg: IngestCfg,
    repo_full: str,
    sha: str,
    diff: str,
    commit_msg: str,
    author_email: str | None,
    author_login: str | None,
    created_at: str | None,
    url: str | None,
    stats: CommitStats,
) -> None:
    """Persist one commit (artifact + chunks). Common to both ingest paths."""
    artifact_id = q.upsert_artifact(
        conn,
        kind="commit",
        external_id=sha,
        source_url=url,
        repo=repo_full,
        language=_primary_language(diff),
        author_email=author_email,
        author_login=author_login,
        created_at=created_at,
        decision=None,
        meta={
            "sha": sha,
            "message_first_line": commit_msg.splitlines()[0][:200] if commit_msg else "",
        },
    )
    q.delete_chunks_for_artifact(conn, artifact_id)

    for ck in chunk_diff(
        diff,
        repo=repo_full,
        sha=sha,
        source_url=url,
        exclude_patterns=cfg.exclude_paths,
    ):
        q.insert_chunk(
            conn,
            artifact_id=artifact_id,
            kind="code",
            text=ck.text,
            context=ck.context,
            language=ck.language,
        )
        stats.code_chunks += 1

    msg_chunk = chunk_commit_message(commit_msg, repo=repo_full, sha=sha, source_url=url)
    if msg_chunk:
        q.insert_chunk(
            conn,
            artifact_id=artifact_id,
            kind="commit_message",
            text=msg_chunk.text,
            context=msg_chunk.context,
            language=None,
        )
        stats.message_chunks += 1

    stats.fetched += 1


# ---------- git-local helpers ----------


@dataclass(frozen=True)
class _GitLogRow:
    sha: str
    date: str  # ISO8601
    email: str  # lowercased
    parents: str  # space-separated parent SHAs
    message: str  # full message (subject + body)


def _git_log(
    clone_path: Path,
    *,
    range_args: list[str],
    author_emails: Iterable[str] | None = None,
) -> Iterator[_GitLogRow]:
    """Walk `git log`, yielding one row per commit.

    Uses unit / record separators (`\\x1f` / `\\x1e`) so commit messages with
    embedded newlines / NULs don't break parsing.
    """
    fmt = "%H%x1f%aI%x1f%aE%x1f%P%x1f%B%x1e"
    args = ["log", f"--format={fmt}", "--no-merges"]
    if author_emails:
        for e in author_emails:
            args += ["--author", e]
    args += range_args
    try:
        out = _git(args, cwd=clone_path)
    except CloneError as e:
        log.warning("git log failed in %s: %s", clone_path, e)
        return
    for record in out.split("\x1e"):
        record = record.lstrip("\n")
        if not record:
            continue
        parts = record.split("\x1f")
        if len(parts) < 5:
            continue
        sha, date, email, parents, message = parts[0], parts[1], parts[2], parts[3], parts[4]
        yield _GitLogRow(
            sha=sha,
            date=date,
            email=email.lower(),
            parents=parents,
            message=message.rstrip("\n"),
        )


def _git_show_diff(clone_path: Path, sha: str) -> str | None:
    """Return the unified diff for a single commit (against its first parent)."""
    try:
        return _git(
            ["show", "--format=", "--no-color", "--first-parent", sha],
            cwd=clone_path,
        )
    except CloneError as e:
        log.warning("git show %s failed: %s", sha[:8], e)
        return None


def _html_url(repo_full: str, sha: str) -> str:
    return f"https://github.com/{repo_full}/commit/{sha}"


def _walk_repo_local(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient | None,
    cfg: IngestCfg,
    repo_full: str,
    clone_path: Path,
    head_sha: str,
    last_walked_sha: str | None,
    author_emails: list[str] | None,
    resolve_author_login: bool,
    limit: int | None,
    stats: CommitStats,
) -> None:
    """Walk a single repo's commits via git locally.

    `author_emails`  → restrict the log to commits authored by these emails
                       (user mode).
    `resolve_author_login` → look up `author_login` per email via the resolver
                       (org mode).
    """
    if last_walked_sha:
        range_args = [f"{last_walked_sha}..{head_sha}"]
    else:
        range_args = [head_sha, f"--since={cfg.since}"]

    seen = 0
    for row in _git_log(clone_path, range_args=range_args, author_emails=author_emails):
        if limit is not None and seen >= limit:
            break
        seen += 1

        diff = _git_show_diff(clone_path, row.sha)
        if diff is None:
            stats.skipped += 1
            continue

        author_login = None
        if resolve_author_login:
            author_login = resolve_login(conn, gh, row.email)

        _write_commit_artifact(
            conn=conn,
            cfg=cfg,
            repo_full=repo_full,
            sha=row.sha,
            diff=diff,
            commit_msg=row.message,
            author_email=row.email or None,
            author_login=author_login,
            created_at=row.date,
            url=_html_url(repo_full, row.sha),
            stats=stats,
        )


# ---------- user-mode entry point ----------


def _discover_user_repos(
    gh: GitHubClient, *, username: str, emails: Iterable[str], since: str
) -> list[str]:
    """One pass over /search/commits to enumerate repos the user touched.

    Capped by GitHub's 1000-result search ceiling, but we only need distinct
    repo names — for most users that's well below the cap.
    """
    seen: set[str] = set()
    queries = [f"author:{username} author-date:>={since}"]
    for e in emails:
        if e.endswith("@users.noreply.github.com"):
            continue
        queries.append(f'author-email:"{e}" author-date:>={since}')
    for qstr in queries:
        log.info("discover user repos: %s", qstr)
        try:
            for item in gh.paginate("/search/commits", params={"q": qstr, "per_page": 100}):
                full = (item.get("repository") or {}).get("full_name")
                if full:
                    seen.add(full)
        except GitHubError as e:
            log.warning("search/commits failed (%s): %s", qstr, e)
    return sorted(seen)


def ingest_commits(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cache: RawCache,
    username: str,
    emails: list[str],
    cfg: IngestCfg,
    since: str | None = None,
    limit: int | None = None,
) -> CommitStats:
    if cfg.use_local_git_for_commits:
        return _ingest_commits_local(
            conn=conn,
            gh=gh,
            username=username,
            emails=emails,
            cfg=cfg,
            limit=limit,
        )
    return _ingest_commits_api(
        conn=conn,
        gh=gh,
        cache=cache,
        username=username,
        emails=emails,
        cfg=cfg,
        since=since,
        limit=limit,
    )


def _ingest_commits_local(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    username: str,
    emails: list[str],
    cfg: IngestCfg,
    limit: int | None,
) -> CommitStats:
    stats = CommitStats()
    email_set = {e.lower() for e in emails}

    repos = _discover_user_repos(gh, username=username, emails=emails, since=cfg.since)
    log.info("user-mode commits: %d repos to walk", len(repos))
    for repo_full in repos:
        if not _allowed_repo(repo_full, cfg):
            continue
        existing = q.get_repo(conn, repo_full)
        last_walked = existing.get("last_commits_walked_sha") if existing else None
        try:
            with commits_clone(repo_full, cache_dir=cfg.clones_dir) as clone:
                # Make sure the repo row exists so cursors persist.
                if existing is None:
                    q.upsert_repo(
                        conn,
                        full_name=repo_full,
                        default_branch=None,
                        pushed_at=None,
                        size_kb=None,
                    )
                _walk_repo_local(
                    conn=conn,
                    gh=gh,
                    cfg=cfg,
                    repo_full=repo_full,
                    clone_path=clone.path,
                    head_sha=clone.head_sha,
                    last_walked_sha=last_walked,
                    author_emails=sorted(email_set),
                    resolve_author_login=False,
                    limit=limit,
                    stats=stats,
                )
                q.set_repo_cursor(
                    conn,
                    full_name=repo_full,
                    commits_walked_sha=clone.head_sha,
                    commits_at=_now_iso(),
                )
        except CloneError as e:
            log.warning("clone %s failed, skipping: %s", repo_full, e)
            stats.skipped += 1
    return stats


# ---------- org-mode entry point ----------


def ingest_commits_org(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cache: RawCache,
    cfg: IngestCfg,
    limit_per_repo: int | None = None,
) -> CommitStats:
    """Walk every repo's commits since its per-repo cursor.

    Author identity (`author_login`) is captured via the email→login cache.
    """
    if cfg.use_local_git_for_commits:
        return _ingest_commits_org_local(
            conn=conn,
            gh=gh,
            cfg=cfg,
            limit_per_repo=limit_per_repo,
        )
    return _ingest_commits_org_api(
        conn=conn,
        gh=gh,
        cache=cache,
        cfg=cfg,
        limit_per_repo=limit_per_repo,
    )


def _ingest_commits_org_local(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cfg: IngestCfg,
    limit_per_repo: int | None,
) -> CommitStats:
    stats = CommitStats()
    for row in q.list_repos(conn):
        repo_full = row["full_name"]
        if not _allowed_repo(repo_full, cfg):
            continue
        last_walked = row.get("last_commits_walked_sha")
        log.info("commits org (local): %s since %s", repo_full, last_walked or cfg.since)
        try:
            with commits_clone(repo_full, cache_dir=cfg.clones_dir) as clone:
                _walk_repo_local(
                    conn=conn,
                    gh=gh,
                    cfg=cfg,
                    repo_full=repo_full,
                    clone_path=clone.path,
                    head_sha=clone.head_sha,
                    last_walked_sha=last_walked,
                    author_emails=None,
                    resolve_author_login=True,
                    limit=limit_per_repo,
                    stats=stats,
                )
                q.set_repo_cursor(
                    conn,
                    full_name=repo_full,
                    commits_walked_sha=clone.head_sha,
                    commits_at=_now_iso(),
                )
        except CloneError as e:
            log.warning("clone %s failed, skipping: %s", repo_full, e)
            stats.skipped += 1
    return stats


# ---------- API path (legacy, kept behind use_local_git_for_commits=False) ----------


def _search_queries(*, username: str, emails: Iterable[str], since: str) -> list[str]:
    qs = [f"author:{username} author-date:>={since}"]
    for e in emails:
        if e.endswith("@users.noreply.github.com"):
            continue  # noreply addresses don't add coverage beyond author:<username>
        qs.append(f'author-email:"{e}" author-date:>={since}')
    return qs


def _iter_commit_items(
    gh: GitHubClient, username: str, emails: Iterable[str], since: str
) -> Iterator[dict[str, Any]]:
    """Union of all search queries, deduped by commit SHA."""
    seen: set[str] = set()
    for q_str in _search_queries(username=username, emails=emails, since=since):
        log.info("commits search: %s", q_str)
        for item in gh.paginate("/search/commits", params={"q": q_str, "per_page": 100}):
            sha = item.get("sha")
            if not sha or sha in seen:
                continue
            seen.add(sha)
            yield item


def _newest_iso(items: Iterable[dict[str, Any]]) -> str | None:
    best: str | None = None
    for item in items:
        when = ((item.get("commit") or {}).get("author") or {}).get("date")
        if when and (best is None or when > best):
            best = when
    return best


def _ingest_commits_api(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cache: RawCache,
    username: str,
    emails: list[str],
    cfg: IngestCfg,
    since: str | None = None,
    limit: int | None = None,
) -> CommitStats:
    cursor = since or q.get_cursor(conn, "commits") or cfg.since
    log.info("ingesting commits (api) since %s", cursor)
    stats = CommitStats()

    items_raw = list(_iter_commit_items(gh, username=username, emails=emails, since=cursor))
    if limit:
        items_raw = items_raw[:limit]
    newest = _newest_iso(items_raw) or cursor

    for item in items_raw:
        sha = item["sha"]
        repo_full = (item.get("repository") or {}).get("full_name")
        if not repo_full:
            stats.skipped += 1
            continue
        if not _allowed_repo(repo_full, cfg):
            stats.skipped += 1
            continue

        cache.write_json("commits", sha, item)
        diff = _fetch_diff(gh, cache, repo_full, sha)
        if diff is None:
            stats.skipped += 1
            continue

        commit_meta = item.get("commit") or {}
        author_meta = commit_meta.get("author") or {}
        _write_commit_artifact(
            conn=conn,
            cfg=cfg,
            repo_full=repo_full,
            sha=sha,
            diff=diff,
            commit_msg=commit_meta.get("message") or "",
            author_email=(author_meta.get("email") or "").lower() or None,
            author_login=None,
            created_at=author_meta.get("date"),
            url=item.get("html_url"),
            stats=stats,
        )

    if newest and stats.fetched > 0:
        bumped = _bump_iso(newest)
        q.set_cursor(conn, "commits", bumped)
    return stats


def _bump_iso(iso: str) -> str:
    """Advance an ISO date by 1 second so we don't re-fetch the boundary."""
    from datetime import timedelta

    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    return (dt.astimezone(UTC) + timedelta(seconds=1)).isoformat(timespec="seconds")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _fetch_diff(gh: GitHubClient, cache: RawCache, repo_full: str, sha: str) -> str | None:
    diff = cache.read_text("commits", sha, "diff")
    if diff is not None:
        return diff
    try:
        diff = gh.get_text(
            f"/repos/{repo_full}/commits/{sha}",
            accept="application/vnd.github.diff",
        )
    except GitHubError as e:
        log.warning("skip %s/%s diff: %s", repo_full, sha[:8], e)
        return None
    cache.write_text("commits", sha, "diff", diff)
    return diff


def _ingest_commits_org_api(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cache: RawCache,
    cfg: IngestCfg,
    limit_per_repo: int | None = None,
) -> CommitStats:
    stats = CommitStats()
    for row in q.list_repos(conn):
        repo_full = row["full_name"]
        cursor = row.get("last_commits_at") or cfg.since
        log.info("commits org (api): %s since %s", repo_full, cursor)
        params = {"per_page": 100, "since": cursor}
        items = gh.paginate(f"/repos/{repo_full}/commits", params=params)

        for seen, item in enumerate(items):
            if limit_per_repo is not None and seen >= limit_per_repo:
                break
            sha = item.get("sha")
            if not sha:
                stats.skipped += 1
                continue
            diff = _fetch_diff(gh, cache, repo_full, sha)
            if diff is None:
                stats.skipped += 1
                continue
            commit_meta = item.get("commit") or {}
            author_meta = commit_meta.get("author") or {}
            author_acct = item.get("author") or {}
            _write_commit_artifact(
                conn=conn,
                cfg=cfg,
                repo_full=repo_full,
                sha=sha,
                diff=diff,
                commit_msg=commit_meta.get("message") or "",
                author_email=(author_meta.get("email") or "").lower() or None,
                author_login=author_acct.get("login"),
                created_at=author_meta.get("date"),
                url=item.get("html_url"),
                stats=stats,
            )
        q.set_repo_cursor(conn, full_name=repo_full, commits_at=_now_iso())
    return stats
