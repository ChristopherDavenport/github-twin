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

import hashlib
import logging
import sqlite3
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from github_twin.config import IngestCfg
from github_twin.ingest.cache import RawCache
from github_twin.ingest.clone import CloneError, EmptyRepoError, _git, commits_clone
from github_twin.ingest.github_client import GitHubClient, GitHubError
from github_twin.ingest.identity import bulk_resolve_logins, resolve_login
from github_twin.process.chunkers import chunk_commit_message, chunk_diff
from github_twin.process.language import language_for_path
from github_twin.store import queries as q
from github_twin.store.db import transaction

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


def _commit_content_hash(diff: str, commit_msg: str) -> str:
    """sha256 over (diff, message). The message is included so a commit
    amended with only a message change still re-chunks the message side."""
    h = hashlib.sha256()
    h.update(diff.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(commit_msg.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _write_commit_artifact(
    *,
    conn: sqlite3.Connection,
    cfg: IngestCfg,
    target_id: int,
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
    """Persist one commit (artifact + chunks). Common to both ingest paths.

    Short-circuits when the (diff, message) hash matches the stored
    `artifact.content_hash` — refreshes metadata in place and skips the
    chunk wipe+re-insert (and the implicit `vec_chunk` invalidation).
    """
    new_hash = _commit_content_hash(diff, commit_msg)
    existing_id, existing_hash = q.get_artifact_content_hash(
        conn, target_id=target_id, kind="commit", external_id=sha
    )
    meta = {
        "sha": sha,
        "message_first_line": commit_msg.splitlines()[0][:200] if commit_msg else "",
    }
    if existing_id is not None and existing_hash == new_hash:
        # Same content; just refresh fields that may have improved
        # (author_login newly resolved, content_hash backfilled).
        q.update_artifact_metadata(
            conn,
            artifact_id=existing_id,
            author_login=author_login,
            content_hash=new_hash,
        )
        stats.fetched += 1
        return

    artifact_id = q.upsert_artifact(
        conn,
        target_id=target_id,
        kind="commit",
        external_id=sha,
        source_url=url,
        repo=repo_full,
        language=_primary_language(diff),
        author_email=author_email,
        author_login=author_login,
        created_at=created_at,
        decision=None,
        meta=meta,
        content_hash=new_hash,
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


# ---------- parallel org-mode pipeline ----------


@dataclass(frozen=True)
class _PreChunk:
    """A chunk computed in a worker thread, ready to insert on the writer."""

    kind: str  # 'code' | 'commit_message'
    text: str
    context: dict[str, Any] | None
    language: str | None


@dataclass
class _CommitRecord:
    """One commit's worth of data assembled by a worker. No DB I/O involved."""

    sha: str
    diff: str
    message: str
    author_email: str | None
    author_date: str  # ISO8601 from git log
    url: str
    pre_chunks: list[_PreChunk]


@dataclass
class _RepoRecords:
    """Worker output for one repo. `error` set when the clone or walk failed."""

    repo_full: str
    head_sha: str
    commits: list[_CommitRecord] = field(default_factory=list)
    error: BaseException | None = None
    elapsed_seconds: float = 0.0


def _shallow_since_for(last_commits_at: str | None, cfg: IngestCfg) -> str:
    """Compute the `--shallow-since` cutoff: cursor minus pad days, else
    `cfg.since`. The pad absorbs small rebases / clock skew."""
    base = last_commits_at or cfg.since
    try:
        dt = datetime.fromisoformat(base.replace("Z", "+00:00"))
    except ValueError:
        return base
    padded = dt - timedelta(days=cfg.shallow_since_pad_days)
    return padded.date().isoformat()


def _walk_since_for(last_commits_at: str | None, cfg: IngestCfg) -> str:
    """Compute the `git log --since` cutoff: cursor exactly, else `cfg.since`."""
    return last_commits_at or cfg.since


def _build_pre_chunks(
    *, diff: str, message: str, repo_full: str, sha: str, url: str, cfg: IngestCfg
) -> list[_PreChunk]:
    """Pure chunking: no DB access, safe to call from worker threads."""
    out: list[_PreChunk] = []
    for ck in chunk_diff(
        diff,
        repo=repo_full,
        sha=sha,
        source_url=url,
        exclude_patterns=cfg.exclude_paths,
    ):
        out.append(_PreChunk(kind="code", text=ck.text, context=ck.context, language=ck.language))
    msg_chunk = chunk_commit_message(message, repo=repo_full, sha=sha, source_url=url)
    if msg_chunk:
        out.append(
            _PreChunk(
                kind="commit_message",
                text=msg_chunk.text,
                context=msg_chunk.context,
                language=None,
            )
        )
    return out


def _walk_repo_records(
    *,
    cfg: IngestCfg,
    repo_full: str,
    last_commits_at: str | None,
    token: str | None,
    limit: int | None,
    author_emails: list[str] | None = None,
) -> _RepoRecords:
    """Worker: clone, walk, chunk. Pure compute / subprocess / network — no
    DB access (so it's safe to run on a `ThreadPoolExecutor` while the
    consumer owns the single SQLite connection).

    `author_emails` restricts the git log to commits authored by these
    emails (user-mode). Org-mode leaves it None and walks everything.
    """
    shallow_since = _shallow_since_for(last_commits_at, cfg)
    walk_since = _walk_since_for(last_commits_at, cfg)
    t0 = time.monotonic()
    try:
        with commits_clone(
            repo_full,
            cache_dir=cfg.clones_dir,
            token=token,
            shallow_since=shallow_since,
        ) as clone:
            records = _RepoRecords(repo_full=repo_full, head_sha=clone.head_sha)
            seen = 0
            for row in _git_log(
                clone.path,
                range_args=[f"--since={walk_since}"],
                author_emails=author_emails,
            ):
                if limit is not None and seen >= limit:
                    break
                seen += 1
                diff = _git_show_diff(clone.path, row.sha)
                if diff is None:
                    continue
                pre_chunks = _build_pre_chunks(
                    diff=diff,
                    message=row.message,
                    repo_full=repo_full,
                    sha=row.sha,
                    url=_html_url(repo_full, row.sha),
                    cfg=cfg,
                )
                records.commits.append(
                    _CommitRecord(
                        sha=row.sha,
                        diff=diff,
                        message=row.message,
                        author_email=row.email or None,
                        author_date=row.date,
                        url=_html_url(repo_full, row.sha),
                        pre_chunks=pre_chunks,
                    )
                )
            records.elapsed_seconds = time.monotonic() - t0
            return records
    except (CloneError, OSError) as e:
        return _RepoRecords(
            repo_full=repo_full,
            head_sha="",
            error=e,
            elapsed_seconds=time.monotonic() - t0,
        )


def _fetch_repo_pushed_at(
    gh: GitHubClient, repos: list[str], *, max_workers: int
) -> dict[str, str | None]:
    """Parallel `/repos/{owner}/{name}` for fast-skip + repo metadata refresh.

    Returns a dict mapping `full_name` → pushed_at (ISO8601), or None when
    the repo isn't reachable (404, transient error, archived without access).
    The full info dict is dropped — we only need pushed_at for the skip
    decision; callers that want more should re-fetch on demand."""
    if not repos:
        return {}

    def _one(repo_full: str) -> tuple[str, str | None]:
        try:
            data = gh.get_json_cached(f"/repos/{repo_full}")
            return repo_full, data.get("pushed_at")
        except GitHubError as e:
            log.warning("/repos/%s failed: %s", repo_full, e)
            return repo_full, None

    workers = max(1, min(max_workers, len(repos)))
    out: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for repo_full, pushed_at in pool.map(_one, repos):
            out[repo_full] = pushed_at
    return out


def _needs_walk(pushed_at: str | None, last_commits_at: str | None) -> bool:
    """True when the repo may have new commits we haven't ingested.

    Defaults conservative: if either side is missing we walk (no fast-skip)."""
    if pushed_at is None or last_commits_at is None:
        return True
    return pushed_at > last_commits_at


def _write_repo_records(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cfg: IngestCfg,
    target_id: int,
    records: _RepoRecords,
    stats: CommitStats,
    resolve_author_login: bool = True,
) -> None:
    """Consumer: per-repo transaction; pre-resolve logins; write commits;
    advance cursor. Runs on the main thread (owns the SQLite connection).

    `resolve_author_login=False` skips the email→login lookup (user-mode,
    where artifacts are stored with `author_login=NULL` by convention)."""
    if records.error is not None:
        if isinstance(records.error, EmptyRepoError):
            log.debug("skip %s: %s", records.repo_full, records.error)
        else:
            log.warning("walk %s failed, skipping: %s", records.repo_full, records.error)
        stats.skipped += 1
        return

    if resolve_author_login:
        emails = {c.author_email for c in records.commits if c.author_email}
        bulk_resolve_logins(conn, gh, emails)

    with transaction(conn):
        for rec in records.commits:
            if resolve_author_login:
                author_login = resolve_login(conn, gh=None, email=rec.author_email)
            else:
                author_login = None
            _write_commit_artifact_from_record(
                conn=conn,
                cfg=cfg,
                target_id=target_id,
                repo_full=records.repo_full,
                rec=rec,
                author_login=author_login,
                stats=stats,
            )
        q.set_repo_cursor(
            conn,
            target_id=target_id,
            full_name=records.repo_full,
            commits_walked_sha=records.head_sha,
            commits_at=_now_iso(),
        )


def _write_commit_artifact_from_record(
    *,
    conn: sqlite3.Connection,
    cfg: IngestCfg,
    target_id: int,
    repo_full: str,
    rec: _CommitRecord,
    author_login: str | None,
    stats: CommitStats,
) -> None:
    """Variant of `_write_commit_artifact` that uses pre-built chunks.

    Same content-hash short-circuit, but skips the chunk computation on
    re-ingest of an unchanged commit (the work was already done in the
    worker, but we don't insert)."""
    new_hash = _commit_content_hash(rec.diff, rec.message)
    existing_id, existing_hash = q.get_artifact_content_hash(
        conn, target_id=target_id, kind="commit", external_id=rec.sha
    )
    if existing_id is not None and existing_hash == new_hash:
        q.update_artifact_metadata(
            conn,
            artifact_id=existing_id,
            author_login=author_login,
            content_hash=new_hash,
        )
        stats.fetched += 1
        return

    artifact_id = q.upsert_artifact(
        conn,
        target_id=target_id,
        kind="commit",
        external_id=rec.sha,
        source_url=rec.url,
        repo=repo_full,
        language=_primary_language(rec.diff),
        author_email=rec.author_email,
        author_login=author_login,
        created_at=rec.author_date,
        decision=None,
        meta={
            "sha": rec.sha,
            "message_first_line": (rec.message.splitlines()[0][:200] if rec.message else ""),
        },
        content_hash=new_hash,
    )
    q.delete_chunks_for_artifact(conn, artifact_id)
    for pc in rec.pre_chunks:
        q.insert_chunk(
            conn,
            artifact_id=artifact_id,
            kind=pc.kind,
            text=pc.text,
            context=pc.context,
            language=pc.language,
        )
        if pc.kind == "code":
            stats.code_chunks += 1
        elif pc.kind == "commit_message":
            stats.message_chunks += 1
    stats.fetched += 1


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
    target_id: int,
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
            target_id=target_id,
            limit=limit,
        )
    return _ingest_commits_api(
        conn=conn,
        gh=gh,
        cache=cache,
        username=username,
        emails=emails,
        cfg=cfg,
        target_id=target_id,
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
    target_id: int,
    limit: int | None,
) -> CommitStats:
    """Parallel user-mode commits walk — mirrors `_ingest_commits_org_local`.

    Three phases:

    1. **Discover + fast-skip.** `/search/commits` enumerates repos the user
       touched. A concurrent `/repos/{r}` batch collects `pushed_at` for
       each; repos whose `pushed_at` hasn't moved past `last_commits_at`
       are skipped — no clone, just a `pushed_at` refresh in the DB.
    2. **Parallel walk.** A `ThreadPoolExecutor` runs `_walk_repo_records`
       per remaining repo: shallow clone bounded by `--shallow-since`,
       walk `git log --author <email>` + show, pre-build chunks. No DB
       access from workers.
    3. **Serial write.** As workers complete, each repo's commits land in
       their own transaction. A failure in one repo doesn't roll back
       the others; partial progress survives Ctrl-C.
    """
    stats = CommitStats()
    discovered = _discover_user_repos(gh, username=username, emails=emails, since=cfg.since)
    allowed = [r for r in discovered if _allowed_repo(r, cfg)]
    log.info(
        "user commits: discovered %d repos (%d after filters)",
        len(discovered),
        len(allowed),
    )
    if not allowed:
        return stats

    # Ensure repo rows exist so per-repo cursors persist across syncs.
    for repo_full in allowed:
        if q.get_repo(conn, target_id=target_id, full_name=repo_full) is None:
            q.upsert_repo(
                conn,
                target_id=target_id,
                full_name=repo_full,
                default_branch=None,
                pushed_at=None,
                size_kb=None,
            )

    pushed_at_by_repo = _fetch_repo_pushed_at(gh, allowed, max_workers=cfg.repo_concurrency)

    to_walk: list[dict[str, Any]] = []
    skipped_unchanged = 0
    for repo_full in allowed:
        row = q.get_repo(conn, target_id=target_id, full_name=repo_full)
        if row is None:
            continue
        pushed = pushed_at_by_repo.get(repo_full)
        last = row.get("last_commits_at")
        if not _needs_walk(pushed, last):
            if pushed is not None and pushed != row.get("pushed_at"):
                q.upsert_repo(
                    conn,
                    target_id=target_id,
                    full_name=repo_full,
                    default_branch=row.get("default_branch"),
                    pushed_at=pushed,
                    archived=bool(row.get("archived")),
                    fork=bool(row.get("fork")),
                    size_kb=row.get("size_kb"),
                )
            skipped_unchanged += 1
            continue
        to_walk.append(row)

    if skipped_unchanged:
        log.info("user commits: fast-skipped %d unchanged repos", skipped_unchanged)
    log.info(
        "user commits: walking %d repos in parallel (max %d workers)",
        len(to_walk),
        cfg.repo_concurrency,
    )
    if not to_walk:
        return stats

    sorted_emails = sorted({e.lower() for e in emails})
    token = gh.token
    workers = max(1, min(cfg.repo_concurrency, len(to_walk)))
    total = len(to_walk)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gt-user-commits") as pool:
        futures = {
            pool.submit(
                _walk_repo_records,
                cfg=cfg,
                repo_full=row["full_name"],
                last_commits_at=row.get("last_commits_at"),
                token=token,
                limit=limit,
                author_emails=sorted_emails,
            ): row
            for row in to_walk
        }
        for fut in as_completed(futures):
            completed += 1
            row = futures[fut]
            try:
                records = fut.result()
            except BaseException as e:  # noqa: BLE001 — worker errors degrade gracefully
                log.warning(
                    "user commits: [%d/%d] worker %s crashed: %s",
                    completed,
                    total,
                    row["full_name"],
                    e,
                )
                stats.skipped += 1
                continue
            pushed = pushed_at_by_repo.get(records.repo_full)
            if pushed is not None and pushed != row.get("pushed_at"):
                q.upsert_repo(
                    conn,
                    target_id=target_id,
                    full_name=records.repo_full,
                    default_branch=row.get("default_branch"),
                    pushed_at=pushed,
                    archived=bool(row.get("archived")),
                    fork=bool(row.get("fork")),
                    size_kb=row.get("size_kb"),
                )
            _write_repo_records(
                conn=conn,
                gh=gh,
                cfg=cfg,
                target_id=target_id,
                records=records,
                stats=stats,
                resolve_author_login=False,
            )
            if records.error is None:
                log.info(
                    "user commits: [%d/%d] %s: %d commits in %.1fs",
                    completed,
                    total,
                    records.repo_full,
                    len(records.commits),
                    records.elapsed_seconds,
                )
    return stats


# ---------- org-mode entry point ----------


def ingest_commits_org(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cache: RawCache,
    cfg: IngestCfg,
    target_id: int,
    limit_per_repo: int | None = None,
    pushed_at_by_repo: dict[str, str | None] | None = None,
) -> CommitStats:
    """Walk every repo's commits since its per-repo cursor.

    Author identity (`author_login`) is captured via the email→login cache.
    The optional `pushed_at_by_repo` lets the pipeline share the
    `/repos/{r}` batch with the reviews phase so we don't pay it twice.
    """
    if cfg.use_local_git_for_commits:
        return _ingest_commits_org_local(
            conn=conn,
            gh=gh,
            cfg=cfg,
            target_id=target_id,
            limit_per_repo=limit_per_repo,
            pushed_at_by_repo=pushed_at_by_repo,
        )
    return _ingest_commits_org_api(
        conn=conn,
        gh=gh,
        cache=cache,
        cfg=cfg,
        target_id=target_id,
        limit_per_repo=limit_per_repo,
    )


def _ingest_commits_org_local(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cfg: IngestCfg,
    target_id: int,
    limit_per_repo: int | None,
    pushed_at_by_repo: dict[str, str | None] | None = None,
) -> CommitStats:
    """Parallel org-mode commits walk.

    Three phases:

    1. **Fast-skip.** Batch `/repos/{r}` calls (concurrent) collect
       `pushed_at` for every allowed repo. Repos whose `pushed_at` hasn't
       moved past `last_commits_at` are skipped — no clone, no fetch,
       just a `pushed_at` refresh in the DB.
    2. **Parallel walk.** A `ThreadPoolExecutor` runs
       `_walk_repo_records` per remaining repo: shallow clone bounded
       by `--shallow-since`, walk git log + show, pre-build chunks.
       No DB access from workers.
    3. **Serial write.** As workers complete, the main thread resolves
       any new author emails (`bulk_resolve_logins`) and writes each
       repo's commits in its own transaction, advancing the per-repo
       cursor on commit. A failure in one repo doesn't roll back the
       others.

    `pushed_at_by_repo` may be supplied by the caller (e.g. pipeline
    sharing the batch with the reviews phase). When None, this function
    fetches its own batch.
    """
    stats = CommitStats()
    all_repos = q.list_repos(conn, target_id=target_id)
    allowed = [r for r in all_repos if _allowed_repo(r["full_name"], cfg)]
    if not allowed:
        return stats

    repo_names = [r["full_name"] for r in allowed]
    if pushed_at_by_repo is None:
        pushed_at_by_repo = _fetch_repo_pushed_at(gh, repo_names, max_workers=cfg.repo_concurrency)

    to_walk: list[dict[str, Any]] = []
    skipped_unchanged = 0
    for row in allowed:
        repo_full = row["full_name"]
        pushed = pushed_at_by_repo.get(repo_full)
        last = row.get("last_commits_at")
        if not _needs_walk(pushed, last):
            # Repo hasn't moved; just refresh pushed_at and skip the clone.
            if pushed is not None and pushed != row.get("pushed_at"):
                q.upsert_repo(
                    conn,
                    target_id=target_id,
                    full_name=repo_full,
                    default_branch=row.get("default_branch"),
                    pushed_at=pushed,
                    archived=bool(row.get("archived")),
                    fork=bool(row.get("fork")),
                    size_kb=row.get("size_kb"),
                )
            skipped_unchanged += 1
            continue
        to_walk.append(row)

    if skipped_unchanged:
        log.info("commits org: fast-skipped %d unchanged repos", skipped_unchanged)
    log.info(
        "commits org: walking %d repos in parallel (max %d workers)",
        len(to_walk),
        cfg.repo_concurrency,
    )

    if not to_walk:
        return stats

    token = gh.token  # resolve once; pass to every worker
    workers = max(1, min(cfg.repo_concurrency, len(to_walk)))
    total = len(to_walk)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gt-commits") as pool:
        futures = {
            pool.submit(
                _walk_repo_records,
                cfg=cfg,
                repo_full=row["full_name"],
                last_commits_at=row.get("last_commits_at"),
                token=token,
                limit=limit_per_repo,
            ): row
            for row in to_walk
        }
        for fut in as_completed(futures):
            completed += 1
            row = futures[fut]
            try:
                records = fut.result()
            except BaseException as e:  # noqa: BLE001 — worker errors degrade gracefully
                log.warning(
                    "commits org: [%d/%d] worker %s crashed: %s",
                    completed,
                    total,
                    row["full_name"],
                    e,
                )
                stats.skipped += 1
                continue
            # Refresh pushed_at on success too, so future syncs can fast-skip.
            pushed = pushed_at_by_repo.get(records.repo_full)
            if pushed is not None and pushed != row.get("pushed_at"):
                q.upsert_repo(
                    conn,
                    target_id=target_id,
                    full_name=records.repo_full,
                    default_branch=row.get("default_branch"),
                    pushed_at=pushed,
                    archived=bool(row.get("archived")),
                    fork=bool(row.get("fork")),
                    size_kb=row.get("size_kb"),
                )
            _write_repo_records(
                conn=conn,
                gh=gh,
                cfg=cfg,
                target_id=target_id,
                records=records,
                stats=stats,
            )
            if records.error is None:
                log.info(
                    "commits org: [%d/%d] %s: %d commits in %.1fs",
                    completed,
                    total,
                    records.repo_full,
                    len(records.commits),
                    records.elapsed_seconds,
                )
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
    target_id: int,
    since: str | None = None,
    limit: int | None = None,
) -> CommitStats:
    cursor = since or q.get_cursor(conn, "commits", target_id=target_id) or cfg.since
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
            target_id=target_id,
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
        q.set_cursor(conn, "commits", bumped, target_id=target_id)
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
    target_id: int,
    limit_per_repo: int | None = None,
) -> CommitStats:
    stats = CommitStats()
    for row in q.list_repos(conn, target_id=target_id):
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
                target_id=target_id,
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
        q.set_repo_cursor(conn, target_id=target_id, full_name=repo_full, commits_at=_now_iso())
    return stats
