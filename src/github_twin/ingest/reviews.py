"""Reviews ingest.

We use `/search/issues?q=commenter:<me>+type:pr+updated:>=<cursor>` to find PRs
I've participated in, then per PR pull:
  - PR-line review comments via /repos/{r}/pulls/{n}/comments (these carry diff_hunk)
  - my own review decisions via /repos/{r}/pulls/{n}/reviews
  - general issue-style PR comments via /repos/{r}/issues/{n}/comments

All comment filtering is by login (identity-stable), not email.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from github_twin.config import IngestCfg
from github_twin.ingest.cache import RawCache
from github_twin.ingest.commits import (
    _allowed_repo,
    _bump_iso,
    _fetch_repo_pushed_at,
    _match,
    _needs_walk,
)
from github_twin.ingest.github_client import GitHubClient, GitHubError
from github_twin.process.chunkers import chunk_pr_summary
from github_twin.process.language import language_for_path
from github_twin.store import queries as q
from github_twin.store.db import transaction

log = logging.getLogger(__name__)


def _hash_text(*parts: str) -> str:
    """sha256 over a sequence of strings, NUL-separated for unambiguous join."""
    h = hashlib.sha256()
    for i, part in enumerate(parts):
        if i:
            h.update(b"\x00")
        h.update(part.encode("utf-8", errors="replace"))
    return h.hexdigest()


@dataclass
class ReviewStats:
    prs_seen: int = 0
    review_comments: int = 0
    issue_comments: int = 0
    skipped: int = 0


def _iter_my_prs(gh: GitHubClient, *, username: str, since: str) -> Iterator[dict[str, Any]]:
    """PRs I've commented on, newest-first per GH search default."""
    qstr = f"commenter:{username} type:pr updated:>={since}"
    log.info("reviews search: %s", qstr)
    yield from gh.paginate("/search/issues", params={"q": qstr, "per_page": 100, "sort": "updated"})


def _repo_and_number(pr_search_item: dict[str, Any]) -> tuple[str, int] | None:
    repo_url = pr_search_item.get("repository_url")  # https://api.github.com/repos/o/r
    number = pr_search_item.get("number")
    if not repo_url or number is None:
        return None
    owner_repo = repo_url.rsplit("/repos/", 1)[-1]
    return owner_repo, int(number)


def _decision_from_reviews(my_reviews: list[dict[str, Any]]) -> str | None:
    """If I submitted multiple reviews, the most recent wins."""
    if not my_reviews:
        return None
    sortable = [(r.get("submitted_at") or "", (r.get("state") or "").lower()) for r in my_reviews]
    sortable.sort()
    state = sortable[-1][1]
    if state == "approved":
        return "approved"
    if state in ("changes_requested", "request_changes"):
        return "changes_requested"
    return "commented"


def ingest_reviews(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cache: RawCache,
    username: str,
    cfg: IngestCfg,
    target_id: int,
    since: str | None = None,
    limit_prs: int | None = None,
) -> ReviewStats:
    """Parallel user-mode reviews walk.

    Discovery is one `/search/issues` pass (kept serial — it's already
    one call with pagination). For each discovered PR, a worker pool
    fetches the three sub-endpoints (`/pulls/{n}/comments`,
    `/pulls/{n}/reviews`, `/issues/{n}/comments`) in parallel. Completed
    payloads are written on the main thread, each PR in its own
    transaction so partial failures don't roll back peers.

    Artifact storage mirrors org-mode (`content_hash` + `author_login`)
    so re-syncs can fast-skip unchanged comments via
    `_ingest_one_pr`'s content-hash short-circuit. `keep_only_login`
    filters comments to the target user and stamps the PR-level
    `decision` from the user's review state — preserving the
    `predict_review_outcome` contract."""
    cursor = since or q.get_cursor(conn, "reviews", target_id=target_id) or cfg.since
    log.info("ingesting reviews since %s", cursor)
    stats = ReviewStats()

    prs = list(_iter_my_prs(gh, username=username, since=cursor))
    if limit_prs:
        prs = prs[:limit_prs]

    filtered: list[tuple[str, dict[str, Any]]] = []
    newest_updated: str | None = None
    for item in prs:
        rn = _repo_and_number(item)
        if rn is None:
            stats.skipped += 1
            continue
        repo_full, _ = rn
        if cfg.exclude_repos and any(_match(repo_full, pat) for pat in cfg.exclude_repos):
            stats.skipped += 1
            continue
        if cfg.include_repos and not any(_match(repo_full, pat) for pat in cfg.include_repos):
            stats.skipped += 1
            continue
        updated = item.get("updated_at")
        if updated and (newest_updated is None or updated > newest_updated):
            newest_updated = updated
        filtered.append((repo_full, item))

    log.info(
        "user reviews: %d PRs to process (max %d workers)",
        len(filtered),
        cfg.repo_concurrency,
    )
    if not filtered:
        return stats

    workers = max(1, min(cfg.repo_concurrency, len(filtered)))
    total = len(filtered)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gt-user-reviews") as pool:
        futures = {
            pool.submit(_fetch_pr_payload_timed, gh, repo_full, pr_item): (repo_full, pr_item)
            for repo_full, pr_item in filtered
        }
        for fut in as_completed(futures):
            completed += 1
            repo_full, pr_item = futures[fut]
            try:
                payload, elapsed = fut.result()
            except BaseException as e:  # noqa: BLE001 — worker errors degrade gracefully
                log.warning(
                    "user reviews: [%d/%d] worker %s#%s crashed: %s",
                    completed,
                    total,
                    repo_full,
                    pr_item.get("number"),
                    e,
                )
                stats.skipped += 1
                continue
            if payload is None:
                log.info(
                    "user reviews: [%d/%d] %s#%s: skipped (fetch returned no payload) in %.1fs",
                    completed,
                    total,
                    repo_full,
                    pr_item.get("number"),
                    elapsed,
                )
                stats.skipped += 1
                continue
            with transaction(conn):
                _ingest_one_pr(
                    conn=conn,
                    gh=gh,
                    cache=cache,
                    target_id=target_id,
                    repo_full=repo_full,
                    pr_item=pr_item,
                    stats=stats,
                    payload=payload,
                    keep_only_login=username,
                )
            log.info(
                "user reviews: [%d/%d] %s#%s: %d review + %d issue comments in %.1fs",
                completed,
                total,
                repo_full,
                pr_item.get("number"),
                len(payload.review_comments),
                len(payload.issue_comments),
                elapsed,
            )

    if newest_updated and stats.prs_seen > 0:
        q.set_cursor(conn, "reviews", _bump_iso(newest_updated), target_id=target_id)
    return stats


# ---------- Org-mode (per-repo PRs, all comments retained) ----------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _iter_prs_until_cursor(
    gh: GitHubClient, *, repo_full: str, cursor: str | None
) -> Iterator[dict[str, Any]]:
    """List PRs sorted by updated desc, stopping the first time we hit one
    that's no newer than `cursor`. `/repos/{r}/pulls` has no `since` param,
    so we filter client-side and break early."""
    params = {
        "state": "all",
        "sort": "updated",
        "direction": "desc",
        "per_page": 100,
    }
    for item in gh.paginate(f"/repos/{repo_full}/pulls", params=params):
        updated = item.get("updated_at")
        if cursor and updated and updated <= cursor:
            return
        yield item


def _per_reviewer_decisions(reviews_all: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the per-reviewer review list into one entry per reviewer
    (most recent submission wins), mirroring how `_decision_from_reviews`
    picks a single decision per author."""
    by_login: dict[str, dict[str, Any]] = {}
    for r in reviews_all:
        login = (r.get("user") or {}).get("login")
        if not login:
            continue
        state = (r.get("state") or "").lower()
        submitted = r.get("submitted_at") or ""
        if state == "request_changes":
            state = "changes_requested"
        existing = by_login.get(login)
        if existing is None or submitted > existing["submitted_at"]:
            by_login[login] = {
                "login": login,
                "state": state,
                "submitted_at": submitted,
            }
    return sorted(by_login.values(), key=lambda d: d["submitted_at"])


@dataclass
class _PrPayload:
    """Raw PR data fetched by a worker; consumed on the writer thread."""

    pr_item: dict[str, Any]
    review_comments: list[dict[str, Any]]
    reviews_all: list[dict[str, Any]]
    issue_comments: list[dict[str, Any]]


def _fetch_pr_payload(
    gh: GitHubClient, repo_full: str, pr_item: dict[str, Any]
) -> _PrPayload | None:
    """Pull the three sub-endpoints for one PR. No DB access — safe in workers.

    Returns None on GitHubError so the caller can record the skip without
    aborting the whole repo."""
    pr_number = pr_item.get("number")
    if pr_number is None:
        return None
    try:
        review_comments = list(
            gh.paginate(
                f"/repos/{repo_full}/pulls/{pr_number}/comments",
                params={"per_page": 100},
            )
        )
        reviews_all = list(
            gh.paginate(
                f"/repos/{repo_full}/pulls/{pr_number}/reviews",
                params={"per_page": 100},
            )
        )
        issue_comments = list(
            gh.paginate(
                f"/repos/{repo_full}/issues/{pr_number}/comments",
                params={"per_page": 100},
            )
        )
    except GitHubError as e:
        log.warning("skip %s#%d: %s", repo_full, pr_number, e)
        return None
    return _PrPayload(
        pr_item=pr_item,
        review_comments=review_comments,
        reviews_all=reviews_all,
        issue_comments=issue_comments,
    )


def _ingest_one_pr(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cache: RawCache,
    target_id: int,
    repo_full: str,
    pr_item: dict[str, Any],
    stats: ReviewStats,
    payload: _PrPayload | None = None,
    keep_only_login: str | None = None,
) -> None:
    """Write one PR's artifacts (PR + comments) with content-hash skip.

    When `payload` is None, fetches the three sub-endpoints synchronously
    (legacy serial path). When supplied (parallel path), uses it and skips
    the fetch.

    `keep_only_login` filters comments to a single author and sets the
    PR-level `decision` from that author's most recent review state
    (user-mode behavior). When None, all authors are kept and `decision`
    is left None (org-mode behavior — per-reviewer decisions live in
    `meta.reviewer_decisions` instead)."""
    pr_number = pr_item.get("number")
    if pr_number is None:
        stats.skipped += 1
        return
    pr_title = pr_item.get("title") or ""
    cache_key = f"{repo_full}__{pr_number}"

    if payload is None:
        payload = _fetch_pr_payload(gh, repo_full, pr_item)
        if payload is None:
            stats.skipped += 1
            return

    review_comments = payload.review_comments
    reviews_all = payload.reviews_all
    issue_comments = payload.issue_comments

    cache.write_json(
        "reviews",
        cache_key,
        {
            "pr": pr_item,
            "review_comments": review_comments,
            "reviews": reviews_all,
            "issue_comments": issue_comments,
        },
    )
    stats.prs_seen += 1

    # PR-level artifact. content_hash covers the chunk source (title+body);
    # reviewer_decisions sits in meta and refreshes on every pass without
    # needing a chunk rewrite. User-mode also sets the artifact `decision`
    # column from the keep-only author's most recent review state, which
    # `predict_review_outcome` aggregates over.
    reviewer_decisions = _per_reviewer_decisions(reviews_all)
    decision: str | None = None
    if keep_only_login is not None:
        my_reviews = [
            r for r in reviews_all if (r.get("user") or {}).get("login") == keep_only_login
        ]
        decision = _decision_from_reviews(my_reviews)
    pr_body = pr_item.get("body") or ""
    pr_hash = _hash_text(pr_title, pr_body)
    pr_external = f"{repo_full}#{pr_number}"
    pr_meta = {
        "title": pr_title,
        "state": pr_item.get("state"),
        "reviewer_decisions": reviewer_decisions,
    }
    existing_pr_id, existing_pr_hash = q.get_artifact_content_hash(
        conn, target_id=target_id, kind="pr", external_id=pr_external
    )
    if existing_pr_id is not None and existing_pr_hash == pr_hash:
        # Title + body unchanged; refresh metadata (reviewer_decisions may
        # have moved) and keep the existing pr_summary chunk + embedding.
        q.update_artifact_metadata(
            conn,
            artifact_id=existing_pr_id,
            meta=pr_meta,
            content_hash=pr_hash,
        )
    else:
        pr_artifact_id = q.upsert_artifact(
            conn,
            target_id=target_id,
            kind="pr",
            external_id=pr_external,
            source_url=pr_item.get("html_url"),
            repo=repo_full,
            language=None,
            author_email=None,
            author_login=(pr_item.get("user") or {}).get("login"),
            created_at=pr_item.get("created_at"),
            decision=decision,
            meta=pr_meta,
            content_hash=pr_hash,
        )
        q.delete_chunks_for_artifact(conn, pr_artifact_id)
        summary = chunk_pr_summary(
            title=pr_title,
            body=pr_item.get("body"),
            repo=repo_full,
            pr_number=pr_number,
            source_url=pr_item.get("html_url"),
        )
        if summary is not None:
            q.insert_chunk(
                conn,
                artifact_id=pr_artifact_id,
                kind="pr_summary",
                text=summary.text,
                context=summary.context,
                language=None,
            )

    # Per-line review comments (have diff_hunk). Keep ALL authors in org
    # mode; user mode filters to `keep_only_login`.
    for rc in review_comments:
        author_login = (rc.get("user") or {}).get("login")
        if not author_login:
            continue
        if keep_only_login is not None and author_login != keep_only_login:
            continue
        body = (rc.get("body") or "").strip()
        if not body:
            continue
        self_id = str(rc.get("id"))
        path = rc.get("path") or ""
        lang = language_for_path(path)
        diff_hunk = rc.get("diff_hunk") or ""
        rc_hash = _hash_text(body, diff_hunk)

        existing_id, existing_hash = q.get_artifact_content_hash(
            conn, target_id=target_id, kind="review_comment", external_id=self_id
        )
        if existing_id is not None and existing_hash == rc_hash:
            q.update_artifact_metadata(
                conn,
                artifact_id=existing_id,
                author_login=author_login,
                content_hash=rc_hash,
            )
            stats.review_comments += 1
            continue

        artifact_id = q.upsert_artifact(
            conn,
            target_id=target_id,
            kind="review_comment",
            external_id=self_id,
            source_url=rc.get("html_url"),
            repo=repo_full,
            language=lang,
            author_email=None,
            author_login=author_login,
            created_at=rc.get("created_at"),
            decision=None,
            meta={"pr_number": pr_number, "pr_title": pr_title, "path": path},
            content_hash=rc_hash,
        )
        q.delete_chunks_for_artifact(conn, artifact_id)
        q.insert_chunk(
            conn,
            artifact_id=artifact_id,
            kind="review_comment",
            text=body,
            context={
                "diff_hunk": diff_hunk,
                "path": path,
                "language": lang,
                "pr_title": pr_title,
                "pr_number": pr_number,
                "repo": repo_full,
                "url": rc.get("html_url"),
                "author_login": author_login,
            },
            language=lang,
        )
        stats.review_comments += 1

    # Issue-style PR comments (no diff_hunk). Keep ALL authors in org
    # mode; user mode filters to `keep_only_login`.
    for ic in issue_comments:
        author_login = (ic.get("user") or {}).get("login")
        if not author_login:
            continue
        if keep_only_login is not None and author_login != keep_only_login:
            continue
        body = (ic.get("body") or "").strip()
        if not body:
            continue
        self_id = str(ic.get("id"))
        ic_hash = _hash_text(body)

        existing_id, existing_hash = q.get_artifact_content_hash(
            conn, target_id=target_id, kind="issue_comment", external_id=self_id
        )
        if existing_id is not None and existing_hash == ic_hash:
            q.update_artifact_metadata(
                conn,
                artifact_id=existing_id,
                author_login=author_login,
                content_hash=ic_hash,
            )
            stats.issue_comments += 1
            continue

        artifact_id = q.upsert_artifact(
            conn,
            target_id=target_id,
            kind="issue_comment",
            external_id=self_id,
            source_url=ic.get("html_url"),
            repo=repo_full,
            language=None,
            author_email=None,
            author_login=author_login,
            created_at=ic.get("created_at"),
            decision=None,
            meta={"pr_number": pr_number, "pr_title": pr_title},
            content_hash=ic_hash,
        )
        q.delete_chunks_for_artifact(conn, artifact_id)
        q.insert_chunk(
            conn,
            artifact_id=artifact_id,
            kind="review_comment",
            text=body,
            context={
                "diff_hunk": None,
                "pr_title": pr_title,
                "pr_number": pr_number,
                "repo": repo_full,
                "url": ic.get("html_url"),
                "author_login": author_login,
            },
            language=None,
        )
        stats.issue_comments += 1


@dataclass
class _RepoReviewRecords:
    """Worker output: pre-fetched PR payloads for one repo."""

    repo_full: str
    payloads: list[_PrPayload] = field(default_factory=list)
    error: BaseException | None = None
    elapsed_seconds: float = 0.0


def _walk_repo_reviews(
    *,
    gh: GitHubClient,
    repo_full: str,
    cursor: str | None,
    limit_prs: int | None,
) -> _RepoReviewRecords:
    """Worker: enumerate PRs above the cursor and fetch each PR's
    sub-endpoint payload. No DB or RawCache writes — pure fetch."""
    records = _RepoReviewRecords(repo_full=repo_full)
    t0 = time.monotonic()
    try:
        for seen, pr_item in enumerate(
            _iter_prs_until_cursor(gh, repo_full=repo_full, cursor=cursor)
        ):
            if limit_prs is not None and seen >= limit_prs:
                break
            payload = _fetch_pr_payload(gh, repo_full, pr_item)
            if payload is not None:
                records.payloads.append(payload)
    except GitHubError as e:
        records.error = e
    records.elapsed_seconds = time.monotonic() - t0
    return records


def _fetch_pr_payload_timed(
    gh: GitHubClient, repo_full: str, pr_item: dict[str, Any]
) -> tuple[_PrPayload | None, float]:
    """User-mode worker wrapper: time the per-PR fetch without touching
    `_fetch_pr_payload`'s signature."""
    t0 = time.monotonic()
    payload = _fetch_pr_payload(gh, repo_full, pr_item)
    return payload, time.monotonic() - t0


def ingest_reviews_org(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cache: RawCache,
    cfg: IngestCfg,
    target_id: int,
    limit_prs_per_repo: int | None = None,
    pushed_at_by_repo: dict[str, str | None] | None = None,
) -> ReviewStats:
    """Parallel org-mode reviews walk.

    Mirrors the commits-org pipeline (fast-skip → parallel fetch → serial
    write). Per-repo cursor is `last_reviews_at`; advance after each repo
    in its own transaction so partial failures don't roll back peers."""
    stats = ReviewStats()
    all_repos = q.list_repos(conn, target_id=target_id, include_archived=cfg.include_archived)
    allowed = [r for r in all_repos if _allowed_repo(r["full_name"], cfg)]
    if not allowed:
        return stats

    repo_names = [r["full_name"] for r in allowed]
    if pushed_at_by_repo is None:
        pushed_at_by_repo = _fetch_repo_pushed_at(gh, repo_names, max_workers=cfg.repo_concurrency)

    to_walk: list[dict[str, Any]] = []
    skipped_unchanged = 0
    for row in allowed:
        pushed = pushed_at_by_repo.get(row["full_name"])
        last_reviews = row.get("last_reviews_at")
        if not _needs_walk(pushed, last_reviews):
            skipped_unchanged += 1
            continue
        to_walk.append(row)

    if skipped_unchanged:
        log.info("reviews org: fast-skipped %d unchanged repos", skipped_unchanged)
    log.info(
        "reviews org: walking %d repos in parallel (max %d workers)",
        len(to_walk),
        cfg.repo_concurrency,
    )

    if not to_walk:
        return stats

    workers = max(1, min(cfg.repo_concurrency, len(to_walk)))
    total = len(to_walk)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gt-reviews") as pool:
        futures = {
            pool.submit(
                _walk_repo_reviews,
                gh=gh,
                repo_full=row["full_name"],
                cursor=row.get("last_reviews_at"),
                limit_prs=limit_prs_per_repo,
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
                    "reviews org: [%d/%d] worker %s crashed: %s",
                    completed,
                    total,
                    row["full_name"],
                    e,
                )
                stats.skipped += 1
                continue
            if records.error is not None:
                log.warning(
                    "reviews org: [%d/%d] fetch %s failed in %.1fs: %s",
                    completed,
                    total,
                    records.repo_full,
                    records.elapsed_seconds,
                    records.error,
                )
                stats.skipped += 1
                continue
            with transaction(conn):
                for payload in records.payloads:
                    _ingest_one_pr(
                        conn=conn,
                        gh=gh,
                        cache=cache,
                        target_id=target_id,
                        repo_full=records.repo_full,
                        pr_item=payload.pr_item,
                        stats=stats,
                        payload=payload,
                    )
                q.set_repo_cursor(
                    conn,
                    target_id=target_id,
                    full_name=records.repo_full,
                    reviews_at=_now_iso(),
                )
            n_comments = sum(
                len(p.review_comments) + len(p.issue_comments) for p in records.payloads
            )
            log.info(
                "reviews org: [%d/%d] %s: %d PRs / %d comments in %.1fs",
                completed,
                total,
                records.repo_full,
                len(records.payloads),
                n_comments,
                records.elapsed_seconds,
            )
    return stats
