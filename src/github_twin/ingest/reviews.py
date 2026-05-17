"""Reviews ingest.

We use `/search/issues?q=commenter:<me>+type:pr+updated:>=<cursor>` to find PRs
I've participated in, then per PR pull:
  - PR-line review comments via /repos/{r}/pulls/{n}/comments (these carry diff_hunk)
  - my own review decisions via /repos/{r}/pulls/{n}/reviews
  - general issue-style PR comments via /repos/{r}/issues/{n}/comments

All comment filtering is by login (identity-stable), not email.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from github_twin.config import IngestCfg
from github_twin.ingest.cache import RawCache
from github_twin.ingest.commits import _bump_iso, _match
from github_twin.ingest.github_client import GitHubClient, GitHubError
from github_twin.process.chunkers import chunk_pr_summary
from github_twin.process.language import language_for_path
from github_twin.store import queries as q

log = logging.getLogger(__name__)


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
    cursor = since or q.get_cursor(conn, "reviews", target_id=target_id) or cfg.since
    log.info("ingesting reviews since %s", cursor)
    stats = ReviewStats()
    newest_updated: str | None = None

    prs = list(_iter_my_prs(gh, username=username, since=cursor))
    if limit_prs:
        prs = prs[:limit_prs]

    for item in prs:
        rn = _repo_and_number(item)
        if rn is None:
            stats.skipped += 1
            continue
        repo_full, pr_number = rn

        if cfg.exclude_repos and any(_match(repo_full, pat) for pat in cfg.exclude_repos):
            stats.skipped += 1
            continue
        if cfg.include_repos and not any(_match(repo_full, pat) for pat in cfg.include_repos):
            stats.skipped += 1
            continue

        updated = item.get("updated_at")
        if updated and (newest_updated is None or updated > newest_updated):
            newest_updated = updated

        pr_title = item.get("title") or ""
        cache_key = f"{repo_full}__{pr_number}"

        # Pull review comments + my reviews + issue comments
        try:
            review_comments = list(
                gh.paginate(
                    f"/repos/{repo_full}/pulls/{pr_number}/comments",
                    params={"per_page": 100},
                )
            )
            my_reviews_all = list(
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
            stats.skipped += 1
            continue

        cache.write_json(
            "reviews",
            cache_key,
            {
                "pr": item,
                "review_comments": review_comments,
                "reviews": my_reviews_all,
                "issue_comments": issue_comments,
            },
        )

        my_reviews = [r for r in my_reviews_all if (r.get("user") or {}).get("login") == username]
        decision = _decision_from_reviews(my_reviews)

        # PR-level artifact: stores the decision so predict_review_outcome
        # can aggregate over similar past PRs at query time.
        pr_artifact_id = q.upsert_artifact(
            conn,
            target_id=target_id,
            kind="pr",
            external_id=f"{repo_full}#{pr_number}",
            source_url=item.get("html_url"),
            repo=repo_full,
            language=None,
            author_email=None,
            created_at=item.get("created_at"),
            decision=decision,
            meta={"title": pr_title, "state": item.get("state")},
        )
        # PR-summary chunk for vector retrieval (P3). Idempotent re-ingest:
        # we drop any existing chunks for this PR artifact first.
        q.delete_chunks_for_artifact(conn, pr_artifact_id)
        summary = chunk_pr_summary(
            title=pr_title,
            body=item.get("body"),
            repo=repo_full,
            pr_number=pr_number,
            source_url=item.get("html_url"),
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

        stats.prs_seen += 1

        # Per-line review comments (have diff_hunk)
        for rc in review_comments:
            if (rc.get("user") or {}).get("login") != username:
                continue
            self_id = str(rc.get("id"))
            path = rc.get("path") or ""
            lang = language_for_path(path)
            body = (rc.get("body") or "").strip()
            if not body:
                continue
            diff_hunk = rc.get("diff_hunk") or ""

            artifact_id = q.upsert_artifact(
                conn,
                target_id=target_id,
                kind="review_comment",
                external_id=self_id,
                source_url=rc.get("html_url"),
                repo=repo_full,
                language=lang,
                author_email=None,
                created_at=rc.get("created_at"),
                decision=None,
                meta={
                    "pr_number": pr_number,
                    "pr_title": pr_title,
                    "path": path,
                },
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
                },
                language=lang,
            )
            stats.review_comments += 1

        # Issue-style PR comments (no diff_hunk)
        for ic in issue_comments:
            if (ic.get("user") or {}).get("login") != username:
                continue
            self_id = str(ic.get("id"))
            body = (ic.get("body") or "").strip()
            if not body:
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
                created_at=ic.get("created_at"),
                decision=None,
                meta={"pr_number": pr_number, "pr_title": pr_title},
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
                },
                language=None,
            )
            stats.issue_comments += 1

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


def _ingest_one_pr(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cache: RawCache,
    target_id: int,
    repo_full: str,
    pr_item: dict[str, Any],
    stats: ReviewStats,
) -> None:
    pr_number = pr_item.get("number")
    if pr_number is None:
        stats.skipped += 1
        return
    pr_title = pr_item.get("title") or ""
    cache_key = f"{repo_full}__{pr_number}"

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
        stats.skipped += 1
        return

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

    # PR-level artifact carries per-reviewer decisions so predict_review_outcome
    # can filter to one reviewer in org-mode. Top-level `decision` stays NULL
    # (a multi-reviewer PR has no single decision).
    reviewer_decisions = _per_reviewer_decisions(reviews_all)
    pr_artifact_id = q.upsert_artifact(
        conn,
        target_id=target_id,
        kind="pr",
        external_id=f"{repo_full}#{pr_number}",
        source_url=pr_item.get("html_url"),
        repo=repo_full,
        language=None,
        author_email=None,
        author_login=(pr_item.get("user") or {}).get("login"),
        created_at=pr_item.get("created_at"),
        decision=None,
        meta={
            "title": pr_title,
            "state": pr_item.get("state"),
            "reviewer_decisions": reviewer_decisions,
        },
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

    # Per-line review comments (have diff_hunk). Keep ALL authors.
    for rc in review_comments:
        author_login = (rc.get("user") or {}).get("login")
        if not author_login:
            continue
        body = (rc.get("body") or "").strip()
        if not body:
            continue
        self_id = str(rc.get("id"))
        path = rc.get("path") or ""
        lang = language_for_path(path)
        diff_hunk = rc.get("diff_hunk") or ""

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

    # Issue-style PR comments (no diff_hunk). Keep ALL authors.
    for ic in issue_comments:
        author_login = (ic.get("user") or {}).get("login")
        if not author_login:
            continue
        body = (ic.get("body") or "").strip()
        if not body:
            continue
        self_id = str(ic.get("id"))

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


def ingest_reviews_org(
    *,
    conn: sqlite3.Connection,
    gh: GitHubClient,
    cache: RawCache,
    cfg: IngestCfg,
    target_id: int,
    limit_prs_per_repo: int | None = None,
) -> ReviewStats:
    """Walk every repo's recently-updated PRs, capturing comments from all
    authors. Per-repo cursor is `last_reviews_at`; advance after each repo."""
    stats = ReviewStats()
    repos = q.list_repos(conn, target_id=target_id)
    for row in repos:
        repo_full = row["full_name"]
        cursor = row.get("last_reviews_at")
        for seen, pr_item in enumerate(
            _iter_prs_until_cursor(gh, repo_full=repo_full, cursor=cursor)
        ):
            if limit_prs_per_repo is not None and seen >= limit_prs_per_repo:
                break
            _ingest_one_pr(
                conn=conn,
                gh=gh,
                cache=cache,
                target_id=target_id,
                repo_full=repo_full,
                pr_item=pr_item,
                stats=stats,
            )
        q.set_repo_cursor(conn, target_id=target_id, full_name=repo_full, reviews_at=_now_iso())
    return stats
