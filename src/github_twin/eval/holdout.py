"""Holdout selection for RAG-vs-baseline eval.

The eval needs (input, ground-truth) pairs that the retriever has NOT seen.
We carve them out by a `since` cutoff on `artifact.created_at` rather than
deleting rows: callers pass the cutoff through to retrieval pipelines so
they can drop any hit ≥ cutoff before scoring.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReviewExample:
    artifact_id: int
    diff_hunk: str
    truth_comment: str
    repo: str | None
    url: str | None
    language: str | None
    pr_title: str | None
    created_at: str


@dataclass(frozen=True)
class PRExample:
    artifact_id: int
    title: str
    body: str | None
    truth_decision: str  # 'approved' | 'changes_requested' | 'commented'
    repo: str | None
    url: str | None
    created_at: str


def iter_held_out_review_comments(
    conn: sqlite3.Connection,
    *,
    since: str,
    author_login: str | None = None,
    repo: str | None = None,
    limit: int | None = None,
) -> Iterator[ReviewExample]:
    """Review-comment artifacts at or after `since`, paired with the diff
    hunk we reacted to. Skips comments missing either side.

    When `author_login` is set, restricts to that reviewer's comments
    (org-mode use case). In user-mode the column is NULL so this filter
    drops everything — don't pass it there.

    When `repo` is set, restricts to that 'owner/name'.
    """
    sql = """
        SELECT a.id, a.repo, a.source_url, a.language, a.created_at,
               c.text, c.context_json
        FROM artifact a
        JOIN chunk c ON c.artifact_id = a.id AND c.kind = 'review_comment'
        WHERE a.kind = 'review_comment'
          AND a.created_at IS NOT NULL
          AND a.created_at >= ?
    """
    params: list[Any] = [since]
    if author_login is not None:
        sql += " AND a.author_login = ?"
        params.append(author_login)
    if repo is not None:
        sql += " AND a.repo = ?"
        params.append(repo)
    sql += " ORDER BY a.created_at ASC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    for r in conn.execute(sql, params).fetchall():
        ctx = json.loads(r["context_json"]) if r["context_json"] else {}
        hunk = ctx.get("diff_hunk") or ""
        comment = r["text"] or ""
        if not hunk.strip() or not comment.strip():
            continue
        yield ReviewExample(
            artifact_id=r["id"],
            diff_hunk=hunk,
            truth_comment=comment,
            repo=r["repo"],
            url=r["source_url"] or ctx.get("url"),
            language=r["language"] or ctx.get("language"),
            pr_title=ctx.get("pr_title"),
            created_at=r["created_at"],
        )


_DECISIONS = ("approved", "changes_requested", "commented")


def count_eligible(
    conn: sqlite3.Connection,
    *,
    since: str,
    author_login: str | None = None,
    repo: str | None = None,
) -> dict[str, int]:
    """Pre-flight: how many holdout candidates the eval would consider.

    Surfaces typo'd --author / --repo early. Uses the same filtering logic as
    the runner (PRs are author-aware: in org-mode we count only those the
    author actually decided), so the numbers match what `gt eval` would see."""
    rc = sum(
        1
        for _ in iter_held_out_review_comments(
            conn,
            since=since,
            author_login=author_login,
            repo=repo,
        )
    )
    pr = sum(
        1
        for _ in iter_held_out_prs(
            conn,
            since=since,
            author_login=author_login,
            repo=repo,
        )
    )
    return {"review_comments": rc, "decisioned_prs": pr}


def _decision_from_reviewer_meta(meta: dict[str, Any], author_login: str) -> str | None:
    """Pull a single reviewer's decision out of `meta.reviewer_decisions`."""
    for entry in meta.get("reviewer_decisions", []) or []:
        if entry.get("login") != author_login:
            continue
        state = (entry.get("state") or "").lower()
        if state == "request_changes":
            state = "changes_requested"
        if state in _DECISIONS:
            return state
    return None


def iter_held_out_prs(
    conn: sqlite3.Connection,
    *,
    since: str,
    author_login: str | None = None,
    repo: str | None = None,
    limit: int | None = None,
) -> Iterator[PRExample]:
    """PR artifacts at or after `since` with a usable decision.

    Two modes:

    - **User-mode (author_login=None)** — uses `artifact.decision` (set by
      `ingest_reviews` user-mode path).
    - **Org-mode (author_login=X)** — `artifact.decision` is NULL because
      no single decision exists. The per-reviewer state lives in
      `meta.reviewer_decisions = [{login, state, submitted_at}]`; we pull
      that author's `state` as truth and skip PRs they didn't decide.

    Pulls title/body out of the `pr_summary` chunk so they survive even
    when the cached search JSON is gone.
    """
    base_sql = """
        SELECT a.id, a.repo, a.source_url, a.decision, a.meta_json,
               a.created_at, c.text AS summary_text
        FROM artifact a
        LEFT JOIN chunk c ON c.artifact_id = a.id AND c.kind = 'pr_summary'
        WHERE a.kind = 'pr'
          AND a.created_at IS NOT NULL
          AND a.created_at >= ?
    """
    params: list[Any] = [since]
    if author_login is None:
        # User-mode: rely on the decision column.
        base_sql += " AND a.decision IN ('approved','changes_requested','commented')"
    # Org-mode (`author_login is not None`) keeps `decision` filtering off; we
    # filter in Python using `meta.reviewer_decisions`.
    if repo is not None:
        base_sql += " AND a.repo = ?"
        params.append(repo)
    sql = base_sql + " ORDER BY a.created_at ASC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    for r in conn.execute(sql, params).fetchall():
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        if author_login is None:
            truth = r["decision"]
        else:
            truth = _decision_from_reviewer_meta(meta, author_login)
            if truth is None:
                continue
        title = meta.get("title") or ""
        body: str | None = None
        if r["summary_text"]:
            parts = r["summary_text"].split("\n\n", 1)
            if len(parts) == 2:
                body = parts[1]
        if not title and not body:
            continue
        yield PRExample(
            artifact_id=r["id"],
            title=title,
            body=body,
            truth_decision=truth,
            repo=r["repo"],
            url=r["source_url"],
            created_at=r["created_at"],
        )
