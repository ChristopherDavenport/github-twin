"""All SQL access for github-twin. Keep SQL strings here, not scattered across modules.

Multi-target shape:

- Every per-target writer takes `target_id` and stamps it.
- `sync_cursor` is keyed `(target_id, resource)`; pass `target_id=0` for
  global resources like `embed_text_version`.
- Search queries accept an optional `target_id` filter. When `None`
  (coalesce mode), results are deduped on
  `(artifact.kind, artifact.external_id, chunk_idx)` so the same commit
  ingested under multiple targets contributes one hit. When explicit,
  results are pre-narrowed to that target and no dedup is needed.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# Sentinel target_id for cross-target sync cursors (e.g. embed_text_version).
GLOBAL_TARGET_ID = 0

# ---------- Helpers ----------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _pack_vec(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _json_or_none(obj: Any) -> str | None:
    return None if obj is None else json.dumps(obj, separators=(",", ":"))


# ---------- Dataclasses (return shapes) ----------


@dataclass
class ArtifactRow:
    id: int
    kind: str
    external_id: str | None
    source_url: str | None
    repo: str | None
    language: str | None
    created_at: str | None
    decision: str | None
    meta: dict[str, Any]


@dataclass
class ChunkRow:
    id: int
    artifact_id: int
    kind: str
    text: str
    context: dict[str, Any]
    embed_model: str | None
    summary: str | None = None


@dataclass
class SearchHit:
    chunk_id: int
    artifact_id: int
    distance: float
    text: str
    context: dict[str, Any]
    artifact_kind: str
    artifact_language: str | None
    artifact_repo: str | None
    artifact_source_url: str | None
    artifact_decision: str | None
    target_id: int | None = None
    target_name: str | None = None
    target_kind: str | None = None


# ---------- Target ----------


def upsert_target(
    conn: sqlite3.Connection,
    *,
    kind: str,
    name: str,
    external_id: int,
    emails: list[str] | None,
) -> int:
    """Insert or refresh a target keyed by (kind, name). Returns its id."""
    emails_json = json.dumps(emails) if emails is not None else None
    cur = conn.execute(
        "INSERT INTO target (kind, name, external_id, emails_json, discovered_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(kind, name) DO UPDATE SET "
        "external_id=excluded.external_id, emails_json=excluded.emails_json, "
        "discovered_at=excluded.discovered_at "
        "RETURNING id",
        (kind, name, external_id, emails_json, _now_iso()),
    )
    row_id: int = cur.fetchone()["id"]
    return row_id


def get_all_targets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, kind, name, external_id, emails_json, discovered_at FROM target ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_target_by_id(conn: sqlite3.Connection, target_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, kind, name, external_id, emails_json, discovered_at FROM target WHERE id = ?",
        (target_id,),
    ).fetchone()
    return dict(row) if row else None


def get_targets_by_name(conn: sqlite3.Connection, name: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, kind, name, external_id, emails_json, discovered_at "
        "FROM target WHERE name = ? ORDER BY id",
        (name,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_targets_by_kind(conn: sqlite3.Connection, kind: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, kind, name, external_id, emails_json, discovered_at "
        "FROM target WHERE kind = ? ORDER BY id",
        (kind,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_target(conn: sqlite3.Connection, target_id: int) -> None:
    """Remove a target and cascade-delete its artifacts/repos/chunks/vectors.

    Cleans up `sync_cursor` rows too (no FK there). Vectors get cleared via
    the chunk ON DELETE CASCADE chain, but the `vec_chunk` virtual table
    holds its own rows — wipe them explicitly first.
    """
    chunk_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT c.id FROM chunk c "
            "JOIN artifact a ON a.id = c.artifact_id "
            "WHERE a.target_id = ?",
            (target_id,),
        ).fetchall()
    ]
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(f"DELETE FROM vec_chunk WHERE chunk_id IN ({placeholders})", chunk_ids)
    conn.execute("DELETE FROM sync_cursor WHERE target_id = ?", (target_id,))
    conn.execute("DELETE FROM target WHERE id = ?", (target_id,))


# ---------- Repos (org-mode / repo-mode: one row per (target, repo)) ----------


def upsert_repo(
    conn: sqlite3.Connection,
    *,
    target_id: int,
    full_name: str,
    default_branch: str | None,
    pushed_at: str | None,
    archived: bool = False,
    fork: bool = False,
    size_kb: int | None = None,
) -> None:
    """Insert or refresh repo metadata. Cursors (head_sha, last_*_at) are not
    touched here — those advance through `set_repo_cursor` after each phase."""
    conn.execute(
        "INSERT INTO repo (target_id, full_name, default_branch, pushed_at, "
        "archived, fork, size_kb) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(target_id, full_name) DO UPDATE SET "
        "default_branch=excluded.default_branch, pushed_at=excluded.pushed_at, "
        "archived=excluded.archived, fork=excluded.fork, size_kb=excluded.size_kb",
        (target_id, full_name, default_branch, pushed_at, int(archived), int(fork), size_kb),
    )


def get_repo(
    conn: sqlite3.Connection,
    *,
    target_id: int,
    full_name: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM repo WHERE target_id = ? AND full_name = ?",
        (target_id, full_name),
    ).fetchone()
    return dict(row) if row else None


def list_repos(
    conn: sqlite3.Connection,
    *,
    target_id: int | None = None,
    include_archived: bool = False,
    include_forks: bool = False,
) -> list[dict[str, Any]]:
    """List repos. `target_id=None` returns rows from all targets (each
    row carries `target_id`); explicit `target_id` narrows."""
    where: list[str] = []
    params: list[Any] = []
    if target_id is not None:
        where.append("target_id = ?")
        params.append(target_id)
    if not include_archived:
        where.append("archived = 0")
    if not include_forks:
        where.append("fork = 0")
    sql = "SELECT * FROM repo"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY full_name"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def all_cached_repos(conn: sqlite3.Connection) -> set[str]:
    """Union of `full_name` across every target. Used by `prune_cache` so
    a clone is kept as long as *any* target still references it."""
    rows = conn.execute("SELECT DISTINCT full_name FROM repo").fetchall()
    return {r["full_name"] for r in rows}


def set_repo_cursor(
    conn: sqlite3.Connection,
    *,
    target_id: int,
    full_name: str,
    head_sha: str | None = None,
    files_at: str | None = None,
    commits_at: str | None = None,
    commits_walked_sha: str | None = None,
    reviews_at: str | None = None,
) -> None:
    """Partial update: only the supplied phase cursors advance.

    Each phase (files / commits / reviews) sets its own cursor on success so
    that a partial failure in one phase doesn't lose progress in the others.
    """
    fields: list[str] = []
    params: list[Any] = []
    if head_sha is not None:
        fields.append("head_sha=?")
        params.append(head_sha)
    if files_at is not None:
        fields.append("last_files_at=?")
        params.append(files_at)
    if commits_at is not None:
        fields.append("last_commits_at=?")
        params.append(commits_at)
    if commits_walked_sha is not None:
        fields.append("last_commits_walked_sha=?")
        params.append(commits_walked_sha)
    if reviews_at is not None:
        fields.append("last_reviews_at=?")
        params.append(reviews_at)
    if not fields:
        return
    params.extend([target_id, full_name])
    conn.execute(
        f"UPDATE repo SET {', '.join(fields)} WHERE target_id=? AND full_name=?",
        params,
    )


# ---------- Email → GitHub login cache ----------


def get_email_login(conn: sqlite3.Connection, email: str) -> tuple[str | None, bool]:
    """Return (login, resolved). resolved=False means we've never tried this email.

    A resolved row with login=None is a deliberate cached miss — caller should
    not re-query the GitHub API for it.
    """
    row = conn.execute(
        "SELECT login FROM email_login_map WHERE email=?", (email.lower(),)
    ).fetchone()
    if row is None:
        return (None, False)
    return (row["login"], True)


def upsert_email_login(
    conn: sqlite3.Connection, *, email: str, login: str | None, source: str
) -> None:
    conn.execute(
        "INSERT INTO email_login_map (email, login, resolved_at, source) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(email) DO UPDATE SET "
        "login=excluded.login, resolved_at=excluded.resolved_at, "
        "source=excluded.source",
        (email.lower(), login, _now_iso(), source),
    )


# ---------- Sync cursors ----------


def get_cursor(
    conn: sqlite3.Connection,
    resource: str,
    *,
    target_id: int = GLOBAL_TARGET_ID,
) -> str | None:
    row = conn.execute(
        "SELECT cursor FROM sync_cursor WHERE target_id = ? AND resource = ?",
        (target_id, resource),
    ).fetchone()
    return row["cursor"] if row else None


def set_cursor(
    conn: sqlite3.Connection,
    resource: str,
    cursor: str,
    *,
    target_id: int = GLOBAL_TARGET_ID,
) -> None:
    conn.execute(
        "INSERT INTO sync_cursor (target_id, resource, cursor, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(target_id, resource) DO UPDATE SET cursor=excluded.cursor, "
        "updated_at=excluded.updated_at",
        (target_id, resource, cursor, _now_iso()),
    )


# ---------- Artifacts ----------


def upsert_artifact(
    conn: sqlite3.Connection,
    *,
    target_id: int,
    kind: str,
    external_id: str,
    source_url: str | None,
    repo: str | None,
    language: str | None,
    author_email: str | None,
    created_at: str | None,
    decision: str | None,
    meta: dict[str, Any] | None,
    author_login: str | None = None,
    content_hash: str | None = None,
) -> int:
    """Insert or update an artifact keyed by (target_id, kind, external_id).
    Returns artifact id."""
    cur = conn.execute(
        "INSERT INTO artifact (target_id, kind, external_id, source_url, repo, language, "
        "author_email, author_login, created_at, decision, meta_json, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(target_id, kind, external_id) DO UPDATE SET "
        "source_url=excluded.source_url, repo=excluded.repo, language=excluded.language, "
        "author_email=excluded.author_email, author_login=excluded.author_login, "
        "created_at=excluded.created_at, decision=excluded.decision, "
        "meta_json=excluded.meta_json, content_hash=excluded.content_hash "
        "RETURNING id",
        (
            target_id,
            kind,
            external_id,
            source_url,
            repo,
            language,
            author_email,
            author_login,
            created_at,
            decision,
            _json_or_none(meta),
            content_hash,
        ),
    )
    row_id: int = cur.fetchone()["id"]
    return row_id


def get_artifact_content_hash(
    conn: sqlite3.Connection,
    *,
    target_id: int,
    kind: str,
    external_id: str,
) -> tuple[int | None, str | None]:
    """Look up (artifact_id, content_hash) for an existing artifact.

    Returns (None, None) when the artifact doesn't exist yet, and
    (id, None) for pre-content-hash rows. Callers use the hash to short-
    circuit chunk wipe+re-insert when the source content is unchanged.
    """
    row = conn.execute(
        "SELECT id, content_hash FROM artifact "
        "WHERE target_id = ? AND kind = ? AND external_id = ?",
        (target_id, kind, external_id),
    ).fetchone()
    if row is None:
        return None, None
    return row["id"], row["content_hash"]


def update_artifact_metadata(
    conn: sqlite3.Connection,
    *,
    artifact_id: int,
    author_login: str | None = None,
    created_at: str | None = None,
    meta: dict[str, Any] | None = None,
    content_hash: str | None = None,
    source_url: str | None = None,
) -> None:
    """Light update of artifact metadata WITHOUT touching its chunks.

    Used on the content-hash skip path: the diff/body hasn't changed, but
    we may have newly resolved an author_login or want to bump content_hash
    onto a legacy row. Passing None for a field leaves it unchanged.
    """
    sets: list[str] = []
    params: list[Any] = []
    if author_login is not None:
        sets.append("author_login = ?")
        params.append(author_login)
    if created_at is not None:
        sets.append("created_at = ?")
        params.append(created_at)
    if meta is not None:
        sets.append("meta_json = ?")
        params.append(_json_or_none(meta))
    if content_hash is not None:
        sets.append("content_hash = ?")
        params.append(content_hash)
    if source_url is not None:
        sets.append("source_url = ?")
        params.append(source_url)
    if not sets:
        return
    params.append(artifact_id)
    conn.execute(f"UPDATE artifact SET {', '.join(sets)} WHERE id = ?", params)


def list_rules(
    conn: sqlite3.Connection,
    *,
    target_id: int | None = None,
    language: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
    limit: int = 20,
    chunk_kind: str = "rule",
) -> list[dict[str, Any]]:
    """Return distilled rules with their stored metadata + chunk text.

    `chunk_kind` discriminates between review-derived rules (`'rule'`,
    default) and code-pattern rules (`'code_rule'`). Both share
    `artifact.kind='rule'`.

    `target_id=None` coalesces across every target; explicit narrows.
    Cross-target duplicates collapse via the `(kind, external_id)` index
    on the underlying member-chunk-id hash, so two targets that distilled
    the same rule key surface it once.
    """
    where = ["a.kind = 'rule'", "c.kind = ?"]
    params: list[Any] = [chunk_kind]
    if target_id is not None:
        where.append("a.target_id = ?")
        params.append(target_id)
    if language is not None:
        where.append("a.language = ?")
        params.append(language)
    if repo is not None:
        where.append("a.repo = ?")
        params.append(repo)
    if author_login is not None:
        where.append("a.author_login = ?")
        params.append(author_login)
    if target_id is None:
        # Coalesce dedup: one rule per (kind, external_id) across targets.
        inner = (
            "SELECT a.id, a.language, a.meta_json, c.text, "
            "ROW_NUMBER() OVER (PARTITION BY a.kind, a.external_id "
            "ORDER BY a.target_id, a.id) AS rn "
            "FROM artifact a JOIN chunk c ON c.artifact_id = a.id "
            f"WHERE {' AND '.join(where)}"
        )
        sql = (
            f"SELECT id, language, meta_json, text FROM ({inner}) WHERE rn = 1 ORDER BY id LIMIT ?"
        )
    else:
        sql = (
            "SELECT a.id, a.language, a.meta_json, c.text "
            "FROM artifact a JOIN chunk c ON c.artifact_id = a.id "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY a.id LIMIT ?"
        )
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        out.append(
            {
                "rule": r["text"],
                "language": r["language"],
                "examples": meta.get("example_quotes", []),
                "cluster_size": meta.get("cluster_size", 0),
                "repos": meta.get("member_repos", []),
                "urls": meta.get("member_urls", []),
            }
        )
    return out


def artifact_exists(
    conn: sqlite3.Connection,
    kind: str,
    external_id: str,
    *,
    target_id: int | None = None,
) -> bool:
    if target_id is None:
        row = conn.execute(
            "SELECT 1 FROM artifact WHERE kind=? AND external_id=? LIMIT 1",
            (kind, external_id),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT 1 FROM artifact WHERE target_id=? AND kind=? AND external_id=? LIMIT 1",
            (target_id, kind, external_id),
        ).fetchone()
    return row is not None


# ---------- Chunks ----------


def insert_chunk(
    conn: sqlite3.Connection,
    *,
    artifact_id: int,
    kind: str,
    text: str,
    context: dict[str, Any] | None,
    language: str | None = None,
) -> int:
    # node_kind / symbol_name ride on the context dict for AST chunks and are
    # absent for line-window / non-code chunks. Pulling them here keeps the
    # ingest call-sites unchanged.
    node_kind = context.get("node_kind") if context else None
    symbol_name = context.get("symbol_name") if context else None
    cur = conn.execute(
        "INSERT INTO chunk (artifact_id, kind, text, context_json, language, "
        "node_kind, symbol_name) VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
        (
            artifact_id,
            kind,
            text,
            _json_or_none(context),
            language,
            node_kind,
            symbol_name,
        ),
    )
    row_id: int = cur.fetchone()["id"]
    return row_id


def delete_chunks_for_artifact(conn: sqlite3.Connection, artifact_id: int) -> None:
    """Remove old chunks (and their vectors) so re-ingest is idempotent per artifact.

    Order matters: vec_chunk first (no triggers), then chunk — the chunk DELETE
    fires the chunk_ad trigger which removes the corresponding chunk_fts rows.
    Do not add a manual `DELETE FROM chunk_fts`; external-content FTS5 tables
    reject direct deletes.
    """
    rows = conn.execute("SELECT id FROM chunk WHERE artifact_id=?", (artifact_id,)).fetchall()
    if not rows:
        return
    ids = [r["id"] for r in rows]
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM vec_chunk WHERE chunk_id IN ({placeholders})", ids)
    conn.execute("DELETE FROM chunk WHERE artifact_id=?", (artifact_id,))


def pending_embed_chunks(conn: sqlite3.Connection, batch_size: int = 64) -> Iterable[ChunkRow]:
    """Iterate chunks with no embedding yet, in batches. Caller embeds + writes back."""
    last_id = 0
    while True:
        rows = conn.execute(
            "SELECT id, artifact_id, kind, text, context_json, summary, embed_model FROM chunk "
            "WHERE embed_model IS NULL AND id > ? ORDER BY id LIMIT ?",
            (last_id, batch_size),
        ).fetchall()
        if not rows:
            return
        for r in rows:
            yield ChunkRow(
                id=r["id"],
                artifact_id=r["artifact_id"],
                kind=r["kind"],
                text=r["text"],
                context=json.loads(r["context_json"]) if r["context_json"] else {},
                summary=r["summary"],
                embed_model=r["embed_model"],
            )
        last_id = rows[-1]["id"]


def pending_summary_chunks(
    conn: sqlite3.Connection,
    *,
    kinds: tuple[str, ...] = ("code", "file", "code_rule"),
    batch_size: int = 32,
) -> Iterable[ChunkRow]:
    """Iterate chunks with no LLM summary yet, scoped to `kinds`.

    Defaults exclude `commit_message` / `pr_summary` / `review_comment` /
    `rule` — those are already natural-language, so summarization is
    redundant and would just compress with information loss. Code-shaped
    kinds are where the NL summary bridges the embedder's gap between
    "Eq instance for a wrapper case class" and the identifiers in the
    actual definition.
    """
    placeholders = ",".join("?" * len(kinds))
    last_id = 0
    while True:
        rows = conn.execute(
            f"SELECT id, artifact_id, kind, text, context_json, summary, embed_model "
            f"FROM chunk WHERE summary IS NULL AND kind IN ({placeholders}) "
            f"AND id > ? ORDER BY id LIMIT ?",
            (*kinds, last_id, batch_size),
        ).fetchall()
        if not rows:
            return
        for r in rows:
            yield ChunkRow(
                id=r["id"],
                artifact_id=r["artifact_id"],
                kind=r["kind"],
                text=r["text"],
                context=json.loads(r["context_json"]) if r["context_json"] else {},
                summary=r["summary"],
                embed_model=r["embed_model"],
            )
        last_id = rows[-1]["id"]


def write_chunk_summary(conn: sqlite3.Connection, *, chunk_id: int, summary: str) -> None:
    """Persist the LLM-generated summary. Trim before write — long
    summaries pollute the embed prefix and BM25 index alike."""
    summary = summary.strip()
    conn.execute("UPDATE chunk SET summary=? WHERE id=?", (summary, chunk_id))


def clear_chunk_summaries(
    conn: sqlite3.Connection,
    *,
    kinds: tuple[str, ...] | None = None,
) -> int:
    """Wipe summaries so `gt summarize --rebuild` re-runs them. Returns
    the count of rows reset."""
    if kinds is None:
        cur = conn.execute("UPDATE chunk SET summary=NULL WHERE summary IS NOT NULL")
    else:
        placeholders = ",".join("?" * len(kinds))
        cur = conn.execute(
            f"UPDATE chunk SET summary=NULL WHERE summary IS NOT NULL AND kind IN ({placeholders})",
            kinds,
        )
    return cur.rowcount


def write_embedding(
    conn: sqlite3.Connection,
    *,
    chunk_id: int,
    embedding: list[float],
    model_id: str,
) -> None:
    # sqlite-vec virtual tables don't support ON CONFLICT, so we replace explicitly.
    conn.execute("DELETE FROM vec_chunk WHERE chunk_id=?", (chunk_id,))
    conn.execute(
        "INSERT INTO vec_chunk (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, _pack_vec(embedding)),
    )
    conn.execute("UPDATE chunk SET embed_model=? WHERE id=?", (model_id, chunk_id))


# ---------- Search ----------


def _select_columns() -> str:
    """Shared SELECT-list for hit construction. Joins `target` so each hit
    can carry the source target name back to the caller."""
    return (
        "c.id AS chunk_id, c.artifact_id, c.text, c.context_json, "
        "c.language AS chunk_language, "
        "a.kind AS artifact_kind, a.repo AS artifact_repo, "
        "a.source_url AS artifact_source_url, a.decision AS artifact_decision, "
        "a.kind AS dedup_kind, a.external_id AS dedup_external_id, "
        "a.target_id AS target_id, t.name AS target_name, t.kind AS target_kind"
    )


def _hit_from_row(r: sqlite3.Row, distance: float) -> SearchHit:
    return SearchHit(
        chunk_id=r["chunk_id"],
        artifact_id=r["artifact_id"],
        distance=distance,
        text=r["text"],
        context=json.loads(r["context_json"]) if r["context_json"] else {},
        artifact_kind=r["artifact_kind"],
        artifact_language=r["chunk_language"],
        artifact_repo=r["artifact_repo"],
        artifact_source_url=r["artifact_source_url"],
        artifact_decision=r["artifact_decision"],
        target_id=r["target_id"],
        target_name=r["target_name"],
        target_kind=r["target_kind"],
    )


def _dedup_hits(hits: list[SearchHit], conn: sqlite3.Connection) -> list[SearchHit]:
    """Coalesce-mode dedup: when the same commit / PR / file is ingested
    under multiple targets, keep the first occurrence by ranked order.

    The dedup key is `(artifact_kind, artifact_external_id, chunk_index)`.
    `chunk_index` is fetched on-demand for the candidate set — multi-chunk
    artifacts (e.g. a commit with N diff chunks) need it to distinguish
    each chunk's identity across targets.
    """
    if not hits:
        return hits
    chunk_ids = list({h.chunk_id for h in hits})
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT c.id AS chunk_id, c.artifact_id, a.kind AS akind, "
        f"a.external_id AS ext_id, "
        f"ROW_NUMBER() OVER (PARTITION BY c.artifact_id ORDER BY c.id) AS chunk_idx "
        f"FROM chunk c JOIN artifact a ON a.id = c.artifact_id "
        f"WHERE c.id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    chunk_idx_by_id = {r["chunk_id"]: (r["akind"], r["ext_id"], r["chunk_idx"]) for r in rows}
    seen: set[tuple[str, str | None, int]] = set()
    out: list[SearchHit] = []
    for h in hits:
        key = chunk_idx_by_id.get(h.chunk_id)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def vector_search(
    conn: sqlite3.Connection,
    *,
    query_vec: list[float],
    chunk_kind: str,
    language: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
    node_kind: str | None = None,
    target_id: int | None = None,
    k: int = 5,
) -> list[SearchHit]:
    """KNN search restricted to chunks of a given kind (and optionally
    language / repo / author_login / node_kind / target_id).

    `target_id=None` searches every target with on-the-fly dedup of
    cross-target duplicates. An explicit `target_id` narrows the
    candidate set (cheaper, no dedup needed).

    sqlite-vec supports `chunk_id IN (subquery)` as a pre-filter on the KNN,
    so we restrict the candidate set *before* the vector match. Crucial for
    rare kinds (e.g., review_comments make up ~5% of chunks — without the
    pre-filter, top-k unfiltered will be almost entirely code chunks).
    """
    where = ["c.kind = ?"]
    cand_params: list[Any] = [chunk_kind]
    if language is not None:
        where.append("c.language = ?")
        cand_params.append(language)
    if node_kind is not None:
        where.append("c.node_kind = ?")
        cand_params.append(node_kind)
    needs_artifact_join = repo is not None or author_login is not None or target_id is not None
    if target_id is not None:
        where.append("a.target_id = ?")
        cand_params.append(target_id)
    if repo is not None:
        where.append("a.repo = ?")
        cand_params.append(repo)
    if author_login is not None:
        where.append("a.author_login = ?")
        cand_params.append(author_login)

    join_artifact = " JOIN artifact a ON a.id = c.artifact_id" if needs_artifact_join else ""
    candidates_sql = f"SELECT c.id FROM chunk c{join_artifact} WHERE " + " AND ".join(where)

    # Overscan when coalescing so dedup still has enough to return k.
    fetch_k = k * 3 if target_id is None else k

    rows = conn.execute(
        f"""
        SELECT
          v.chunk_id, v.distance,
          c.artifact_id, c.text, c.context_json, c.language AS chunk_language,
          a.kind AS artifact_kind,
          a.repo AS artifact_repo, a.source_url AS artifact_source_url,
          a.decision AS artifact_decision,
          a.target_id AS target_id, t.name AS target_name, t.kind AS target_kind
        FROM vec_chunk v
        JOIN chunk c ON c.id = v.chunk_id
        JOIN artifact a ON a.id = c.artifact_id
        JOIN target t ON t.id = a.target_id
        WHERE v.chunk_id IN ({candidates_sql})
          AND v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
        """,
        (*cand_params, _pack_vec(query_vec), fetch_k),
    ).fetchall()

    hits = [
        SearchHit(
            chunk_id=r["chunk_id"],
            artifact_id=r["artifact_id"],
            distance=r["distance"],
            text=r["text"],
            context=json.loads(r["context_json"]) if r["context_json"] else {},
            artifact_kind=r["artifact_kind"],
            artifact_language=r["chunk_language"],
            artifact_repo=r["artifact_repo"],
            artifact_source_url=r["artifact_source_url"],
            artifact_decision=r["artifact_decision"],
            target_id=r["target_id"],
            target_name=r["target_name"],
            target_kind=r["target_kind"],
        )
        for r in rows
    ]
    if target_id is None:
        hits = _dedup_hits(hits, conn)[:k]
    return hits


def _fts_escape(text: str) -> str:
    """Wrap each whitespace-split token in double-quotes so FTS5 treats it as
    a literal phrase. Without this, callers' raw text containing `:`, `(`,
    or reserved words like `AND`/`OR`/`NEAR` raises `fts5: syntax error`.
    Embedded double-quotes inside a token get doubled per FTS5 quoting rules.
    """
    tokens = [t for t in text.split() if t]
    if not tokens:
        return '""'
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def _fts_quote(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def _fts_match_from_groups(groups: list[list[str]]) -> str:
    """Build an FTS5 MATCH expression from per-token OR-groups.

    Each group becomes `("orig" OR "alt1" OR "alt2")`; groups are joined
    with explicit `AND`. Implicit-AND (space-separated) works for bare
    phrases but breaks the moment a parenthesized OR-expression is in
    the mix — SQLite reports `syntax error near "OR"`. Always-explicit
    `AND` is the only form that round-trips reliably for our shape.
    Empty groups → `""` so FTS5 still parses.
    """
    if not groups:
        return '""'
    out: list[str] = []
    for grp in groups:
        terms = [t for t in grp if t]
        if not terms:
            continue
        # Dedup case-insensitively to keep the expression compact.
        seen: set[str] = set()
        uniq: list[str] = []
        for t in terms:
            low = t.lower()
            if low in seen:
                continue
            seen.add(low)
            uniq.append(t)
        if len(uniq) == 1:
            out.append(_fts_quote(uniq[0]))
        else:
            inner = " OR ".join(_fts_quote(t) for t in uniq)
            out.append(f"({inner})")
    return " AND ".join(out) if out else '""'


def bm25_search(
    conn: sqlite3.Connection,
    *,
    query_text: str,
    chunk_kind: str,
    language: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
    node_kind: str | None = None,
    target_id: int | None = None,
    k: int = 5,
    expander: Any | None = None,
) -> list[SearchHit]:
    """Keyword search over chunk text via FTS5 BM25, filtered to the same
    candidate set as vector_search. Returned `SearchHit.distance` carries
    the raw bm25() score (already negative — smaller is better, matching
    vector L2 distance semantics).

    When `expander` is provided, each token in `query_text` is replaced
    by an OR-group of alternates before the MATCH expression is built.
    The expander runs on BM25 ONLY — passing it through to vector search
    is intentionally not supported, since dense embeddings already
    capture synonymy and expansion only adds noise there.
    """
    where = ["c.kind = ?"]
    cand_params: list[Any] = [chunk_kind]
    if language is not None:
        where.append("c.language = ?")
        cand_params.append(language)
    if node_kind is not None:
        where.append("c.node_kind = ?")
        cand_params.append(node_kind)
    needs_artifact_join = repo is not None or author_login is not None or target_id is not None
    if target_id is not None:
        where.append("a.target_id = ?")
        cand_params.append(target_id)
    if repo is not None:
        where.append("a.repo = ?")
        cand_params.append(repo)
    if author_login is not None:
        where.append("a.author_login = ?")
        cand_params.append(author_login)

    join_artifact = " JOIN artifact a ON a.id = c.artifact_id" if needs_artifact_join else ""
    candidates_sql = f"SELECT c.id FROM chunk c{join_artifact} WHERE " + " AND ".join(where)

    if expander is not None:
        match_expr = _fts_match_from_groups(expander.expand(query_text))
    else:
        match_expr = _fts_escape(query_text)

    fetch_k = k * 3 if target_id is None else k

    rows = conn.execute(
        f"""
        SELECT
          f.rowid AS chunk_id, bm25(chunk_fts) AS score,
          c.artifact_id, c.text, c.context_json, c.language AS chunk_language,
          a.kind AS artifact_kind,
          a.repo AS artifact_repo, a.source_url AS artifact_source_url,
          a.decision AS artifact_decision,
          a.target_id AS target_id, t.name AS target_name, t.kind AS target_kind
        FROM chunk_fts f
        JOIN chunk c ON c.id = f.rowid
        JOIN artifact a ON a.id = c.artifact_id
        JOIN target t ON t.id = a.target_id
        WHERE f.rowid IN ({candidates_sql})
          AND chunk_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (*cand_params, match_expr, fetch_k),
    ).fetchall()

    hits = [
        SearchHit(
            chunk_id=r["chunk_id"],
            artifact_id=r["artifact_id"],
            distance=r["score"],
            text=r["text"],
            context=json.loads(r["context_json"]) if r["context_json"] else {},
            artifact_kind=r["artifact_kind"],
            artifact_language=r["chunk_language"],
            artifact_repo=r["artifact_repo"],
            artifact_source_url=r["artifact_source_url"],
            artifact_decision=r["artifact_decision"],
            target_id=r["target_id"],
            target_name=r["target_name"],
            target_kind=r["target_kind"],
        )
        for r in rows
    ]
    if target_id is None:
        hits = _dedup_hits(hits, conn)[:k]
    return hits


# ---------- Per-repo / per-author rollups (wiki vault) ----------


def repo_overview(
    conn: sqlite3.Connection,
    *,
    target_id: int,
    full_name: str,
    top_n: int = 10,
) -> dict[str, Any]:
    """Aggregate per-repo stats for the wiki vault's repo overview page.

    Returns counts by kind, top file paths (chunk count desc), top
    contributors (distinct author_login counts), plus pass-through
    metadata (default_branch, head_sha, pushed_at) from the `repo` row.

    Top-paths uses `json_extract(chunk.context_json, '$.path')` since
    the file path lives only in the per-chunk context dict in this
    schema — there's no top-level `artifact.path` column.
    """
    counts: dict[str, int] = {
        r["kind"]: r["n"]
        for r in conn.execute(
            "SELECT kind, COUNT(*) AS n FROM artifact "
            "WHERE target_id = ? AND repo = ? GROUP BY kind",
            (target_id, full_name),
        ).fetchall()
    }
    top_paths = [
        (r["path"], r["n"])
        for r in conn.execute(
            "SELECT json_extract(c.context_json, '$.path') AS path, COUNT(*) AS n "
            "FROM chunk c JOIN artifact a ON a.id = c.artifact_id "
            "WHERE a.target_id = ? AND a.repo = ? "
            "AND json_extract(c.context_json, '$.path') IS NOT NULL "
            "GROUP BY path ORDER BY n DESC LIMIT ?",
            (target_id, full_name, top_n),
        ).fetchall()
    ]
    top_authors = [
        (r["author_login"], r["n"])
        for r in conn.execute(
            "SELECT author_login, COUNT(*) AS n FROM artifact "
            "WHERE target_id = ? AND repo = ? AND author_login IS NOT NULL "
            "GROUP BY author_login ORDER BY n DESC LIMIT ?",
            (target_id, full_name, top_n),
        ).fetchall()
    ]
    repo_row = conn.execute(
        "SELECT default_branch, head_sha, pushed_at, archived, fork "
        "FROM repo WHERE target_id = ? AND full_name = ?",
        (target_id, full_name),
    ).fetchone()
    meta = dict(repo_row) if repo_row else {}
    return {
        "full_name": full_name,
        "counts": counts,
        "top_paths": top_paths,
        "top_authors": top_authors,
        "default_branch": meta.get("default_branch"),
        "head_sha": meta.get("head_sha"),
        "pushed_at": meta.get("pushed_at"),
        "archived": bool(meta.get("archived", 0)),
        "fork": bool(meta.get("fork", 0)),
    }


def list_authors_for_target(
    conn: sqlite3.Connection,
    *,
    target_id: int,
    min_review_comments: int = 1,
) -> list[tuple[str, int]]:
    """Return `[(author_login, review_comment_count), ...]` ordered by
    count desc, for authors meeting `min_review_comments`.

    Source of truth is `review_comment` artifacts — this is what
    `developer_profile` samples from. User-mode targets have
    `author_login IS NULL` on every artifact by design (the corpus is
    one person) and so return an empty list here; the wiki exporter
    handles that case by falling back to the target's own name.
    """
    rows = conn.execute(
        "SELECT author_login, COUNT(*) AS n FROM artifact "
        "WHERE target_id = ? AND kind = 'review_comment' "
        "AND author_login IS NOT NULL "
        "GROUP BY author_login HAVING n >= ? ORDER BY n DESC, author_login",
        (target_id, min_review_comments),
    ).fetchall()
    return [(r["author_login"], r["n"]) for r in rows]


def upsert_note_artifact(
    conn: sqlite3.Connection,
    *,
    target_id: int,
    external_id: str,
    source_url: str,
    meta: dict[str, Any] | None,
) -> int:
    """Convenience writer for scratch-note artifacts. Same call shape as
    `upsert_artifact` but pre-fills the discriminator fields that don't
    apply to notes (kind, repo, language, decision, author_email)."""
    return upsert_artifact(
        conn,
        target_id=target_id,
        kind="note",
        external_id=external_id,
        source_url=source_url,
        repo=None,
        language=None,
        author_email=None,
        author_login=None,
        created_at=_now_iso(),
        decision=None,
        meta=meta,
    )


def list_note_artifacts(conn: sqlite3.Connection, *, target_id: int) -> list[dict[str, Any]]:
    """All kind='note' artifacts for `target_id`. Used by ingest_notes
    to find ones whose source file has been deleted from disk."""
    rows = conn.execute(
        "SELECT id, external_id, source_url, meta_json FROM artifact "
        "WHERE target_id = ? AND kind = 'note'",
        (target_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "external_id": r["external_id"],
                "source_url": r["source_url"],
                "meta": json.loads(r["meta_json"]) if r["meta_json"] else {},
            }
        )
    return out


def delete_artifact(conn: sqlite3.Connection, artifact_id: int) -> None:
    """Remove an artifact + cascade its chunks/vectors.

    Chunks cascade via FK ON DELETE CASCADE; vec_chunk rows are NOT FK'd
    (sqlite-vec virtual tables can't be foreign-key referenced) so we
    wipe them first by chunk_id, identical to the pattern in
    `delete_chunks_for_artifact`. Order: vec_chunk → chunk (triggers
    chunk_ad which removes chunk_fts rows) → artifact.
    """
    chunk_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM chunk WHERE artifact_id = ?", (artifact_id,)
        ).fetchall()
    ]
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(f"DELETE FROM vec_chunk WHERE chunk_id IN ({placeholders})", chunk_ids)
    conn.execute("DELETE FROM chunk WHERE artifact_id = ?", (artifact_id,))
    conn.execute("DELETE FROM artifact WHERE id = ?", (artifact_id,))


# ---------- Recency lookup ----------

# SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999; chunk 500 to stay well clear
# even after we add a few extra slots elsewhere in the future.
_RECENCY_BATCH = 500


def get_chunk_recency(
    conn: sqlite3.Connection, chunk_ids: Iterable[int]
) -> dict[int, tuple[str | None, str]]:
    """Return `{chunk_id: (artifact.created_at, artifact.kind)}`.

    Used by `hybrid_search` to apply recency decay across the fused
    candidate set in one batched SQL call instead of per-hit joins.
    Chunks whose artifact has no `created_at` come back as `(None, kind)`
    and the caller decides how to handle them (current policy: don't
    penalize undatable chunks).
    """
    ids = [int(cid) for cid in chunk_ids]
    if not ids:
        return {}
    out: dict[int, tuple[str | None, str]] = {}
    for i in range(0, len(ids), _RECENCY_BATCH):
        batch = ids[i : i + _RECENCY_BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT c.id AS chunk_id, a.created_at, a.kind "
            f"FROM chunk c JOIN artifact a ON a.id = c.artifact_id "
            f"WHERE c.id IN ({placeholders})",
            batch,
        ).fetchall()
        for r in rows:
            out[r["chunk_id"]] = (r["created_at"], r["kind"])
    return out


# ---------- Developer profile (cache + sampling) ----------


def recent_review_comments(
    conn: sqlite3.Connection,
    *,
    author_login: str | None = None,
    repo: str | None = None,
    language: str | None = None,
    target_id: int | None = None,
    limit: int = 50,
) -> list[ChunkRow]:
    """Return the N most-recent review-comment chunks, ordered by their
    parent artifact's `created_at` DESC.

    `target_id=None` coalesces across all targets and dedupes on
    `(artifact.kind, artifact.external_id)` so the same comment ingested
    twice surfaces once.

    `author_login=None` means "no filter" — appropriate in user mode
    where the whole corpus is one person's reviews; in org mode the
    caller almost always wants a specific login.
    """
    where = ["c.kind = 'review_comment'"]
    params: list[Any] = []
    if target_id is not None:
        where.append("a.target_id = ?")
        params.append(target_id)
    if author_login is not None:
        where.append("a.author_login = ?")
        params.append(author_login)
    if repo is not None:
        where.append("a.repo = ?")
        params.append(repo)
    if language is not None:
        where.append("c.language = ?")
        params.append(language)
    params.append(limit)
    if target_id is None:
        # Coalesce: one row per (kind, external_id) across targets.
        inner = (
            "SELECT c.id, c.artifact_id, c.kind, c.text, c.context_json, "
            "c.summary, c.embed_model, a.created_at, "
            "ROW_NUMBER() OVER (PARTITION BY a.kind, a.external_id "
            "ORDER BY a.target_id, a.id) AS rn "
            "FROM chunk c JOIN artifact a ON a.id = c.artifact_id "
            f"WHERE {' AND '.join(where)}"
        )
        sql = (
            f"SELECT id, artifact_id, kind, text, context_json, summary, "
            f"embed_model FROM ({inner}) "
            "WHERE rn = 1 "
            "ORDER BY COALESCE(created_at, '') DESC LIMIT ?"
        )
    else:
        sql = (
            "SELECT c.id, c.artifact_id, c.kind, c.text, c.context_json, "
            "c.summary, c.embed_model "
            "FROM chunk c JOIN artifact a ON a.id = c.artifact_id "
            "WHERE " + " AND ".join(where) + " "
            "ORDER BY COALESCE(a.created_at, '') DESC "
            "LIMIT ?"
        )
    rows = conn.execute(sql, params).fetchall()
    return [
        ChunkRow(
            id=r["id"],
            artifact_id=r["artifact_id"],
            kind=r["kind"],
            text=r["text"],
            context=json.loads(r["context_json"]) if r["context_json"] else {},
            summary=r["summary"],
            embed_model=r["embed_model"],
        )
        for r in rows
    ]


def get_cached_profile(conn: sqlite3.Connection, login: str) -> dict[str, Any] | None:
    """Fetch the cached developer profile for `login`, or None if absent.

    `login` is a composite string built by `_profile_cache_key` (it folds
    in author / language / repo / target). The hash comparison decides
    whether to re-synthesize.
    """
    row = conn.execute(
        "SELECT login, profile_md, sample_hash, n_samples, generated_at "
        "FROM developer_profile_cache WHERE login = ?",
        (login,),
    ).fetchone()
    if row is None:
        return None
    return {
        "login": row["login"],
        "profile_md": row["profile_md"],
        "sample_hash": row["sample_hash"],
        "n_samples": row["n_samples"],
        "generated_at": row["generated_at"],
    }


def set_cached_profile(
    conn: sqlite3.Connection,
    *,
    login: str,
    profile_md: str,
    sample_hash: str,
    n_samples: int,
) -> None:
    """Upsert the cached developer profile for the composite cache key."""
    conn.execute(
        "INSERT INTO developer_profile_cache "
        "(login, profile_md, sample_hash, n_samples, generated_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(login) DO UPDATE SET "
        "profile_md = excluded.profile_md, "
        "sample_hash = excluded.sample_hash, "
        "n_samples = excluded.n_samples, "
        "generated_at = excluded.generated_at",
        (login, profile_md, sample_hash, n_samples, _now_iso()),
    )


# ---------- Stats ----------


def stats(conn: sqlite3.Connection, *, target_id: int | None = None) -> dict[str, Any]:
    """Aggregate counts. `target_id=None` reports the whole DB."""
    base_where = ""
    params: tuple[Any, ...] = ()
    if target_id is not None:
        base_where = " WHERE target_id = ?"
        params = (target_id,)
    artifact_counts = {
        r["kind"]: r["n"]
        for r in conn.execute(
            f"SELECT kind, COUNT(*) AS n FROM artifact{base_where} GROUP BY kind",
            params,
        ).fetchall()
    }
    if target_id is None:
        chunk_counts = {
            r["kind"]: r["n"]
            for r in conn.execute("SELECT kind, COUNT(*) AS n FROM chunk GROUP BY kind").fetchall()
        }
        lang_counts = {
            r["language"]: r["n"]
            for r in conn.execute(
                "SELECT language, COUNT(*) AS n FROM chunk "
                "WHERE language IS NOT NULL GROUP BY language ORDER BY n DESC"
            ).fetchall()
        }
        pending_embed = conn.execute(
            "SELECT COUNT(*) AS n FROM chunk WHERE embed_model IS NULL"
        ).fetchone()["n"]
        vec_rows = conn.execute("SELECT COUNT(*) AS n FROM vec_chunk").fetchone()["n"]
    else:
        chunk_counts = {
            r["kind"]: r["n"]
            for r in conn.execute(
                "SELECT c.kind AS kind, COUNT(*) AS n FROM chunk c "
                "JOIN artifact a ON a.id = c.artifact_id "
                "WHERE a.target_id = ? GROUP BY c.kind",
                params,
            ).fetchall()
        }
        lang_counts = {
            r["language"]: r["n"]
            for r in conn.execute(
                "SELECT c.language AS language, COUNT(*) AS n FROM chunk c "
                "JOIN artifact a ON a.id = c.artifact_id "
                "WHERE a.target_id = ? AND c.language IS NOT NULL "
                "GROUP BY c.language ORDER BY n DESC",
                params,
            ).fetchall()
        }
        pending_embed = conn.execute(
            "SELECT COUNT(*) AS n FROM chunk c JOIN artifact a ON a.id = c.artifact_id "
            "WHERE a.target_id = ? AND c.embed_model IS NULL",
            params,
        ).fetchone()["n"]
        vec_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM vec_chunk v "
            "JOIN chunk c ON c.id = v.chunk_id "
            "JOIN artifact a ON a.id = c.artifact_id "
            "WHERE a.target_id = ?",
            params,
        ).fetchone()["n"]
    return {
        "artifacts": artifact_counts,
        "chunks": chunk_counts,
        "languages": lang_counts,
        "pending_embed": pending_embed,
        "vectors": vec_rows,
    }
