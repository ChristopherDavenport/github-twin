"""All SQL access for github-twin. Keep SQL strings here, not scattered across modules."""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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


# ---------- Target (singleton: who or what this DB tracks) ----------


def upsert_target(
    conn: sqlite3.Connection,
    *,
    kind: str,
    name: str,
    external_id: int,
    emails: list[str] | None,
) -> None:
    emails_json = json.dumps(emails) if emails is not None else None
    conn.execute(
        "INSERT INTO target (id, kind, name, external_id, emails_json, discovered_at) "
        "VALUES (1, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, name=excluded.name, "
        "external_id=excluded.external_id, emails_json=excluded.emails_json, "
        "discovered_at=excluded.discovered_at",
        (kind, name, external_id, emails_json, _now_iso()),
    )


def get_target(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT kind, name, external_id, emails_json, discovered_at FROM target WHERE id=1"
    ).fetchone()
    if row is None:
        return None
    return {
        "kind": row["kind"],
        "name": row["name"],
        "external_id": row["external_id"],
        "emails": json.loads(row["emails_json"]) if row["emails_json"] else None,
        "discovered_at": row["discovered_at"],
    }


# ---------- Repos (org-mode: one row per repo we know about) ----------


def upsert_repo(
    conn: sqlite3.Connection,
    *,
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
        "INSERT INTO repo (full_name, default_branch, pushed_at, archived, fork, size_kb) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(full_name) DO UPDATE SET "
        "default_branch=excluded.default_branch, pushed_at=excluded.pushed_at, "
        "archived=excluded.archived, fork=excluded.fork, size_kb=excluded.size_kb",
        (full_name, default_branch, pushed_at, int(archived), int(fork), size_kb),
    )


def get_repo(conn: sqlite3.Connection, full_name: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM repo WHERE full_name=?", (full_name,)).fetchone()
    return dict(row) if row else None


def list_repos(
    conn: sqlite3.Connection,
    *,
    include_archived: bool = False,
    include_forks: bool = False,
) -> list[dict[str, Any]]:
    where: list[str] = []
    if not include_archived:
        where.append("archived = 0")
    if not include_forks:
        where.append("fork = 0")
    sql = "SELECT * FROM repo"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY full_name"
    return [dict(r) for r in conn.execute(sql).fetchall()]


def set_repo_cursor(
    conn: sqlite3.Connection,
    *,
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
    params.append(full_name)
    conn.execute(f"UPDATE repo SET {', '.join(fields)} WHERE full_name=?", params)


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


def get_cursor(conn: sqlite3.Connection, resource: str) -> str | None:
    row = conn.execute("SELECT cursor FROM sync_cursor WHERE resource=?", (resource,)).fetchone()
    return row["cursor"] if row else None


def set_cursor(conn: sqlite3.Connection, resource: str, cursor: str) -> None:
    conn.execute(
        "INSERT INTO sync_cursor (resource, cursor, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(resource) DO UPDATE SET cursor=excluded.cursor, "
        "updated_at=excluded.updated_at",
        (resource, cursor, _now_iso()),
    )


# ---------- Artifacts ----------


def upsert_artifact(
    conn: sqlite3.Connection,
    *,
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
) -> int:
    """Insert or update an artifact keyed by (kind, external_id). Returns artifact id."""
    cur = conn.execute(
        "INSERT INTO artifact (kind, external_id, source_url, repo, language, "
        "author_email, author_login, created_at, decision, meta_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(kind, external_id) DO UPDATE SET "
        "source_url=excluded.source_url, repo=excluded.repo, language=excluded.language, "
        "author_email=excluded.author_email, author_login=excluded.author_login, "
        "created_at=excluded.created_at, decision=excluded.decision, "
        "meta_json=excluded.meta_json "
        "RETURNING id",
        (
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
        ),
    )
    row_id: int = cur.fetchone()["id"]
    return row_id


def list_rules(
    conn: sqlite3.Connection,
    *,
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

    `repo` filters on `artifact.repo` — the dominant repo stamped at
    distill time. Cross-repo rules whose dominant repo doesn't match
    won't surface even if `repo` appears in `meta.member_repos`; if
    that becomes a problem we can promote the meta lookup later.
    """
    where = ["a.kind = 'rule'", "c.kind = ?"]
    params: list[Any] = [chunk_kind]
    if language is not None:
        where.append("a.language = ?")
        params.append(language)
    if repo is not None:
        where.append("a.repo = ?")
        params.append(repo)
    if author_login is not None:
        where.append("a.author_login = ?")
        params.append(author_login)
    params.append(limit)
    sql = (
        "SELECT a.id, a.language, a.meta_json, c.text "
        "FROM artifact a JOIN chunk c ON c.artifact_id = a.id "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY a.id LIMIT ?"
    )
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


def artifact_exists(conn: sqlite3.Connection, kind: str, external_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM artifact WHERE kind=? AND external_id=? LIMIT 1",
        (kind, external_id),
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


def vector_search(
    conn: sqlite3.Connection,
    *,
    query_vec: list[float],
    chunk_kind: str,
    language: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
    node_kind: str | None = None,
    k: int = 5,
) -> list[SearchHit]:
    """KNN search restricted to chunks of a given kind (and optionally
    language / repo / author_login / node_kind).

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
    # repo and author_login both live on artifact, so join once if either supplied.
    needs_artifact_join = repo is not None or author_login is not None
    if repo is not None:
        where.append("a.repo = ?")
        cand_params.append(repo)
    if author_login is not None:
        where.append("a.author_login = ?")
        cand_params.append(author_login)

    join_artifact = " JOIN artifact a ON a.id = c.artifact_id" if needs_artifact_join else ""
    candidates_sql = f"SELECT c.id FROM chunk c{join_artifact} WHERE " + " AND ".join(where)

    rows = conn.execute(
        f"""
        SELECT
          v.chunk_id, v.distance,
          c.artifact_id, c.text, c.context_json, c.language AS chunk_language,
          a.kind AS artifact_kind,
          a.repo AS artifact_repo, a.source_url AS artifact_source_url,
          a.decision AS artifact_decision
        FROM vec_chunk v
        JOIN chunk c ON c.id = v.chunk_id
        JOIN artifact a ON a.id = c.artifact_id
        WHERE v.chunk_id IN ({candidates_sql})
          AND v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
        """,
        (*cand_params, _pack_vec(query_vec), k),
    ).fetchall()

    return [
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
        )
        for r in rows
    ]


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
    needs_artifact_join = repo is not None or author_login is not None
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

    rows = conn.execute(
        f"""
        SELECT
          f.rowid AS chunk_id, bm25(chunk_fts) AS score,
          c.artifact_id, c.text, c.context_json, c.language AS chunk_language,
          a.kind AS artifact_kind,
          a.repo AS artifact_repo, a.source_url AS artifact_source_url,
          a.decision AS artifact_decision
        FROM chunk_fts f
        JOIN chunk c ON c.id = f.rowid
        JOIN artifact a ON a.id = c.artifact_id
        WHERE f.rowid IN ({candidates_sql})
          AND chunk_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (*cand_params, match_expr, k),
    ).fetchall()

    return [
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
        )
        for r in rows
    ]


# ---------- Developer profile (cache + sampling) ----------


def recent_review_comments(
    conn: sqlite3.Connection,
    *,
    author_login: str | None = None,
    repo: str | None = None,
    language: str | None = None,
    limit: int = 50,
) -> list[ChunkRow]:
    """Return the N most-recent review-comment chunks, ordered by their
    parent artifact's `created_at` DESC.

    `author_login=None` means "no filter" — appropriate in user mode
    where the whole corpus is one person's reviews; in org mode the
    caller almost always wants a specific login.

    `repo` filters on `artifact.repo`; `language` filters on
    `chunk.language` (populated from the review's diff path), so it
    picks out comments anchored to code in that language.
    """
    where = ["c.kind = 'review_comment'"]
    params: list[Any] = []
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

    Caller compares the returned `sample_hash` against the current set
    of recent comments and decides whether to re-synthesize.
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
    """Upsert the cached developer profile for `login`."""
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


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    artifact_counts = {
        r["kind"]: r["n"]
        for r in conn.execute("SELECT kind, COUNT(*) AS n FROM artifact GROUP BY kind").fetchall()
    }
    chunk_counts = {
        r["kind"]: r["n"]
        for r in conn.execute("SELECT kind, COUNT(*) AS n FROM chunk GROUP BY kind").fetchall()
    }
    # Per-chunk language is what retrieval filters on; report that here.
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
    return {
        "artifacts": artifact_counts,
        "chunks": chunk_counts,
        "languages": lang_counts,
        "pending_embed": pending_embed,
        "vectors": vec_rows,
    }
