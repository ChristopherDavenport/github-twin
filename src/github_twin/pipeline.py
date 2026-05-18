"""Pipeline operations shared by the CLI and the MCP server.

Both `gt ingest`/`gt embed` and the MCP `sync` tool call into these.
Output goes through a `Reporter` so callers control where progress lines land
(stdout vs. server logs vs. silent).

Multi-target: `run_ingest` iterates every target in the DB (or a single
specified target) and dispatches per-target. `run_embed` and `run_summarize`
are corpus-wide — they work over any chunk regardless of target.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from github_twin.config import Config, resolved_clones_dir
from github_twin.embed import Embedder, make_embedder
from github_twin.embed.prefix import prefix_chunk
from github_twin.eval.llm import TextLLM, make_text_llm
from github_twin.ingest.cache import RawCache
from github_twin.ingest.commits import ingest_commits, ingest_commits_org
from github_twin.ingest.files import ingest_files
from github_twin.ingest.github_client import GitHubClient
from github_twin.ingest.reviews import ingest_reviews, ingest_reviews_org
from github_twin.process.summarize import summarize_chunks
from github_twin.store import queries as q
from github_twin.store.db import transaction
from github_twin.target import Target, load_targets


class IdentityMissingError(RuntimeError):
    """Raised if `gt init` hasn't been run yet."""


# Bumped whenever the embed-time chunk prefix (`embed.prefix`) changes in a
# way that would shift vectors. The pipeline reads/writes this to the
# `sync_cursor` table; on mismatch `run_embed` wipes vec_chunk and re-embeds.
#
# History:
#   1 — raw chunk.text (pre-contextual-retrieval baseline).
#   2 — deterministic per-kind headers (path / symbol / leading_doc / etc.).
#   3 — adds `chunk.summary` (LLM-generated NL description) to the code/file
#       header lines. Bridges NL queries to identifier-only code chunks.
#   4 — adds the `note` chunk kind (scratch-note round-trip from the wiki
#       vault) with a `# note: {title}` header. Pre-existing chunk kinds
#       still emit identical prefixes; the version bump is so corpora
#       built before this change re-embed once and the cursor advances
#       atomically across the new + old kinds at the same version.
EMBED_TEXT_VERSION = 4
_EMBED_VERSION_KEY = "embed_text_version"


Reporter = Callable[[str], None]


def _noop(_: str) -> None:
    return None


def _resolve_targets(conn: sqlite3.Connection, target_filter: int | str | None) -> list[Target]:
    """Return the targets `run_ingest` should walk. `None` = all of them."""
    if target_filter is None:
        return load_targets(conn)
    targets = load_targets(conn)
    if isinstance(target_filter, int):
        out = [t for t in targets if t.id == target_filter]
    else:
        out = [t for t in targets if t.name == target_filter]
    if not out:
        raise IdentityMissingError(
            f"No target matches {target_filter!r}. Existing: {[(t.kind, t.name) for t in targets]}"
        )
    return out


def run_ingest(
    cfg: Config,
    conn: sqlite3.Connection,
    *,
    since: str | None = None,
    commits_only: bool = False,
    reviews_only: bool = False,
    limit: int | None = None,
    target: int | str | None = None,
    report: Reporter = _noop,
) -> dict[str, object]:
    targets = _resolve_targets(conn, target)
    if not targets:
        raise IdentityMissingError("No targets. Run `gt init` first.")
    # Resolve clones_dir against data_dir once, here, so every ingest
    # function downstream sees a concrete Path instead of None.
    cfg = cfg.model_copy(
        update={"ingest": cfg.ingest.model_copy(update={"clones_dir": resolved_clones_dir(cfg)})}
    )
    aggregate: dict[str, object] = {}
    for t in targets:
        assert t.id is not None
        if len(targets) > 1:
            report(f"=== target: {t.kind} {t.name} (id {t.id}) ===")
        summary = _ingest_one(
            cfg=cfg,
            conn=conn,
            target=t,
            since=since,
            commits_only=commits_only,
            reviews_only=reviews_only,
            limit=limit,
            report=report,
        )
        aggregate[f"{t.kind}:{t.name}"] = summary
    return aggregate


def _ingest_one(
    *,
    cfg: Config,
    conn: sqlite3.Connection,
    target: Target,
    since: str | None,
    commits_only: bool,
    reviews_only: bool,
    limit: int | None,
    report: Reporter,
) -> dict[str, object]:
    assert target.id is not None
    if target.is_org or target.is_repo:
        # Repo mode shares the org-mode pipeline because the repo table just
        # has one row instead of many; the per-repo iteration inside
        # ingest_files/commits_org/reviews_org handles either width.
        summary: dict[str, object] = {}
        cache = RawCache(cfg.paths.raw_dir)
        if not (commits_only or reviews_only):
            s_files = ingest_files(conn=conn, cfg=cfg.ingest, target_id=target.id, limit=limit)
            summary["files"] = s_files
            report(f"files: {s_files}")
        with GitHubClient() as gh:
            if not reviews_only:
                with transaction(conn):
                    s_commits = ingest_commits_org(
                        conn=conn,
                        gh=gh,
                        cache=cache,
                        cfg=cfg.ingest,
                        target_id=target.id,
                        limit_per_repo=limit,
                    )
                summary["commits"] = s_commits
                report(f"commits: {s_commits}")
            if not commits_only:
                with transaction(conn):
                    s_reviews = ingest_reviews_org(
                        conn=conn,
                        gh=gh,
                        cache=cache,
                        cfg=cfg.ingest,
                        target_id=target.id,
                        limit_prs_per_repo=limit,
                    )
                summary["reviews"] = s_reviews
                report(f"reviews: {s_reviews}")
        return summary

    cache = RawCache(cfg.paths.raw_dir)
    summary = {}
    with GitHubClient() as gh:
        if not reviews_only:
            with transaction(conn):
                s_commits = ingest_commits(
                    conn=conn,
                    gh=gh,
                    cache=cache,
                    username=target.name,
                    emails=target.emails,
                    cfg=cfg.ingest,
                    target_id=target.id,
                    since=since,
                    limit=limit,
                )
            summary["commits"] = s_commits
            report(f"commits: {s_commits}")
        if not commits_only:
            with transaction(conn):
                s_reviews = ingest_reviews(
                    conn=conn,
                    gh=gh,
                    cache=cache,
                    username=target.name,
                    cfg=cfg.ingest,
                    target_id=target.id,
                    since=since,
                    limit_prs=limit,
                )
            summary["reviews"] = s_reviews
            report(f"reviews: {s_reviews}")
    return summary


def run_summarize(
    cfg: Config,
    conn: sqlite3.Connection,
    *,
    kinds: tuple[str, ...] | None = None,
    limit: int | None = None,
    rebuild: bool = False,
    report: Reporter = _noop,
    llm: TextLLM | None = None,
) -> int:
    """Generate LLM summaries for chunks missing one. Returns count.

    Picks the LLM via `cfg.summarize.backend` (default `auto` → Claude
    if a key is set, else Gemini, else Ollama). For fully-local runs
    set `cfg.summarize.backend = "ollama"` (or `GT_SUMMARIZE__BACKEND=ollama`).
    """
    if llm is None:
        llm = make_text_llm(
            claude_model=cfg.summarize.claude_model,
            gemini_model=cfg.summarize.gemini_model,
            ollama_model=cfg.summarize.ollama_model,
            prefer=cfg.summarize.backend,
        )
    return summarize_chunks(
        conn,
        llm,
        kinds=tuple(kinds) if kinds is not None else tuple(cfg.summarize.kinds),
        limit=limit,
        max_tokens=cfg.summarize.max_tokens,
        report=report,
        rebuild=rebuild,
    )


def run_embed(
    cfg: Config,
    conn: sqlite3.Connection,
    *,
    rebuild: bool = False,
    batch_size: int | None = None,
    report: Reporter = _noop,
    embedder: Embedder | None = None,
) -> int:
    embedder = embedder or make_embedder(cfg.embed)
    if batch_size is None:
        batch_size = cfg.embed.batch_size
    if rebuild:
        with transaction(conn):
            conn.execute("UPDATE chunk SET embed_model = NULL")
            conn.execute("DELETE FROM vec_chunk")
        report("rebuild: cleared all vectors")
    elif _embed_text_version_needs_bump(conn):
        with transaction(conn):
            conn.execute("UPDATE chunk SET embed_model = NULL")
            conn.execute("DELETE FROM vec_chunk")
        report(
            f"re-embedding all chunks: embed text version "
            f"{_stored_embed_version(conn)!r} -> {EMBED_TEXT_VERSION} "
            f"(contextual retrieval upgrade)"
        )

    total = conn.execute("SELECT COUNT(*) AS n FROM chunk WHERE embed_model IS NULL").fetchone()[
        "n"
    ]
    report(f"embedding {total} chunks with {embedder.model_id} (batch={batch_size})")

    done = 0
    buffer: list[q.ChunkRow] = []
    for chunk in q.pending_embed_chunks(conn, batch_size=batch_size):
        buffer.append(chunk)
        if len(buffer) >= batch_size:
            _flush(conn, embedder, buffer)
            done += len(buffer)
            report(f"  ... {done}/{total}")
            buffer = []
    if buffer:
        _flush(conn, embedder, buffer)
        done += len(buffer)
    if done > 0 or total == 0:
        # Mark the corpus as having been embedded at the current version,
        # whether we re-embedded the whole thing or there was nothing to do.
        q.set_cursor(conn, _EMBED_VERSION_KEY, str(EMBED_TEXT_VERSION))
    report(f"embedded {done} chunks")
    return done


def _stored_embed_version(conn: sqlite3.Connection) -> int:
    raw = q.get_cursor(conn, _EMBED_VERSION_KEY)
    if raw is None:
        return 1
    try:
        return int(raw)
    except ValueError:
        return 1


def _embed_text_version_needs_bump(conn: sqlite3.Connection) -> bool:
    """Only trigger a re-embed when there are already vectors at an older
    version. A brand-new DB with no vectors gets stamped at the current
    version when run_embed completes — no wasted churn."""
    if _stored_embed_version(conn) >= EMBED_TEXT_VERSION:
        return False
    has_vectors = conn.execute("SELECT 1 FROM vec_chunk LIMIT 1").fetchone()
    return has_vectors is not None


def _flush(conn: sqlite3.Connection, embedder: Embedder, batch: list[q.ChunkRow]) -> None:
    vecs = embedder.embed([prefix_chunk(c) for c in batch])
    with transaction(conn):
        for chunk, vec in zip(batch, vecs, strict=True):
            q.write_embedding(conn, chunk_id=chunk.id, embedding=vec, model_id=embedder.model_id)
