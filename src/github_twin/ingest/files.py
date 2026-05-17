"""File-at-HEAD ingest for org-mode (phase O-C).

For each repo in the `repo` table:

1. Cheap skip if `repo.pushed_at <= repo.last_files_at` (nothing pushed since
   last walk) or `repo.size_kb > cfg.ingest.max_repo_size_kb` (e.g. a monorepo).
2. Otherwise shallow-clone (tempdir by default, persistent if
   `cfg.ingest.cache_clones=true`), walk the working tree, chunk every
   recognized-language file via `chunk_file`, upsert one `kind='file'`
   artifact per `(repo, path)`.
3. Stamp `repo.last_files_at` + `repo.head_sha` on success.

Disk footprint: ≤ one repo at a time when `cache_clones=false` (the default).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from github_twin.config import IngestCfg
from github_twin.ingest.clone import CloneError, cloned_repo
from github_twin.process.chunkers import chunk_file
from github_twin.process.language import language_for_path
from github_twin.store import queries as q
from github_twin.store.db import transaction

log = logging.getLogger(__name__)

# Files above this size on disk are skipped wholesale — they're almost
# always generated, vendored, or minified bundles that the language filter
# would let through (e.g. a 12 MB committed .json schema).
MAX_FILE_BYTES = 256 * 1024


@dataclass
class FilesStats:
    repos_visited: int = 0
    repos_skipped: int = 0
    files_chunked: int = 0
    chunks_written: int = 0


def _walk_repo_files(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield (relative_path, absolute_path) for files in the working tree.

    Skips the `.git` directory and obvious symlinks. The language + exclude
    filters happen later in `chunk_file` — this function is only responsible
    for traversal.
    """
    for p in root.rglob("*"):
        if p.is_symlink() or not p.is_file():
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if parts and parts[0] == ".git":
            continue
        yield rel.as_posix(), p


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _source_url(full_name: str, head_sha: str, path: str) -> str:
    return f"https://github.com/{full_name}/blob/{head_sha}/{path}"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _should_skip(repo_row: dict[str, Any], *, max_repo_size_kb: int) -> tuple[bool, str | None]:
    size = repo_row.get("size_kb") or 0
    if size and size > max_repo_size_kb:
        return True, f"size_kb={size} > max_repo_size_kb={max_repo_size_kb}"
    pushed = repo_row.get("pushed_at")
    last = repo_row.get("last_files_at")
    if pushed and last and pushed <= last:
        return True, f"unchanged since last walk (pushed_at={pushed} <= last_files_at={last})"
    return False, None


def ingest_files(
    *,
    conn: sqlite3.Connection,
    cfg: IngestCfg,
    target_id: int,
    limit: int | None = None,
) -> FilesStats:
    stats = FilesStats()
    repos = q.list_repos(conn, target_id=target_id)
    if limit is not None:
        repos = repos[:limit]
    cache_dir: Path | None = Path(cfg.clones_dir) if cfg.cache_clones else None

    for row in repos:
        full_name: str = row["full_name"]
        skip, reason = _should_skip(row, max_repo_size_kb=cfg.max_repo_size_kb)
        if skip:
            log.info("skip %s: %s", full_name, reason)
            stats.repos_skipped += 1
            continue

        try:
            with cloned_repo(full_name, cache_dir=cache_dir) as clone:
                files_seen, chunks_seen = _ingest_one_repo(
                    conn=conn,
                    clone_path=clone.path,
                    full_name=full_name,
                    head_sha=clone.head_sha,
                    cfg=cfg,
                    target_id=target_id,
                )
            with transaction(conn):
                q.set_repo_cursor(
                    conn,
                    target_id=target_id,
                    full_name=full_name,
                    head_sha=clone.head_sha,
                    files_at=_now_iso(),
                )
        except CloneError as e:
            log.warning("clone failed for %s: %s", full_name, e)
            stats.repos_skipped += 1
            continue

        stats.repos_visited += 1
        stats.files_chunked += files_seen
        stats.chunks_written += chunks_seen
        log.info(
            "%s: %d files / %d chunks (head=%s)",
            full_name,
            files_seen,
            chunks_seen,
            clone.head_sha[:8],
        )
    return stats


def _ingest_one_repo(
    *,
    conn: sqlite3.Connection,
    clone_path: Path,
    full_name: str,
    head_sha: str,
    cfg: IngestCfg,
    target_id: int,
) -> tuple[int, int]:
    """Walk a single clone, write `kind='file'` artifacts + chunks. Returns
    (files_chunked, chunks_written). Runs inside its own transaction so a
    crash mid-repo doesn't leave half-ingested state for that repo."""
    files = 0
    chunks = 0
    with transaction(conn):
        for rel, abs_path in _walk_repo_files(clone_path):
            if language_for_path(rel) is None:
                continue
            content = _read_text(abs_path)
            if content is None:
                continue

            url = _source_url(full_name, head_sha, rel)
            lang = language_for_path(rel)

            # Materialize chunks first so we don't create an artifact row for a
            # file that produces zero chunks (small/excluded files).
            file_chunks = list(
                chunk_file(
                    content,
                    repo=full_name,
                    path=rel,
                    source_url=url,
                    head_sha=head_sha,
                    exclude_patterns=cfg.exclude_paths,
                )
            )
            if not file_chunks:
                continue

            artifact_id = q.upsert_artifact(
                conn,
                target_id=target_id,
                kind="file",
                external_id=f"{full_name}:{rel}",
                source_url=url,
                repo=full_name,
                language=lang,
                author_email=None,
                created_at=None,
                decision=None,
                meta={"head_sha": head_sha, "path": rel},
            )
            q.delete_chunks_for_artifact(conn, artifact_id)
            for ck in file_chunks:
                q.insert_chunk(
                    conn,
                    artifact_id=artifact_id,
                    kind="file",
                    text=ck.text,
                    context=ck.context,
                    language=ck.language,
                )
                chunks += 1
            files += 1
    return files, chunks
