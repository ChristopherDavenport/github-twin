"""Cluster review-comment chunks by their embeddings.

Uses HDBSCAN (density-based, no need to pick `k`). Chunks that don't fit any
cluster get label -1 and are dropped — they were either one-offs or genuinely
unique observations, not patterns.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from dataclasses import dataclass
from typing import Any

import hdbscan
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ClusterMember:
    chunk_id: int
    text: str
    context: dict[str, Any]
    language: str | None


@dataclass
class Cluster:
    cluster_id: int
    members: list[ClusterMember]

    @property
    def size(self) -> int:
        return len(self.members)


def _load_review_chunks_with_vectors(
    conn: sqlite3.Connection,
    *,
    author_login: str | None = None,
    repo: str | None = None,
) -> tuple[list[ClusterMember], np.ndarray]:
    import json

    where = ["c.kind = 'review_comment'"]
    params: list[Any] = []
    join_artifact = author_login is not None or repo is not None
    if author_login is not None:
        where.append("a.author_login = ?")
        params.append(author_login)
    if repo is not None:
        where.append("a.repo = ?")
        params.append(repo)

    artifact_join = "JOIN artifact a ON a.id = c.artifact_id" if join_artifact else ""
    sql = (
        "SELECT c.id, c.text, c.context_json, c.language, v.embedding "
        "FROM chunk c JOIN vec_chunk v ON v.chunk_id = c.id "
        f"{artifact_join} "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY c.id"
    )
    rows = conn.execute(sql, params).fetchall()
    members: list[ClusterMember] = []
    vecs: list[np.ndarray] = []
    for r in rows:
        ctx = json.loads(r["context_json"]) if r["context_json"] else {}
        members.append(
            ClusterMember(
                chunk_id=r["id"],
                text=r["text"],
                context=ctx,
                language=r["language"],
            )
        )
        raw = r["embedding"]
        n = len(raw) // 4
        vecs.append(np.array(struct.unpack(f"{n}f", raw), dtype=np.float32))
    if not vecs:
        return members, np.zeros((0, 0), dtype=np.float32)
    return members, np.stack(vecs)


def _load_code_chunks_with_vectors(
    conn: sqlite3.Connection,
    *,
    author_login: str | None = None,
    language: str | None = None,
    repo: str | None = None,
) -> tuple[list[ClusterMember], np.ndarray]:
    """Load `kind='code'` chunks (diff-added blocks) with their embeddings.

    Code chunks already carry per-chunk language (set at chunk_diff time);
    filter at the chunk level rather than at the artifact level so a
    multi-language commit yields per-language clusters cleanly. The `repo`
    filter is artifact-level (commits carry the repo).
    """
    import json

    where = ["c.kind = 'code'"]
    params: list[Any] = []
    join_artifact = author_login is not None or repo is not None
    if author_login is not None:
        where.append("a.author_login = ?")
        params.append(author_login)
    if repo is not None:
        where.append("a.repo = ?")
        params.append(repo)
    if language is not None:
        where.append("c.language = ?")
        params.append(language)

    artifact_join = "JOIN artifact a ON a.id = c.artifact_id" if join_artifact else ""
    sql = (
        "SELECT c.id, c.text, c.context_json, c.language, v.embedding "
        "FROM chunk c JOIN vec_chunk v ON v.chunk_id = c.id "
        f"{artifact_join} "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY c.id"
    )
    rows = conn.execute(sql, params).fetchall()

    members: list[ClusterMember] = []
    vecs: list[np.ndarray] = []
    for r in rows:
        ctx = json.loads(r["context_json"]) if r["context_json"] else {}
        members.append(
            ClusterMember(
                chunk_id=r["id"],
                text=r["text"],
                context=ctx,
                language=r["language"],
            )
        )
        raw = r["embedding"]
        n = len(raw) // 4
        vecs.append(np.array(struct.unpack(f"{n}f", raw), dtype=np.float32))
    if not vecs:
        return members, np.zeros((0, 0), dtype=np.float32)
    return members, np.stack(vecs)


def _cluster_from_vectors(
    members: list[ClusterMember],
    vecs: np.ndarray,
    *,
    min_cluster_size: int,
    max_cluster_size: int | None,
    kind_label: str,
) -> list[Cluster]:
    """Shared HDBSCAN-with-cosine-distance core. Pulled out of
    `cluster_review_comments` so `cluster_code_chunks` can reuse it byte-for-byte."""
    if len(members) < min_cluster_size:
        log.info("not enough %s chunks (%d) to cluster", kind_label, len(members))
        return []

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / np.clip(norms, 1e-12, None)
    sim = normed @ normed.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = (1.0 - sim).astype(np.float64)
    np.fill_diagonal(dist, 0.0)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="precomputed",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(dist)

    by_label: dict[int, list[ClusterMember]] = {}
    for member, label in zip(members, labels, strict=True):
        if label < 0:
            continue
        by_label.setdefault(int(label), []).append(member)

    clusters: list[Cluster] = []
    for label, ms in sorted(by_label.items()):
        if max_cluster_size and len(ms) > max_cluster_size:
            log.info("dropping oversized cluster %d (size=%d)", label, len(ms))
            continue
        clusters.append(Cluster(cluster_id=label, members=ms))
    log.info(
        "clustering %s: %d members -> %d clusters (noise: %d)",
        kind_label,
        len(members),
        len(clusters),
        sum(1 for l in labels if l < 0),  # noqa: E741
    )
    return clusters


def cluster_review_comments(
    conn: sqlite3.Connection,
    *,
    min_cluster_size: int = 3,
    max_cluster_size: int | None = None,
    author_login: str | None = None,
    repo: str | None = None,
) -> list[Cluster]:
    """Return clusters of review-comment chunks, smallest cluster_id first.

    Drops the noise cluster (label -1) and any cluster outside [min, max].
    `author_login` scopes to one reviewer's history (useful in org-mode
    where a global cluster would mix many reviewers' styles). `repo`
    scopes to one repo, letting a shared org DB produce repo-specific
    rule sets.
    """
    members, vecs = _load_review_chunks_with_vectors(
        conn,
        author_login=author_login,
        repo=repo,
    )
    return _cluster_from_vectors(
        members,
        vecs,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        kind_label="review_comment",
    )


def cluster_code_chunks(
    conn: sqlite3.Connection,
    *,
    min_cluster_size: int = 3,
    max_cluster_size: int | None = None,
    author_login: str | None = None,
    language: str | None = None,
    repo: str | None = None,
) -> list[Cluster]:
    """Return clusters of `kind='code'` chunks (diff-added blocks).

    Code patterns ride on top of the same retrieval pipeline as review
    comments — see `_cluster_from_vectors` for the shared core. The
    `language` filter is per-chunk (not artifact-level) so a polyglot
    history yields clean per-language clusters. `repo` scopes to one
    repo's commits — combine with `language` for a clean
    "this repo's idioms in this language" rule set.
    """
    members, vecs = _load_code_chunks_with_vectors(
        conn,
        author_login=author_login,
        language=language,
        repo=repo,
    )
    return _cluster_from_vectors(
        members,
        vecs,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        kind_label="code",
    )
