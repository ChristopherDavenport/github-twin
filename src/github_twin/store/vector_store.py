"""Pluggable vector store.

`VectorStore.search` is the only retrieval surface the MCP tools use. Two
implementations:

- `SqliteVecStore` — wraps `queries.vector_search`. Brute-force KNN in
  sqlite-vec. Default. Sub-second up to ~500k vectors.
- `FaissVectorStore` — loads all vectors from sqlite-vec at construction
  and serves search through FAISS. Optional (`pip install github-twin[faiss]`).
  Same filter semantics as the sqlite path; overscan compensates for the
  fact that FAISS searches the full corpus then we intersect with filtered
  candidate IDs.

Both stores return `q.SearchHit`. Switching backends doesn't change
downstream code paths.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from github_twin.store import queries as q

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorSearchFilters:
    chunk_kind: str
    language: str | None = None
    repo: str | None = None
    author_login: str | None = None
    node_kind: str | None = None
    target_id: int | None = None  # None = coalesce across every target; explicit narrows.


@runtime_checkable
class VectorStore(Protocol):
    backend_id: str

    def search(
        self,
        query_vec: list[float],
        *,
        filters: VectorSearchFilters,
        k: int = 5,
    ) -> list[q.SearchHit]: ...


# ---------- sqlite-vec backend (default) ----------


class SqliteVecStore:
    backend_id = "sqlite-vec"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def search(
        self,
        query_vec: list[float],
        *,
        filters: VectorSearchFilters,
        k: int = 5,
    ) -> list[q.SearchHit]:
        return q.vector_search(
            self._conn,
            query_vec=query_vec,
            chunk_kind=filters.chunk_kind,
            language=filters.language,
            repo=filters.repo,
            author_login=filters.author_login,
            node_kind=filters.node_kind,
            target_id=filters.target_id,
            k=k,
        )


# ---------- FAISS backend (opt-in) ----------


class FaissVectorStore:
    """FAISS-backed KNN over an in-memory copy of the sqlite-vec table.

    Loads everything at construction (one full table scan + memcpy into a
    contiguous float32 array). On a 1M-vector corpus that's ~3 GB of RAM at
    768-dim; size accordingly.

    `search` runs FAISS over the full corpus with `k * overscan`, then
    intersects with the SQL-side candidate filter and returns the top-k of
    that intersection. Overscan defaults to 8: enough for typical 5–10%
    filter selectivity without making the FAISS call quadratic.
    """

    backend_id = "faiss"

    def __init__(self, conn: sqlite3.Connection, *, dim: int, overscan: int = 8) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as e:
            raise RuntimeError(
                "faiss-cpu is not installed. "
                "Install with: uv sync --extra faiss  (or pip install github-twin[faiss])"
            ) from e
        self._conn = conn
        self._dim = dim
        self._overscan = overscan
        self._faiss = faiss
        self._np = np

        rows = conn.execute(
            "SELECT chunk_id, embedding FROM vec_chunk ORDER BY chunk_id"
        ).fetchall()
        if not rows:
            log.warning("FaissVectorStore: no vectors in vec_chunk; search will return []")
            self._index = None
            return
        ids = np.array([r["chunk_id"] for r in rows], dtype=np.int64)
        vecs = np.empty((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            vecs[i] = np.frombuffer(r["embedding"], dtype=np.float32, count=dim)
        # L2 matches sqlite-vec's default metric, so the two stores rank the
        # same set of vectors identically (modulo float drift).
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        index.add_with_ids(vecs, ids)
        self._index = index
        log.info("faiss index built: %d vectors (dim=%d)", len(rows), dim)

    def search(
        self,
        query_vec: list[float],
        *,
        filters: VectorSearchFilters,
        k: int = 5,
    ) -> list[q.SearchHit]:
        if self._index is None or self._index.ntotal == 0:
            return []
        np = self._np
        q_arr = np.asarray(query_vec, dtype=np.float32).reshape(1, self._dim)

        # SQL candidate IDs (cheap; index on chunk.kind + optional artifact join).
        candidate_ids = _candidate_chunk_ids(self._conn, filters)
        if not candidate_ids:
            return []

        # Overscan, then intersect with the filter.
        oversample = min(self._index.ntotal, k * self._overscan)
        dists, ids = self._index.search(q_arr, oversample)
        out_ids: list[int] = []
        out_dist: list[float] = []
        keep = candidate_ids  # set
        for chunk_id, d in zip(ids[0].tolist(), dists[0].tolist(), strict=True):
            if chunk_id < 0:  # FAISS sentinel for "no neighbor"
                continue
            if chunk_id in keep:
                out_ids.append(chunk_id)
                out_dist.append(float(d))
                if len(out_ids) >= k:
                    break
        if not out_ids:
            return []
        return _materialize_hits(self._conn, out_ids, out_dist)


# ---------- helpers ----------


def _candidate_chunk_ids(conn: sqlite3.Connection, f: VectorSearchFilters) -> set[int]:
    where: list[str] = ["c.kind = ?"]
    params: list[Any] = [f.chunk_kind]
    if f.language is not None:
        where.append("c.language = ?")
        params.append(f.language)
    if f.node_kind is not None:
        where.append("c.node_kind = ?")
        params.append(f.node_kind)
    needs_artifact = f.repo is not None or f.author_login is not None or f.target_id is not None
    if f.target_id is not None:
        where.append("a.target_id = ?")
        params.append(f.target_id)
    if f.repo is not None:
        where.append("a.repo = ?")
        params.append(f.repo)
    if f.author_login is not None:
        where.append("a.author_login = ?")
        params.append(f.author_login)
    join = " JOIN artifact a ON a.id = c.artifact_id" if needs_artifact else ""
    sql = f"SELECT c.id FROM chunk c{join} WHERE " + " AND ".join(where)
    return {r["id"] for r in conn.execute(sql, params).fetchall()}


def _materialize_hits(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
    distances: list[float],
) -> list[q.SearchHit]:
    """Build SearchHits for FAISS results. Preserves the input ordering."""
    import json

    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT
          c.id AS chunk_id, c.artifact_id, c.text, c.context_json,
          c.language AS chunk_language,
          a.kind AS artifact_kind, a.repo AS artifact_repo,
          a.source_url AS artifact_source_url, a.decision AS artifact_decision,
          a.target_id AS target_id, t.name AS target_name, t.kind AS target_kind
        FROM chunk c
        JOIN artifact a ON a.id = c.artifact_id
        JOIN target t ON t.id = a.target_id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    by_id = {r["chunk_id"]: r for r in rows}
    hits: list[q.SearchHit] = []
    for cid, dist in zip(chunk_ids, distances, strict=True):
        r = by_id.get(cid)
        if r is None:
            continue
        hits.append(
            q.SearchHit(
                chunk_id=r["chunk_id"],
                artifact_id=r["artifact_id"],
                distance=dist,
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
        )
    return hits


# ---------- hybrid retrieval (RRF fusion of vector + BM25) ----------

RRF_K = 60
HYBRID_FETCH = 50


def hybrid_search(
    store: VectorStore,
    conn: sqlite3.Connection,
    *,
    query_vec: list[float],
    query_text: str,
    filters: VectorSearchFilters,
    k: int = 5,
    fetch: int = HYBRID_FETCH,
    expander: Any | None = None,
) -> list[q.SearchHit]:
    """Reciprocal Rank Fusion over `store.search` (vector) and `q.bm25_search`
    (keyword). Both retrievers see the same filter set; results are fused
    with the standard RRF formula `1/(k + rank)` and the top-k returned.

    `SearchHit.distance` is rewritten to `1 - rrf_score` so existing tools
    that just display the field keep working — lower is still better. Tools
    that interpret distance as raw L2 (currently only `predict_review_outcome`)
    must not be wired through here.

    Asymmetry note: `expander` is passed to the BM25 leg only. The vector
    leg consumes `query_vec` as-is, never an expanded form — embeddings
    already capture synonymy, so expansion there dilutes the dense match.
    `test_hybrid_search.py:test_hybrid_passes_raw_vec_when_expander_set`
    pins this contract.
    """
    vec_hits = store.search(query_vec, filters=filters, k=fetch)
    bm25_hits = q.bm25_search(
        conn,
        query_text=query_text,
        chunk_kind=filters.chunk_kind,
        language=filters.language,
        repo=filters.repo,
        author_login=filters.author_login,
        node_kind=filters.node_kind,
        target_id=filters.target_id,
        k=fetch,
        expander=expander,
    )
    scores: dict[int, float] = {}
    by_id: dict[int, q.SearchHit] = {}
    for rank, h in enumerate(vec_hits):
        scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        by_id.setdefault(h.chunk_id, h)
    for rank, h in enumerate(bm25_hits):
        scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        by_id.setdefault(h.chunk_id, h)
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [replace(by_id[cid], distance=1.0 - score) for cid, score in ranked]


# ---------- dispatch ----------


def make_vector_store(conn: sqlite3.Connection, *, backend: str, dim: int) -> VectorStore:
    if backend == "sqlite-vec":
        return SqliteVecStore(conn)
    if backend == "faiss":
        return FaissVectorStore(conn, dim=dim)
    raise ValueError(f"Unknown vector store backend: {backend!r}")
