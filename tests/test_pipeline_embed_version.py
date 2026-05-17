"""EMBED_TEXT_VERSION forced re-embed migration.

When the stored version in `sync_cursor` is behind `pipeline.EMBED_TEXT_VERSION`,
`run_embed` wipes `vec_chunk`, nulls `chunk.embed_model`, and re-embeds
everything. New DBs (no vectors yet) just get stamped at the current
version with no migration churn.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.config import Config, EmbedCfg, VectorStoreCfg
from github_twin.pipeline import _EMBED_VERSION_KEY, EMBED_TEXT_VERSION, run_embed
from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target


class FakeEmbedder:
    dim = 4
    model_id = "fake-v1"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        # Returns one zero-padded vector per input, distinctive per call
        # so callers can distinguish runs.
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


@pytest.fixture
def cfg(tmp_path: Path):
    return Config(
        embed=EmbedCfg(backend="fake", batch_size=10, dim=4),
        vector_store=VectorStoreCfg(backend="sqlite-vec"),
    )


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "embed_version.sqlite", embed_dim=FakeEmbedder.dim)
    seed_target(db)
    yield db
    db.close()


def _seed_chunks(conn, n: int = 3) -> list[int]:
    ids = []
    for i in range(n):
        aid = q.upsert_artifact(
            conn,
            target_id=1,
            kind="commit",
            external_id=f"c-{i}",
            source_url=None,
            repo="me/x",
            language="python",
            author_email=None,
            author_login=None,
            created_at=None,
            decision=None,
            meta=None,
        )
        cid = q.insert_chunk(
            conn,
            artifact_id=aid,
            kind="code",
            text=f"def f{i}(): return {i}",
            context={
                "path": f"src/m{i}.py",
                "symbol_name": f"f{i}",
                "node_kind": "function_definition",
            },
            language="python",
        )
        ids.append(cid)
    return ids


def test_brand_new_db_does_not_trigger_rewrite(conn, cfg):
    """Fresh DB (no vectors yet) → version mismatch shouldn't force a wipe;
    the first embed pass is simply the baseline at the current version."""
    _seed_chunks(conn, 2)
    embedder = FakeEmbedder()
    n = run_embed(cfg, conn, embedder=embedder)
    assert n == 2
    # The version cursor is now stamped at the current value.
    assert q.get_cursor(conn, _EMBED_VERSION_KEY) == str(EMBED_TEXT_VERSION)
    # No second embed call set required — only one batch.
    assert len(embedder.calls) == 1


def test_existing_v1_db_triggers_full_reembed(conn, cfg):
    """Pre-existing vectors at v1 → on next run_embed, wipe and re-embed all."""
    cids = _seed_chunks(conn, 3)
    # Simulate a v1 corpus: pretend the chunks were already embedded.
    q.set_cursor(conn, _EMBED_VERSION_KEY, "1")
    for cid in cids:
        q.write_embedding(conn, chunk_id=cid, embedding=[0.5, 0.0, 0.0, 0.0], model_id="legacy")
    # Sanity: vectors are present.
    n_vec = conn.execute("SELECT COUNT(*) AS n FROM vec_chunk").fetchone()["n"]
    assert n_vec == 3

    embedder = FakeEmbedder()
    n = run_embed(cfg, conn, embedder=embedder)
    assert n == 3, "all chunks should re-embed on a version bump"
    assert q.get_cursor(conn, _EMBED_VERSION_KEY) == str(EMBED_TEXT_VERSION)
    # The texts the embedder saw must include the new prefix headers
    # (proof that the re-embed path uses prefix_chunk, not raw chunk.text).
    embedded_texts = [t for batch in embedder.calls for t in batch]
    assert any("# src/m0.py :: f0" in t for t in embedded_texts)


def test_current_version_corpus_is_idempotent(conn, cfg):
    """When stored version matches current and vectors exist, run_embed has
    nothing to do — no wipe, no re-embed, version stays put."""
    cids = _seed_chunks(conn, 2)
    q.set_cursor(conn, _EMBED_VERSION_KEY, str(EMBED_TEXT_VERSION))
    for cid in cids:
        q.write_embedding(conn, chunk_id=cid, embedding=[0.5, 0.0, 0.0, 0.0], model_id="legacy")
    # `embed_model` is set for these chunks, so pending_embed_chunks returns nothing.
    embedder = FakeEmbedder()
    n = run_embed(cfg, conn, embedder=embedder)
    assert n == 0
    assert embedder.calls == []
    # Vectors still present.
    n_vec = conn.execute("SELECT COUNT(*) AS n FROM vec_chunk").fetchone()["n"]
    assert n_vec == 2


def test_rebuild_flag_bypasses_version_check(conn, cfg):
    """`--rebuild` is the explicit "wipe and redo" path — it works at any
    version state, including current."""
    cids = _seed_chunks(conn, 2)
    q.set_cursor(conn, _EMBED_VERSION_KEY, str(EMBED_TEXT_VERSION))
    for cid in cids:
        q.write_embedding(conn, chunk_id=cid, embedding=[0.5, 0.0, 0.0, 0.0], model_id="legacy")
    embedder = FakeEmbedder()
    n = run_embed(cfg, conn, embedder=embedder, rebuild=True)
    assert n == 2
