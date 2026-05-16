"""Smoke tests for the sentence-transformers embedder.

The dependency is opt-in (`uv sync --extra st`). When it isn't installed,
these tests are skipped — the rest of the suite still runs.
"""

from __future__ import annotations

import pytest

from github_twin.config import EmbedCfg
from github_twin.embed import make_embedder

# 384 is the dim of BAAI/bge-small-en-v1.5; we use it everywhere here.
ST_MODEL = "BAAI/bge-small-en-v1.5"
ST_DIM = 384


def _require_st():
    """Skip the test if sentence-transformers (or its transitive deps) is missing."""
    pytest.importorskip("sentence_transformers")


def test_make_embedder_dispatches_to_st():
    """Dispatch happens without importing the dep — the lazy import in
    make_embedder must defer until the backend is actually requested."""
    _require_st()
    cfg = EmbedCfg(backend="sentence_transformers", model=ST_MODEL, dim=ST_DIM)
    emb = make_embedder(cfg)
    from github_twin.embed.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    assert isinstance(emb, SentenceTransformersEmbedder)
    assert emb.model_id == f"sentence-transformers:{ST_MODEL}"
    assert emb.dim == ST_DIM


def test_st_embedder_round_trip_shape():
    _require_st()
    emb = make_embedder(EmbedCfg(backend="sentence_transformers", model=ST_MODEL, dim=ST_DIM))
    out = emb.embed(["hello world", "another sentence"])
    assert len(out) == 2
    assert all(len(v) == ST_DIM for v in out)


def test_st_embedder_handles_empty_input():
    _require_st()
    emb = make_embedder(EmbedCfg(backend="sentence_transformers", model=ST_MODEL, dim=ST_DIM))
    assert emb.embed([]) == []


def test_st_embedder_dim_mismatch_raises():
    """If `cfg.embed.dim` doesn't match the model's actual dim, fail loud
    instead of writing wrong-shape vectors into the store."""
    _require_st()
    emb = make_embedder(EmbedCfg(backend="sentence_transformers", model=ST_MODEL, dim=999))
    with pytest.raises(RuntimeError, match="dim mismatch"):
        emb.embed(["x"])


def test_make_embedder_unknown_backend_raises():
    """Sanity: bad config still raises ValueError before reaching imports."""
    with pytest.raises(ValueError, match="Unknown embed backend"):
        make_embedder(EmbedCfg(backend="nonsense"))
