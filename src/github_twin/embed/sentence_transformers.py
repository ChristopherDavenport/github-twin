"""Sentence-Transformers embedder.

A pure-Python alternative to OllamaEmbedder. Useful when you want bulk-batch
throughput (especially on a GPU) without standing up an Ollama daemon, or
when you need an embedder that's reproducible across machines without
network deps.

The default model `BAAI/bge-small-en-v1.5` produces 384-dim vectors. Pick a
larger model (e.g. `BAAI/bge-large-en-v1.5` at 1024-dim) for quality at the
cost of latency. Set `cfg.embed.model` and `cfg.embed.dim` accordingly —
the existing dim-check in `db._ensure_vec_table` will refuse a swap on an
existing DB.

The dependency is opt-in: install with `uv sync --extra st` (or
`pip install github-twin[st]`).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Same char-cap pattern as OllamaEmbedder so chunkers can stay backend-agnostic.
MAX_CHARS = 4000


class SentenceTransformersEmbedder:
    """Loads a sentence-transformers model lazily on first `embed()` call.

    `device` defaults to whatever the library autodetects (cuda > mps > cpu).
    Override via `cfg.embed.device` if you want to pin it.
    """

    def __init__(
        self,
        *,
        model: str,
        dim: int,
        device: str | None = None,
        max_chars: int = MAX_CHARS,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.dim = dim
        self.model_id = f"sentence-transformers:{model}"
        self._device = device
        self._max_chars = max_chars
        self._batch_size = batch_size
        self._model = None  # lazy
        self._dim_verified = False

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: uv sync --extra st  (or pip install github-twin[st])"
            ) from e
        log.info(
            "loading sentence-transformers model %s (device=%s)", self.model, self._device or "auto"
        )
        self._model = SentenceTransformer(self.model, device=self._device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [t[: self._max_chars] for t in texts]
        model = self._load()
        # `normalize_embeddings=True` returns unit vectors so cosine distance
        # downstream matches what the clusterer expects.
        arr = model.encode(
            prepared,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vecs = [list(map(float, row)) for row in arr]
        self._verify_dim(vecs)
        return vecs

    def _verify_dim(self, vecs: list[list[float]]) -> None:
        if self._dim_verified or not vecs:
            return
        if len(vecs[0]) != self.dim:
            raise RuntimeError(
                f"Embedder dim mismatch: config says {self.dim}, "
                f"model {self.model!r} returned {len(vecs[0])}. "
                "Update cfg.embed.dim or pick a matching model."
            )
        self._dim_verified = True
