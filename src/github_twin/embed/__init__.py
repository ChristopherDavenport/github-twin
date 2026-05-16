from __future__ import annotations

from github_twin.config import EmbedCfg
from github_twin.embed.base import Embedder
from github_twin.embed.ollama import OllamaEmbedder


def make_embedder(cfg: EmbedCfg) -> Embedder:
    if cfg.backend == "ollama":
        return OllamaEmbedder(model=cfg.model, dim=cfg.dim, host=cfg.ollama_host)
    if cfg.backend == "sentence_transformers":
        # Lazy import keeps the heavy sentence-transformers dep optional.
        from github_twin.embed.sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        return SentenceTransformersEmbedder(
            model=cfg.model,
            dim=cfg.dim,
            device=cfg.device,
            batch_size=cfg.batch_size,
        )
    raise ValueError(f"Unknown embed backend: {cfg.backend!r}")


__all__ = ["Embedder", "OllamaEmbedder", "make_embedder"]
