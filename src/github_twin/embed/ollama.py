from __future__ import annotations

import logging

import ollama

log = logging.getLogger(__name__)

# nomic-embed-text is hard-capped at 2048 tokens. Ollama's `truncate=True` is
# documented as the default but is unreliable for symbol-dense content (minified
# JS, dense Scala), so we always truncate client-side. 4000 chars covers
# almost all real cases; the per-item shrink fallback handles the rest.
MAX_CHARS = 4000
MIN_CHARS = 400
SHRINK_FACTOR = 0.6

_CTX_ERROR_MARKER = "context length"


class OllamaEmbedder:
    """Default embedder. Talks to a local Ollama daemon via its Python client.

    Handles oversized inputs with a two-tier strategy:
      1. Truncate every input to `max_chars` before the batch call.
      2. If the batch still fails on context length, embed each item one at a
         time, shrinking each failure by `SHRINK_FACTOR` down to `MIN_CHARS`.
    """

    def __init__(
        self,
        *,
        model: str,
        dim: int,
        host: str | None = None,
        max_chars: int = MAX_CHARS,
    ) -> None:
        self.model = model
        self.dim = dim
        self.model_id = f"ollama:{model}"
        self._client = ollama.Client(host=host) if host else ollama.Client()
        self._dim_verified = False
        self._max_chars = max_chars

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [t[: self._max_chars] for t in texts]
        try:
            vecs = self._embed_raw(prepared)
        except ollama.ResponseError as e:
            if _CTX_ERROR_MARKER not in str(e).lower():
                raise
            log.warning("batch context overflow, falling back to per-item shrink")
            vecs = [self._embed_single_with_shrink(t) for t in prepared]
        self._verify_dim(vecs)
        return vecs

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embed(model=self.model, input=texts)
        return [list(v) for v in resp["embeddings"]]

    def _embed_single_with_shrink(self, text: str) -> list[float]:
        size = len(text)
        while size >= MIN_CHARS:
            try:
                return self._embed_raw([text[:size]])[0]
            except ollama.ResponseError as e:
                if _CTX_ERROR_MARKER not in str(e).lower():
                    raise
                new_size = max(MIN_CHARS, int(size * SHRINK_FACTOR))
                if new_size == size:
                    break
                size = new_size
        log.error("Could not embed chunk even at %d chars; using shortest", MIN_CHARS)
        return self._embed_raw([text[:MIN_CHARS]])[0]

    def _verify_dim(self, vecs: list[list[float]]) -> None:
        if self._dim_verified or not vecs:
            return
        if len(vecs[0]) != self.dim:
            raise RuntimeError(
                f"Embedder dim mismatch: config says {self.dim}, "
                f"model {self.model!r} returned {len(vecs[0])}. "
                "Update config.embed.dim or pick a matching model."
            )
        self._dim_verified = True
