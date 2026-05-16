"""Gemini embedder.

A remote-API embedder for users who have a Gemini API key but neither
Ollama nor the `[st]` extra installed. `google-genai` is already a hard
dep of the project (used by `distill/synth.py` and `eval/llm.py`), so no
extra is required.

Unlike `OllamaEmbedder` and `SentenceTransformersEmbedder`, this backend
sends chunk text off-box to Google. It is the only embedder that does
so — pick it deliberately if your corpus has nothing private in it, or
if you have a license / agreement that permits the transfer.

Auth: the Gemini SDK reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the
environment when `api_key=None`. Pass `api_key` explicitly to override.

Defaults aimed at `gemini-embedding-001` (3072-dim, the highest-quality
general embedding model). Set `cfg.embed.dim` to 1536 or 768 to request
a shorter output via `output_dimensionality`.

All calls use `task_type="RETRIEVAL_DOCUMENT"`. The Embedder Protocol
doesn't distinguish query vs document embeds — query-side calls take a
small quality hit. This mirrors the asymmetry punt already made for
nomic-embed-text and bge.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Same char-cap pattern as OllamaEmbedder. Gemini's input cap is in tokens
# (~2048 for gemini-embedding-001); 4000 chars is a safe rough mapping.
MAX_CHARS = 4000

# gemini-embedding-001 accepts up to 100 inputs per embed_content request.
DEFAULT_BATCH_SIZE = 100


class GeminiEmbedder:
    """Remote Gemini embedder via the `google.genai` SDK."""

    def __init__(
        self,
        *,
        model: str,
        dim: int,
        api_key: str | None = None,
        max_chars: int = MAX_CHARS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        from google import genai

        self.model = model
        self.dim = dim
        self.model_id = f"gemini:{model}"
        self._max_chars = max_chars
        self._batch_size = batch_size
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._dim_verified = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [t[: self._max_chars] for t in texts]
        vecs: list[list[float]] = []
        for start in range(0, len(prepared), self._batch_size):
            sub = prepared[start : start + self._batch_size]
            vecs.extend(self._embed_raw(sub))
        self._verify_dim(vecs)
        return vecs

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        resp = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.dim,
            ),
        )
        embeddings = resp.embeddings or []
        return [list(e.values or []) for e in embeddings]

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
