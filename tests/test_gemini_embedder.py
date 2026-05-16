"""Smoke tests for the Gemini embedder.

`google-genai` is a hard dep, so no skip gating is needed. We patch
`google.genai.Client` to avoid any real network calls and assert the
shape contract the rest of the pipeline relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from github_twin.config import EmbedCfg
from github_twin.embed import make_embedder

GEMINI_MODEL = "gemini-embedding-001"
GEMINI_DIM = 3072


@dataclass
class _FakeEmbedding:
    values: list[float]


@dataclass
class _FakeEmbedResponse:
    embeddings: list[_FakeEmbedding]


@dataclass
class _FakeModels:
    dim: int
    calls: list[dict[str, Any]] = field(default_factory=list)

    def embed_content(self, *, model: str, contents: list[str], config: Any) -> _FakeEmbedResponse:
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        return _FakeEmbedResponse(
            embeddings=[_FakeEmbedding(values=[0.1] * self.dim) for _ in contents]
        )


@dataclass
class _FakeClient:
    api_key: str | None = None
    models: _FakeModels = field(default_factory=lambda: _FakeModels(dim=GEMINI_DIM))


def _install_fake(monkeypatch: pytest.MonkeyPatch, *, dim: int = GEMINI_DIM) -> _FakeClient:
    """Patch google.genai.Client so constructor returns our fake."""
    fake = _FakeClient(models=_FakeModels(dim=dim))

    def _factory(*args: Any, **kwargs: Any) -> _FakeClient:
        fake.api_key = kwargs.get("api_key")
        return fake

    from google import genai

    monkeypatch.setattr(genai, "Client", _factory)
    return fake


def test_make_embedder_dispatches_to_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = make_embedder(EmbedCfg(backend="gemini", model=GEMINI_MODEL, dim=GEMINI_DIM))
    from github_twin.embed.gemini import GeminiEmbedder

    assert isinstance(emb, GeminiEmbedder)
    assert emb.model_id == f"gemini:{GEMINI_MODEL}"
    assert emb.dim == GEMINI_DIM


def test_gemini_embedder_round_trip_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = make_embedder(EmbedCfg(backend="gemini", model=GEMINI_MODEL, dim=GEMINI_DIM))
    out = emb.embed(["hello world", "another sentence"])
    assert len(out) == 2
    assert all(len(v) == GEMINI_DIM for v in out)


def test_gemini_embedder_handles_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake(monkeypatch)
    emb = make_embedder(EmbedCfg(backend="gemini", model=GEMINI_MODEL, dim=GEMINI_DIM))
    assert emb.embed([]) == []
    assert fake.models.calls == []  # never hits the client


def test_gemini_embedder_dim_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake returns 1536-dim vectors but cfg says 3072 — must fail loud.
    _install_fake(monkeypatch, dim=1536)
    emb = make_embedder(EmbedCfg(backend="gemini", model=GEMINI_MODEL, dim=GEMINI_DIM))
    with pytest.raises(RuntimeError, match="dim mismatch"):
        emb.embed(["x"])


def test_gemini_embedder_batches_oversize_input(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake(monkeypatch)
    from github_twin.embed.gemini import GeminiEmbedder

    emb = GeminiEmbedder(model=GEMINI_MODEL, dim=GEMINI_DIM, batch_size=2)
    out = emb.embed(["a", "b", "c", "d", "e"])
    assert len(out) == 5
    # 5 items at batch_size=2 -> 3 requests (2 + 2 + 1)
    assert len(fake.models.calls) == 3
    assert [len(c["contents"]) for c in fake.models.calls] == [2, 2, 1]


def test_gemini_embedder_truncates_to_max_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake(monkeypatch)
    from github_twin.embed.gemini import MAX_CHARS

    emb = make_embedder(EmbedCfg(backend="gemini", model=GEMINI_MODEL, dim=GEMINI_DIM))
    huge = "x" * 10_000
    emb.embed([huge])
    sent = fake.models.calls[0]["contents"][0]
    assert len(sent) == MAX_CHARS
