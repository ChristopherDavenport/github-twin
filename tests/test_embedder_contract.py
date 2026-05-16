"""Contract test the Embedder protocol — every backend must pass these.

Currently only `OllamaEmbedder` exists; this test will catch regressions and
serve as the spec when a second backend lands.
"""

import pytest

from github_twin.config import EmbedCfg
from github_twin.embed import Embedder, make_embedder

pytestmark = pytest.mark.embedder


def _ollama_available() -> bool:
    import httpx

    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture
def embedder() -> Embedder:
    if not _ollama_available():
        pytest.skip("Ollama daemon not running on 127.0.0.1:11434")
    return make_embedder(EmbedCfg())


def test_embedder_returns_correct_dim(embedder: Embedder):
    vecs = embedder.embed(["hello"])
    assert len(vecs) == 1
    assert len(vecs[0]) == embedder.dim


def test_embedder_handles_batch(embedder: Embedder):
    vecs = embedder.embed(["hello", "world", "github"])
    assert len(vecs) == 3
    assert all(len(v) == embedder.dim for v in vecs)


def test_embedder_empty_input(embedder: Embedder):
    assert embedder.embed([]) == []


def test_embedder_is_deterministic(embedder: Embedder):
    a = embedder.embed(["the quick brown fox"])[0]
    b = embedder.embed(["the quick brown fox"])[0]
    # Allow tiny float drift but require near-identical.
    delta = sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5
    assert delta < 1e-3


def test_embedder_distinct_inputs_diverge(embedder: Embedder):
    a, b = embedder.embed(["python is great", "javascript is awesome"])
    sim = sum(x * y for x, y in zip(a, b, strict=True))
    # Cosine of two distinct sentences should not be ~1.0.
    assert sim < 0.999


def test_embedder_model_id_is_stable(embedder: Embedder):
    assert isinstance(embedder.model_id, str)
    assert embedder.model_id  # non-empty
