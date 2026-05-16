from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into vectors of a fixed dimension.

    Implementations must be deterministic for identical inputs (modulo float drift)
    and return vectors of length `dim`. `model_id` is stamped into the store so that
    a model change is detectable at query time.
    """

    dim: int
    model_id: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...
