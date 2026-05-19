"""Shared Gemini client construction.

Encodes the auth precedence used by every Gemini surface in the project
(embed, distill/synth, eval/llm — and summarize via the eval/llm seam).

Precedence:
    1. Explicit `api_key` (or `GEMINI_API_KEY` / `GOOGLE_API_KEY` env) →
       Google AI Studio API path.
    2. `GT_GEMINI_PROJECT` env set (with `GT_GEMINI_LOCATION` defaulting to
       `us-central1`) → Vertex AI via Application Default Credentials.
       Bootstrap once with `gcloud auth application-default login`.
    3. Bare `genai.Client()` — preserves the SDK's own env auto-config so
       any future auth mode it ships works without changes here.

The `GT_`-namespaced env vars are deliberate: we don't want to silently
hijack a globally-set `GOOGLE_CLOUD_PROJECT` that the user keeps around
for unrelated gcloud work. Opting into the Vertex path is explicit.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.genai import Client
    from google.genai.types import ThinkingConfig


log = logging.getLogger(__name__)

DEFAULT_LOCATION = "us-central1"


def make_gemini_client(api_key: str | None = None) -> Client:
    """Construct a `google.genai.Client` following the project auth precedence.

    Pass `api_key` explicitly when the caller already resolved one (the
    existing backends accept a constructor kwarg). Otherwise the helper
    falls back to ADC if `GT_GEMINI_PROJECT` is set, else to the SDK's
    own env auto-config.
    """
    from google import genai

    if api_key:
        return genai.Client(api_key=api_key)
    project = os.environ.get("GT_GEMINI_PROJECT")
    if project:
        location = os.environ.get("GT_GEMINI_LOCATION", DEFAULT_LOCATION)
        # vertexai=True flips the SDK to the Vertex AI endpoint; ADC is
        # picked up from ~/.config/gcloud/application_default_credentials.json.
        kwargs: dict[str, Any] = {
            "vertexai": True,
            "project": project,
            "location": location,
        }
        return genai.Client(**kwargs)
    return genai.Client()


def make_thinking_config(model: str) -> ThinkingConfig | None:
    """Lowest-reasoning ThinkingConfig appropriate for `model`, or None.

    Our distill / summarize / eval tasks are short-output pattern recognition;
    they don't benefit from extended chain-of-thought, and reasoning tokens
    eat into the same output budget that holds the visible response. Pin
    thinking to the floor so `max_output_tokens` is spent on the actual JSON
    or summary, not internal reasoning that gets truncated mid-string.

    - `gemini-3*`: `thinking_level="minimal"` — 3.x Flash-Lite is a thinking
      model by design; "minimal" is the lowest exposed setting.
    - `gemini-2.5*`: `thinking_budget=0` — fully disables thinking on the 2.5
      family.
    - anything else (2.0 and older non-thinking models, future families we
      don't recognize): None, so the caller omits `thinking_config` and the
      SDK's own default applies.
    """
    from google.genai import types

    if model.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
    if model.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=0)
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Pull a `Retry-After` value off a google.genai APIError, if present.

    Vertex sometimes sends a `Retry-After` header on 429 responses. The SDK
    exposes the underlying HTTP response on `APIError.response`. The header
    is either a delta-seconds integer or an HTTP-date; we only support the
    integer form (the HTTP-date form is rare from Google APIs in practice).
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def with_retry[T](
    fn: Callable[..., T],
    *args: Any,
    max_attempts: int = 5,
    base_delay: float = 4.0,
    max_delay: float = 64.0,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> T:
    """Call `fn(*args, **kwargs)` with retry on transient Gemini errors.

    Retries on:
    - `ClientError` with HTTP 429 (rate limit / quota exhausted).
    - `ServerError` (any 5xx).

    Everything else (auth errors, 400s, schema violations) raises
    immediately — those are not transient and retrying just delays the
    real diagnosis.

    Backoff: exponential starting at `base_delay`, doubling each attempt,
    capped at `max_delay`, with full jitter (random multiplier in
    [0.5, 1.5)). If the server sends a `Retry-After` header, we sleep for
    `max(retry_after, computed_backoff)` so we never undershoot the
    server's hint.

    `sleep` is injectable so tests can drive the backoff without wall-clock
    waits.
    """
    from google.genai import errors

    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except errors.ClientError as exc:
            if getattr(exc, "code", None) != 429 or attempt == max_attempts - 1:
                raise
            backoff = min(base_delay * (2**attempt), max_delay)
            backoff *= 0.5 + random.random()
            hint = _retry_after_seconds(exc)
            delay = max(hint, backoff) if hint is not None else backoff
            log.warning(
                "Gemini 429 (attempt %d/%d), sleeping %.1fs",
                attempt + 1,
                max_attempts,
                delay,
            )
            sleep(delay)
        except errors.ServerError as exc:
            if attempt == max_attempts - 1:
                raise
            backoff = min(base_delay * (2**attempt), max_delay)
            backoff *= 0.5 + random.random()
            log.warning(
                "Gemini %s (attempt %d/%d), sleeping %.1fs",
                getattr(exc, "code", "5xx"),
                attempt + 1,
                max_attempts,
                backoff,
            )
            sleep(backoff)
    # Unreachable: the final attempt either returns or raises above.
    raise RuntimeError("with_retry: exhausted attempts without returning or raising")


def has_gemini_auth() -> bool:
    """True iff any Gemini auth path is configured in the environment.

    Used by `make_synthesizer` / `make_text_llm` to decide whether the
    `auto` backend selector should consider Gemini eligible.
    """
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GT_GEMINI_PROJECT")
    )
