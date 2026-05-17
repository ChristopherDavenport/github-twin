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

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.genai import Client


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
