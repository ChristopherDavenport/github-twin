"""OpenTelemetry integration — auto-detected via env vars, stdout-safe.

Design constraints that MUST hold for the MCP server:

1. **stdio is sacred.** The MCP protocol talks JSON over stdin/stdout. A
   stray console exporter would corrupt the channel and break Claude
   Code. We deliberately never register a ConsoleSpanExporter; the
   OTLP HTTP exporter posts to the configured endpoint and is the
   only output sink we wire up.
2. **Failures stay local.** A misconfigured endpoint, a partial dep
   install, or a malformed env var must never raise into a tool
   handler. Every OTel-touching path is wrapped or fenced.
3. **The API is always importable.** `opentelemetry-api` is a hard
   dep so call sites can use `tracer().start_as_current_span(...)`
   without conditional imports. When no SDK is installed or no
   endpoint is configured, the API returns its built-in noop tracer
   and span creation is essentially free.

Activation:

- Auto-on when `OTEL_EXPORTER_OTLP_ENDPOINT` (or the trace-specific
  variant) is set AND the `[otel]` extra is installed (provides
  `opentelemetry-sdk` and `opentelemetry-exporter-otlp-proto-http`).
- Auto-off otherwise. `OTEL_SDK_DISABLED=true` forces off even when
  endpoints are set, matching the standard OTel SDK behaviour.

Env vars consulted (standard OTel SDK ones; we don't invent any):
- `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`
- `OTEL_SERVICE_NAME` (default: `github-twin`)
- `OTEL_RESOURCE_ATTRIBUTES`
- `OTEL_SDK_DISABLED`

Call `init_otel()` once during process startup (the CLI main callback
does this; the MCP server inherits because it boots through the CLI).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

log = logging.getLogger(__name__)

_TRACER_NAME = "github_twin"
_initialized = False
_real_sdk_active = False


def _truthy_env(name: str) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def is_otel_configured() -> bool:
    """Return True when env vars indicate the user wants OTel on.

    Doesn't probe for installed deps — `init_otel()` does the full
    "configured AND deps available" check and reports whether real
    export is happening."""
    if _truthy_env("OTEL_SDK_DISABLED"):
        return False
    return bool(
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )


def is_active() -> bool:
    """True when `init_otel()` actually wired a real exporter.
    False during noop runs (no env vars, missing SDK, or init failed)."""
    return _real_sdk_active


def init_otel(service_name: str = "github-twin") -> bool:
    """Configure the OTel tracer SDK if env vars opt in AND the SDK
    package is installed. Idempotent. Returns `is_active()` after the
    call so callers can log whether telemetry is live."""
    global _initialized, _real_sdk_active
    if _initialized:
        return _real_sdk_active
    _initialized = True

    if not is_otel_configured():
        return False

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        # SDK not installed — leave the API's noop providers in place.
        # User wanted OTel (env var set) but didn't install `[otel]`.
        log.debug(
            "OpenTelemetry SDK not installed; install with the `[otel]` extra. Spans will be noop."
        )
        return False

    try:
        from github_twin import __version__
    except Exception:  # pragma: no cover — extremely defensive
        __version__ = "0.0.0+unknown"

    try:
        resource = Resource.create(
            {
                "service.name": os.environ.get("OTEL_SERVICE_NAME", service_name),
                "service.version": __version__,
            }
        )
        provider = TracerProvider(resource=resource)
        # Bounded queue: the batcher drops spans rather than blocking the
        # MCP request thread when the collector is slow or down.
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(),
                max_queue_size=2048,
                max_export_batch_size=512,
                schedule_delay_millis=5000,
            )
        )
        trace.set_tracer_provider(provider)
        _real_sdk_active = True
        log.info("OpenTelemetry tracing initialized; exporting to OTLP HTTP endpoint")
        return True
    except Exception as exc:  # noqa: BLE001
        # A broken endpoint URL or a TLS misconfig must never crash MCP.
        log.warning("OpenTelemetry init failed (%s); continuing without telemetry", exc)
        return False


def tracer() -> Tracer:
    """Return the github_twin tracer. Always safe to call — when no
    SDK is configured, the API returns a noop tracer whose spans
    have ~zero overhead."""
    return trace.get_tracer(_TRACER_NAME)


def set_safe_attributes(span: Span, **attrs: Any) -> None:
    """Set span attributes, dropping None/empty values. None isn't a
    legal attribute value in OTel; this avoids littering the trace
    with explicit `null` attributes."""
    for k, v in attrs.items():
        if v is None:
            continue
        # OTel attribute values must be primitive or list of primitives.
        if isinstance(v, str | bool | int | float):
            span.set_attribute(k, v)
        elif isinstance(v, list | tuple):
            try:
                span.set_attribute(k, list(v))
            except Exception:  # noqa: BLE001 - defensive against weird types
                span.set_attribute(k, str(v))
        else:
            span.set_attribute(k, str(v))
