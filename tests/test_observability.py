"""OpenTelemetry wiring — must never break the MCP stdio channel.

Critical invariants:
  1. `init_otel()` is a no-op when no OTLP endpoint env var is set.
  2. Tool calls never write to stdout, on OR off — MCP speaks JSON
     over stdin/stdout and any stray print would corrupt the channel.
  3. `init_otel()` is idempotent (re-entries don't reconfigure).
  4. `tracer()` always returns something whose `start_as_current_span`
     works as a context manager, configured or not.
  5. With an OTLP endpoint set, the SDK initializes a real provider
     and our span attributes round-trip into the captured spans.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from github_twin import observability as obs


def _reset_otel_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `_initialized` / `_real_sdk_active` module globals persist
    across tests. Reset them so each test sees a fresh init_otel."""
    monkeypatch.setattr(obs, "_initialized", False)
    monkeypatch.setattr(obs, "_real_sdk_active", False)


# ---------- no-op behavior ----------


def test_init_otel_returns_false_without_env_vars(monkeypatch: pytest.MonkeyPatch):
    _reset_otel_state(monkeypatch)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert obs.init_otel() is False
    assert obs.is_active() is False


def test_init_otel_respects_disabled_flag(monkeypatch: pytest.MonkeyPatch):
    """OTEL_SDK_DISABLED is the standard kill switch; it must override
    a configured endpoint."""
    _reset_otel_state(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert obs.init_otel() is False
    assert obs.is_active() is False


def test_init_otel_is_idempotent(monkeypatch: pytest.MonkeyPatch):
    """Calling init_otel a second time must short-circuit, even if
    env vars change in between."""
    _reset_otel_state(monkeypatch)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert obs.init_otel() is False
    # Add the env var after the first call — the second should still be off.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    assert obs.init_otel() is False  # still off because already initialized


def test_tracer_is_always_callable(monkeypatch: pytest.MonkeyPatch):
    """The API surface must not depend on init state — calls to
    `tracer().start_as_current_span` work whether OTel is real or not."""
    _reset_otel_state(monkeypatch)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    obs.init_otel()
    with obs.tracer().start_as_current_span("noop") as span:
        # set_attribute on the noop API span is a no-op but must not raise.
        span.set_attribute("anything", "value")


# ---------- stdout safety ----------


def test_mcp_tool_call_does_not_write_to_stdout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Smoke: a tool call (which exercises the OTel spans we added)
    must not emit anything on stdout. MCP would treat any non-JSON line
    on stdout as a protocol violation."""
    from github_twin.mcp_server.tools import find_style_examples
    from github_twin.store import queries as q
    from github_twin.store.db import open_db
    from github_twin.store.vector_store import SqliteVecStore

    _reset_otel_state(monkeypatch)

    class FakeEmbedder:
        dim = 4
        model_id = "fake"

        def embed(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    from tests.conftest import seed_target

    db = open_db(tmp_path / "obs.sqlite", embed_dim=4)
    seed_target(db)
    try:
        aid = q.upsert_artifact(
            db,
            target_id=1,
            kind="commit",
            external_id="a-1",
            source_url=None,
            repo="me/x",
            language="python",
            author_email=None,
            author_login=None,
            created_at=None,
            decision=None,
            meta=None,
        )
        cid = q.insert_chunk(
            db,
            artifact_id=aid,
            kind="code",
            text="def f(): pass",
            context={"language": "python"},
            language="python",
        )
        q.write_embedding(db, chunk_id=cid, embedding=[1.0, 0, 0, 0], model_id="fake")
        store = SqliteVecStore(db)

        buf = io.StringIO()
        with redirect_stdout(buf):
            obs.init_otel()  # no env vars → noop, but still exercise the path
            hits = find_style_examples(db, FakeEmbedder(), store, query="test", k=5)
        assert hits, "sanity: tool should have returned the seeded chunk"
        assert buf.getvalue() == "", (
            f"MCP stdio would break if anything lands on stdout. Captured: {buf.getvalue()!r}"
        )
    finally:
        db.close()


# ---------- real SDK activation ----------


def test_init_otel_activates_when_endpoint_set(monkeypatch: pytest.MonkeyPatch):
    """When the endpoint env var is set AND the SDK is installed, the
    real provider gets wired. We don't actually export to a real
    collector; the BatchSpanProcessor's network call happens in a
    background thread and is harmless if the endpoint is unreachable."""
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")

    _reset_otel_state(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:14318")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    assert obs.init_otel() is True
    assert obs.is_active() is True


def test_spans_capture_attributes_under_real_sdk():
    """End-to-end on a non-global provider: with a real SDK, attributes
    set via `set_safe_attributes` reach the exported span. Builds a
    private TracerProvider so we don't fight the OTel API's "no
    overriding the global provider" guard that other tests trigger."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("github_twin_test")

    with tracer.start_as_current_span("test.span") as span:
        obs.set_safe_attributes(
            span,
            **{
                "gh_twin.tool.k": 5,
                "gh_twin.filter.repo": "me/x",
                "gh_twin.filter.language": None,  # should be dropped
                "gh_twin.list.attr": ["a", "b"],
            },
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs.get("gh_twin.tool.k") == 5
    assert attrs.get("gh_twin.filter.repo") == "me/x"
    assert "gh_twin.filter.language" not in attrs  # None dropped
    assert tuple(attrs.get("gh_twin.list.attr") or ()) == ("a", "b")


def test_set_safe_attributes_handles_unusual_types():
    """`set_safe_attributes` must coerce types the OTel attribute API
    rejects (sets, dicts, etc.) rather than raise into the tool body."""

    # We don't need a real span — use a stub that records what was set.
    class StubSpan:
        def __init__(self):
            self.calls: list[tuple[str, object]] = []

        def set_attribute(self, key: str, value: object) -> None:
            self.calls.append((key, value))

    span = StubSpan()
    obs.set_safe_attributes(
        span,  # type: ignore[arg-type]
        **{
            "primitive.str": "x",
            "primitive.int": 7,
            "primitive.bool": True,
            "primitive.float": 1.5,
            "primitive.none": None,
            "list.ints": [1, 2, 3],
            "tuple.strs": ("a", "b"),
            "object.weird": {"not": "primitive"},
        },
    )
    captured = dict(span.calls)
    assert captured["primitive.str"] == "x"
    assert captured["primitive.int"] == 7
    assert captured["primitive.bool"] is True
    assert captured["primitive.float"] == 1.5
    assert "primitive.none" not in captured  # None silently dropped
    assert list(captured["list.ints"]) == [1, 2, 3]
    assert list(captured["tuple.strs"]) == ["a", "b"]
    # dict isn't a valid OTel attribute type, so it's stringified.
    assert isinstance(captured["object.weird"], str)


def test_init_otel_swallows_sdk_constructor_errors(monkeypatch: pytest.MonkeyPatch):
    """If the SDK is installed but TracerProvider blows up for some
    reason (bad env var, etc.), init_otel must log + return False, not
    raise into the caller's startup path."""
    pytest.importorskip("opentelemetry.sdk")

    _reset_otel_state(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:14318")

    # Sabotage `Resource.create` to raise — simulates a bad
    # OTEL_RESOURCE_ATTRIBUTES env var (the real failure mode).
    import opentelemetry.sdk.resources as _r

    def _boom(*a, **kw):
        raise RuntimeError("simulated bad resource attribute")

    monkeypatch.setattr(_r.Resource, "create", classmethod(_boom))

    assert obs.init_otel() is False
    assert obs.is_active() is False
