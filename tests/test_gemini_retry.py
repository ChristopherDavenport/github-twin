"""Retry behavior for transient Gemini errors.

`with_retry` retries 429s and 5xx with exponential backoff + jitter, and
honors `Retry-After` when the server sends one. Auth errors and 400-class
errors raise immediately. Tests drive the backoff via an injected `sleep`
so they run instantly.
"""

from __future__ import annotations

import pytest
from google.genai import errors

from github_twin.gemini_client import with_retry


class _FakeHeaders(dict):
    pass


class _FakeResponse:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = _FakeHeaders(headers or {})


def _make_client_error(code: int, headers: dict[str, str] | None = None) -> errors.ClientError:
    response_json = {"error": {"code": code, "message": "test", "status": "RESOURCE_EXHAUSTED"}}
    response = _FakeResponse(headers=headers)
    exc = errors.ClientError(code, response_json, response)  # type: ignore[arg-type]
    return exc


def _make_server_error(code: int = 503) -> errors.ServerError:
    response_json = {"error": {"code": code, "message": "test", "status": "UNAVAILABLE"}}
    return errors.ServerError(code, response_json, _FakeResponse())  # type: ignore[arg-type]


def test_succeeds_first_try_without_sleeping():
    sleeps: list[float] = []
    result = with_retry(lambda: "ok", sleep=sleeps.append)
    assert result == "ok"
    assert sleeps == []


def test_retries_429_and_eventually_succeeds():
    sleeps: list[float] = []
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _make_client_error(429)
        return "ok"

    result = with_retry(fn, base_delay=1.0, sleep=sleeps.append)
    assert result == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2  # two retries before success


def test_retries_5xx_and_eventually_succeeds():
    sleeps: list[float] = []
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _make_server_error(503)
        return "ok"

    result = with_retry(fn, base_delay=1.0, sleep=sleeps.append)
    assert result == "ok"
    assert calls["n"] == 2
    assert len(sleeps) == 1


def test_does_not_retry_non_transient_4xx():
    sleeps: list[float] = []
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise _make_client_error(403)

    with pytest.raises(errors.ClientError):
        with_retry(fn, sleep=sleeps.append)
    assert calls["n"] == 1
    assert sleeps == []


def test_raises_after_exhausting_attempts():
    sleeps: list[float] = []
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise _make_client_error(429)

    with pytest.raises(errors.ClientError):
        with_retry(fn, max_attempts=3, base_delay=1.0, sleep=sleeps.append)
    assert calls["n"] == 3
    assert len(sleeps) == 2  # slept between attempts, not after the last


def test_honors_retry_after_header_when_larger_than_backoff():
    sleeps: list[float] = []
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _make_client_error(429, headers={"Retry-After": "30"})
        return "ok"

    # base_delay=1.0 + jitter would give <= 1.5s; Retry-After=30 must win.
    with_retry(fn, base_delay=1.0, sleep=sleeps.append)
    assert sleeps[0] >= 30.0


def test_passes_through_args_and_kwargs():
    seen: dict[str, object] = {}

    def fn(a: int, *, b: str) -> str:
        seen["a"] = a
        seen["b"] = b
        return f"{a}/{b}"

    result = with_retry(fn, 7, b="x", sleep=lambda _s: None)
    assert result == "7/x"
    assert seen == {"a": 7, "b": "x"}
