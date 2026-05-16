"""Device-flow client (`github_twin.ingest.oauth`).

Covers the happy path and each of the four documented error codes
without sleeping in wall clock or hitting GitHub.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from github_twin.ingest import oauth


class _Resp:
    def __init__(self, status_code: int, body: dict[str, Any] | str) -> None:
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else ""

    def json(self) -> Any:
        if isinstance(self._body, str):
            raise ValueError("non-json")
        return self._body


class _HttpStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.responses: list[_Resp] = []


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> _HttpStub:
    """Capture every httpx.post call; return canned responses in order."""
    stub = _HttpStub()

    def fake_post(url: str, *, data: dict[str, str], headers: dict[str, str], timeout: float):
        stub.calls.append((url, data))
        if not stub.responses:
            raise AssertionError(f"unexpected POST to {url} (no canned response)")
        return stub.responses.pop(0)

    monkeypatch.setattr(oauth.httpx, "post", fake_post)
    return stub


def test_request_device_code_parses_response(http: _HttpStub) -> None:
    http.responses.append(
        _Resp(
            200,
            {
                "device_code": "DC",
                "user_code": "WDJB-MJHT",
                "verification_uri": "https://github.com/login/device",
                "verification_uri_complete": "https://github.com/login/device?user_code=WDJB-MJHT",
                "expires_in": 900,
                "interval": 5,
            },
        )
    )
    resp = oauth.request_device_code("CLIENT", "repo")
    assert resp.device_code == "DC"
    assert resp.user_code == "WDJB-MJHT"
    assert resp.verification_uri == "https://github.com/login/device"
    assert resp.verification_uri_complete is not None
    assert resp.expires_in == 900
    assert resp.interval == 5
    assert http.calls[0][0] == oauth.GITHUB_DEVICE_CODE_URL
    assert http.calls[0][1] == {"client_id": "CLIENT", "scope": "repo"}


def test_poll_for_token_returns_access_token_after_pending(http: _HttpStub) -> None:
    http.responses.extend(
        [
            _Resp(200, {"error": "authorization_pending"}),
            _Resp(200, {"error": "authorization_pending"}),
            _Resp(200, {"access_token": "gho_realtoken", "token_type": "bearer"}),
        ]
    )
    sleeps: list[float] = []
    tok = oauth.poll_for_token(
        "CLIENT",
        "DC",
        interval=5,
        expires_in=900,
        sleep=sleeps.append,
    )
    assert tok == "gho_realtoken"
    assert sleeps == [5.0, 5.0, 5.0]
    assert all(call[0] == oauth.GITHUB_ACCESS_TOKEN_URL for call in http.calls)
    assert http.calls[0][1]["grant_type"] == oauth.DEVICE_GRANT_TYPE


def test_slow_down_widens_interval_by_five_seconds(http: _HttpStub) -> None:
    http.responses.extend(
        [
            _Resp(200, {"error": "slow_down"}),
            _Resp(200, {"error": "authorization_pending"}),
            _Resp(200, {"access_token": "T"}),
        ]
    )
    sleeps: list[float] = []
    tok = oauth.poll_for_token("C", "DC", interval=5, expires_in=900, sleep=sleeps.append)
    assert tok == "T"
    # First sleep used original interval; after slow_down each subsequent
    # sleep uses interval + 5.
    assert sleeps == [5.0, 10.0, 10.0]


def test_expired_token_raises(http: _HttpStub) -> None:
    http.responses.append(_Resp(200, {"error": "expired_token"}))
    with pytest.raises(oauth.OAuthError, match="expired"):
        oauth.poll_for_token("C", "DC", interval=1, expires_in=10, sleep=lambda _s: None)


def test_access_denied_raises(http: _HttpStub) -> None:
    http.responses.append(_Resp(200, {"error": "access_denied"}))
    with pytest.raises(oauth.OAuthError, match="denied"):
        oauth.poll_for_token("C", "DC", interval=1, expires_in=10, sleep=lambda _s: None)


def test_unexpected_response_raises(http: _HttpStub) -> None:
    http.responses.append(_Resp(200, {"error": "weird_thing"}))
    with pytest.raises(oauth.OAuthError, match="weird_thing"):
        oauth.poll_for_token("C", "DC", interval=1, expires_in=10, sleep=lambda _s: None)


def test_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, *, data: Any, headers: Any, timeout: Any) -> _Resp:
        return _Resp(500, "internal server error")

    monkeypatch.setattr(oauth.httpx, "post", fake_post)
    with pytest.raises(oauth.OAuthError, match="HTTP 500"):
        oauth.request_device_code("C", "repo")


def test_poll_respects_expires_in_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """If wall time exceeds the deadline mid-poll, we surface a clean expired error."""
    calls = {"n": 0}

    def fake_post(url: str, *, data: Any, headers: Any, timeout: Any) -> _Resp:
        calls["n"] += 1
        return _Resp(200, {"error": "authorization_pending"})

    monkeypatch.setattr(oauth.httpx, "post", fake_post)

    # Advance a fake monotonic clock past the deadline after the first sleep.
    fake_clock = iter([0.0, 0.0, 9999.0, 9999.0, 9999.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(fake_clock))

    with pytest.raises(oauth.OAuthError, match="expired"):
        oauth.poll_for_token("C", "DC", interval=1, expires_in=10, sleep=lambda _s: None)
