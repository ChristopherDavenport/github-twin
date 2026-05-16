"""GitHub OAuth Device Authorization Grant (RFC 8628).

The same flow `gh auth login` uses internally. Exchanges a public client
ID + user-typed code for an access token without ever needing a client
secret or a redirect URI.

Two-step:
  1. `request_device_code(client_id, scope)` — POST /login/device/code.
     Returns the user code to display + the verification URL.
  2. `poll_for_token(client_id, device_code, interval, expires_in)` —
     POST /login/oauth/access_token in a loop until the user
     authorizes in their browser or the window expires.

The poll loop sleeps via an injected `sleep` callable so tests can run
without wall-clock waits. Errors map to the four documented
device-flow error codes; anything else surfaces as `OAuthError`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

# Match GitHubClient's UA so server-side logs attribute requests to us.
USER_AGENT = "github-twin/0.1"


class OAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceCodeResponse:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None
    expires_in: int
    interval: int


def _post_json(url: str, data: dict[str, Any]) -> dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    resp = httpx.post(url, data=data, headers=headers, timeout=30.0)
    if resp.status_code >= 400:
        raise OAuthError(f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        body = resp.json()
    except ValueError as exc:
        raise OAuthError(f"{url} returned non-JSON body: {resp.text[:200]}") from exc
    if not isinstance(body, dict):
        raise OAuthError(f"{url} returned non-object JSON: {body!r}")
    return body


def request_device_code(client_id: str, scope: str) -> DeviceCodeResponse:
    """Start the device flow. Returns the user code + verification URL."""
    body = _post_json(GITHUB_DEVICE_CODE_URL, {"client_id": client_id, "scope": scope})
    try:
        return DeviceCodeResponse(
            device_code=str(body["device_code"]),
            user_code=str(body["user_code"]),
            verification_uri=str(body["verification_uri"]),
            verification_uri_complete=(
                str(body["verification_uri_complete"])
                if body.get("verification_uri_complete")
                else None
            ),
            expires_in=int(body["expires_in"]),
            interval=int(body["interval"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OAuthError(f"Malformed device-code response: {body!r}") from exc


def poll_for_token(
    client_id: str,
    device_code: str,
    *,
    interval: int,
    expires_in: int,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Poll until the user authorizes, the code expires, or they deny.

    Returns the access token on success. Raises `OAuthError` on
    `expired_token` / `access_denied` / unexpected response shapes.
    `slow_down` widens the polling interval by 5s per spec.
    """
    deadline = time.monotonic() + expires_in
    poll_interval = interval
    while True:
        sleep(poll_interval)
        if time.monotonic() > deadline:
            raise OAuthError("Device code expired before authorization. Re-run `gt auth login`.")
        body = _post_json(
            GITHUB_ACCESS_TOKEN_URL,
            {
                "client_id": client_id,
                "device_code": device_code,
                "grant_type": DEVICE_GRANT_TYPE,
            },
        )
        token = body.get("access_token")
        if token:
            return str(token)
        err = body.get("error")
        if err == "authorization_pending":
            continue
        if err == "slow_down":
            # RFC 8628 §3.5: server hints we should back off by 5s.
            poll_interval += 5
            continue
        if err == "expired_token":
            raise OAuthError("Device code expired before authorization. Re-run `gt auth login`.")
        if err == "access_denied":
            raise OAuthError("Authorization denied in browser.")
        raise OAuthError(f"Unexpected device-flow response: {body!r}")
