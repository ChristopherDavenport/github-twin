"""Thin httpx wrapper for the GitHub REST API.

- Auth precedence: persisted device-flow token (from `gt auth login`)
  → `gh auth token` → `GITHUB_TOKEN` env.
- Rate limits: respects `Retry-After` and the secondary `X-RateLimit-Reset` header.
- Pagination: yields all pages via the `Link: rel="next"` header.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import httpx

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_TIMEOUT = 30.0
USER_AGENT = "github-twin/0.1"


class GitHubError(RuntimeError):
    pass


def _gh_cli_token() -> str | None:
    if shutil.which("gh") is None:
        return None
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if out.returncode != 0:
        return None
    tok = out.stdout.strip()
    return tok or None


def _stored_oauth_token() -> str | None:
    # Imported lazily so that test modules that monkeypatch `auth_storage`
    # pick up their patched module rather than a cached reference.
    from github_twin.ingest import auth_storage

    tok = auth_storage.load_token()
    if tok:
        # Defense in depth: a freshly-loaded token may end up in a log
        # line through some third-party traceback. Register its literal
        # value with the redacting filter so it gets scrubbed before
        # any handler renders it.
        from github_twin._logging import register_secret_value

        register_secret_value(tok)
    return tok


def _resolve_token() -> str:
    tok = _stored_oauth_token() or _gh_cli_token() or os.environ.get("GITHUB_TOKEN")
    if not tok:
        raise GitHubError(
            "No GitHub auth. Run `gt auth login`, or `gh auth login`, or set GITHUB_TOKEN."
        )
    return tok


_LINK_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


class GitHubClient:
    def __init__(self, token: str | None = None, *, timeout: float = DEFAULT_TIMEOUT):
        self._token = token or _resolve_token()
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": USER_AGENT,
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *a: object) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> httpx.Response:
        url = path if path.startswith("http") else f"{GITHUB_API}{path}"
        headers: dict[str, str] = {}
        if accept:
            headers["Accept"] = accept

        for attempt in range(5):
            resp = self._client.request(method, url, params=params, headers=headers)
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                self._sleep_until_reset(resp)
                continue
            if resp.status_code == 429:
                retry = float(resp.headers.get("Retry-After", "2"))
                log.warning("429 from %s, sleeping %.1fs", url, retry)
                time.sleep(retry)
                continue
            if resp.status_code >= 500 and attempt < 4:
                backoff = 1.5**attempt
                log.warning("%s on %s, retrying in %.1fs", resp.status_code, url, backoff)
                time.sleep(backoff)
                continue
            return resp
        return resp

    @staticmethod
    def _sleep_until_reset(resp: httpx.Response) -> None:
        reset = resp.headers.get("X-RateLimit-Reset")
        if reset and reset.isdigit():
            delay = max(1, int(reset) - int(time.time()) + 1)
        else:
            delay = int(float(resp.headers.get("Retry-After", "60")))
        log.warning("Rate-limited; sleeping %ds", delay)
        time.sleep(delay)

    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        resp = self.request("GET", path, params=params)
        if resp.status_code >= 400:
            raise GitHubError(f"GET {path} -> {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def get_text(self, path: str, *, params: dict[str, Any] | None = None, accept: str) -> str:
        resp = self.request("GET", path, params=params, accept=accept)
        if resp.status_code >= 400:
            raise GitHubError(f"GET {path} -> {resp.status_code}: {resp.text[:200]}")
        return resp.text

    def paginate(self, path: str, *, params: dict[str, Any] | None = None) -> Iterator[Any]:
        """Yield items across all pages. Works for both list endpoints and the
        `{total_count, items: [...]}` shape returned by /search/*."""
        url: str | None = path
        first = True
        while url:
            req_params = params if first else None
            resp = self.request("GET", url, params=req_params)
            if resp.status_code >= 400:
                raise GitHubError(f"GET {url} -> {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            items = data["items"] if isinstance(data, dict) and "items" in data else data
            yield from items
            link = resp.headers.get("Link", "")
            m = _LINK_NEXT.search(link)
            url = m.group(1) if m else None
            first = False
