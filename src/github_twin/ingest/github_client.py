"""Thin httpx wrapper for the GitHub REST API.

- Auth precedence: persisted device-flow token (from `gt auth login`)
  → `gh auth token` → `GITHUB_TOKEN` env.
- Rate limits: respects `Retry-After` and the secondary `X-RateLimit-Reset` header.
- Pagination: yields all pages via the `Link: rel="next"` header.
- Conditional requests (`get_json_cached` / `paginate_cached`): when an
  `HttpCache` is injected, sends `If-None-Match` / `If-Modified-Since`
  and reuses the cached body on a 304. Per GitHub docs, 304 responses
  to authorized conditional requests don't count against the primary
  rate limit, so a populated cache is effectively free for unchanged
  resources. Existing `request` / `get_json` / `paginate` are
  unchanged so non-cached callers stay byte-identical.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import httpx

from github_twin.ingest.http_cache import HttpCache, NoopHttpCache

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
    def __init__(
        self,
        token: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        cache: HttpCache | None = None,
    ):
        self._token = token or _resolve_token()
        # Pool sized for parallel per-repo workers. Default httpx limits
        # (max_connections=100, max_keepalive_connections=20) are fine for
        # most call sites, but we set them explicitly so the bound is
        # visible at the seam and easy to tune.
        self._client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": USER_AGENT,
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        self._cache: HttpCache = cache or NoopHttpCache()

    @property
    def token(self) -> str:
        """The resolved token, for callers that need to pass it to a
        subprocess (e.g. git clone) without re-running `_resolve_token`."""
        return self._token

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

    # ---------- Conditional-request variants ----------

    def _request_conditional(
        self, path: str, *, params: dict[str, Any] | None
    ) -> tuple[bytes, dict[str, str], str]:
        """Issue a GET with `If-None-Match` / `If-Modified-Since` headers
        when the cache has prior validators for this URL, and persist
        the response on a 200.

        Returns `(body_bytes, response_headers, canonical_url)` so the
        caller can re-parse without re-fetching. On 304, body comes from
        the cache; on 200, body is fresh.

        `canonical_url` is `str(resp.request.url)` — what httpx
        assembled after applying `params`. It's the same string we use
        as the cache key, so callers can pass it to `_LINK_NEXT`'s
        next-page resolver without worrying about query encoding.
        """
        # Build the canonical URL once via httpx, so the cache lookup
        # matches what the request layer will actually send.
        url = path if path.startswith("http") else f"{GITHUB_API}{path}"
        request = self._client.build_request("GET", url, params=params)
        canonical_url = str(request.url)

        cached = self._cache.get(canonical_url)
        cond_headers: dict[str, str] = {}
        if cached is not None:
            if cached.etag:
                cond_headers["If-None-Match"] = cached.etag
            if cached.last_modified:
                cond_headers["If-Modified-Since"] = cached.last_modified

        # `request` builds its own httpx.Request, so passing the same
        # url+params (no pre-built Request) keeps its retry semantics.
        for attempt in range(5):
            resp = self._client.request(
                "GET",
                url,
                params=params,
                headers=cond_headers or None,
            )
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
            break

        if resp.status_code == 304 and cached is not None:
            return cached.body, dict(resp.headers), canonical_url
        if resp.status_code >= 400:
            raise GitHubError(f"GET {canonical_url} -> {resp.status_code}: {resp.text[:200]}")
        # Persist fresh response. Skip storage on empty body — nothing
        # to round-trip on a future 304.
        body = resp.content
        if body:
            self._cache.put(
                canonical_url,
                etag=resp.headers.get("ETag"),
                last_modified=resp.headers.get("Last-Modified"),
                body=body,
            )
        return body, dict(resp.headers), canonical_url

    def get_json_cached(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """Conditional-GET variant of `get_json`. Returns parsed JSON,
        sourced either from a 200 (fresh) or from the cached body on a
        304. Falls through to a non-conditional fetch when the cache is
        empty for this URL."""
        body, _headers, _url = self._request_conditional(path, params=params)
        return json.loads(body) if body else None

    def paginate_cached(self, path: str, *, params: dict[str, Any] | None = None) -> Iterator[Any]:
        """Conditional-GET variant of `paginate`.

        We cache and check ETag only against page 1 (the canonical
        first-page URL with query string included). When page 1 returns
        304 we short-circuit the entire pagination loop and replay the
        cached body. This is sound ONLY for endpoints called with
        `sort=updated&direction=desc&since=...`: any new row would push
        the newest result onto page 1, so an unchanged page 1 means
        nothing new in the whole result set. Don't generalize this
        helper to other shapes without re-checking that invariant.

        Subsequent pages bypass the cache and use the existing
        non-conditional `request` path — we don't want to manage
        per-page ETag rotation here.
        """
        body, headers, _canonical = self._request_conditional(path, params=params)
        first_data = json.loads(body) if body else []
        first_items = (
            first_data["items"]
            if isinstance(first_data, dict) and "items" in first_data
            else first_data
        )
        yield from first_items

        link = headers.get("Link") or headers.get("link") or ""
        m = _LINK_NEXT.search(link)
        next_url: str | None = m.group(1) if m else None
        while next_url:
            resp = self.request("GET", next_url)
            if resp.status_code >= 400:
                raise GitHubError(f"GET {next_url} -> {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            items = data["items"] if isinstance(data, dict) and "items" in data else data
            yield from items
            link = resp.headers.get("Link", "")
            m = _LINK_NEXT.search(link)
            next_url = m.group(1) if m else None
