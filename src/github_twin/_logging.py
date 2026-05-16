"""Logging hardening: redact known secret patterns before records hit handlers.

The audit confirmed our own log statements never interpolate tokens or API
keys. The risk is structural: httpx / httpcore log full request headers
(including `Authorization: Bearer ...`) at DEBUG, and a user passing
`--verbose` lifts everything to DEBUG. We cap those loggers at WARNING
on top of that, but the filter below is defense in depth — it catches
anything we missed (third-party SDKs, future log statements, traceback
strings that include header dicts, etc.).

The filter runs on the root logger so every record propagating up gets
scrubbed. Mutating `record.msg` + clearing `record.args` ensures handlers
emit the scrubbed text; otherwise %-formatting would re-introduce the
secret from `args`.
"""

from __future__ import annotations

import logging
import os
import re

# Loggers known to log full request/response data at DEBUG. We don't
# need their DEBUG output for normal operation; cap at WARNING so
# `--verbose` users don't surface auth headers, response bodies, etc.
# Each name is matched as a prefix by the Python logging hierarchy.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "anthropic",
    "google",
    "google_genai",
    "openai",  # not a dep today but cheap to pre-emptively cap
    "urllib3",
)


# Header-shaped patterns that look like secret-bearing log lines.
# Kept narrow on purpose: each pattern matches exactly one substitution
# shape so they can't stack into nested redactions (`***`...`***`...).
# Tokens that appear via env vars are caught by the literal-value pass
# below, so the patterns here only need to cover transitive cases
# (httpx/httpcore dumping headers, third-party SDK errors).
_HEADER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # `Bearer <token>` anywhere — covers HTTP Authorization headers
    # whether the leading "Authorization:" is present or not.
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-+/=]+"), "Bearer ***"),
    # API-key headers (Anthropic, OpenAI, generic SaaS).
    (re.compile(r"(?i)(x-api-key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-+/=]+"), r"\1***"),
)


_REDACT = "***REDACTED***"


# Env vars whose values are secrets. If any of these are set at process
# start, the filter compiles a literal-text matcher so the exact value
# never appears in logs, even via traceback / repr / third-party paths.
_SECRET_ENV_VARS = (
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


class SecretRedactingFilter(logging.Filter):
    """Mutates outbound log records to strip secret-shaped substrings.

    The filter sees records *before* handlers format them. We rewrite
    `record.msg` to the scrubbed text and clear `record.args` so the
    standard %-formatter doesn't re-introduce a secret from the args
    tuple. This costs one `getMessage()` + regex pass per record, only
    on records that contain a pattern match (the env-value path
    short-circuits when nothing matches)."""

    def __init__(self) -> None:
        super().__init__()
        # Snapshot env values at construction. Re-importing won't pick up
        # later env mutations, which is what we want — a `del
        # os.environ['GITHUB_TOKEN']` after init doesn't reset our matcher.
        self._env_values: list[str] = []
        for name in _SECRET_ENV_VARS:
            val = os.environ.get(name, "").strip()
            # Skip stubs / empty / clearly-not-a-secret values. 12 chars
            # filters out the shortest plausible token and avoids
            # accidentally scrubbing common short strings.
            if val and len(val) >= 12:
                self._env_values.append(val)

    def add_secret(self, value: str) -> None:
        """Register an additional literal secret value to scrub.

        Used by `_stored_oauth_token` to opt the persisted OAuth token
        into the same scrubbing pipeline as env-derived secrets. No-op
        for short/empty values (same 12-char floor as env values)."""
        value = value.strip()
        if value and len(value) >= 12 and value not in self._env_values:
            self._env_values.append(value)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — never let logging fail downstream
            return True
        scrubbed = self._scrub(msg)
        if scrubbed is not msg:
            record.msg = scrubbed
            record.args = None
        return True

    def _scrub(self, msg: str) -> str:
        original = msg
        for pat, repl in _HEADER_PATTERNS:
            msg = pat.sub(repl, msg)
        for val in self._env_values:
            if val in msg:
                msg = msg.replace(val, _REDACT)
        return msg if msg != original else original


def cap_noisy_loggers(level: int = logging.WARNING) -> None:
    """Apply the WARNING cap to loggers we know log secrets at DEBUG.
    Safe to call multiple times — `setLevel` is idempotent."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)


def install_secret_redaction() -> SecretRedactingFilter:
    """Attach `SecretRedactingFilter` to the root logger if not already
    installed. Returns the filter instance (for tests). Idempotent."""
    root = logging.getLogger()
    for existing in root.filters:
        if isinstance(existing, SecretRedactingFilter):
            return existing
    flt = SecretRedactingFilter()
    root.addFilter(flt)
    # Adding the filter to the root logger covers records emitted by
    # root itself, but Python's logging routes through handler.filter()
    # on the handler that emits — and handlers attached to non-root
    # loggers won't see this filter. Mirror onto each existing root
    # handler so format() runs on the scrubbed text.
    for handler in root.handlers:
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(flt)
    return flt


def register_secret_value(value: str) -> None:
    """Add a literal secret to the active filter, if one is installed.

    Lazy entry point for code paths that learn about secrets after
    process start (e.g. loading a persisted OAuth token). Quietly
    no-ops when the filter hasn't been installed yet — that path only
    happens in tests that bypass `_setup_logging`."""
    root = logging.getLogger()
    for existing in root.filters:
        if isinstance(existing, SecretRedactingFilter):
            existing.add_secret(value)
            return
