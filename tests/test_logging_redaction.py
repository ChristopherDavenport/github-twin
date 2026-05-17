"""Secret-redacting log filter.

The audit established that github-twin's own code never logs tokens.
This filter is defense-in-depth for transitive risk: third-party SDKs
(httpx/httpcore in particular) log `Authorization: Bearer ...` at
DEBUG, and a `--verbose` invocation lifts everything to DEBUG. The
filter scrubs known patterns + the literal env-var values so no token
ever leaves the process via the log pipeline.
"""

from __future__ import annotations

import io
import logging

import pytest

from github_twin._logging import (
    SecretRedactingFilter,
    cap_noisy_loggers,
    install_secret_redaction,
)


def _capture_logger(filter_: logging.Filter) -> tuple[logging.Logger, io.StringIO]:
    """Build an isolated logger + stream handler with the given filter
    attached. Returns (logger, captured-output) so tests can assert on
    the rendered output."""
    name = f"test.{id(filter_):x}"  # unique per test
    log = logging.getLogger(name)
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    log.propagate = False  # don't bleed into root
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(filter_)
    log.addHandler(handler)
    return log, buf


# ---------- header-shaped scrubbing ----------


def test_bearer_token_in_message_is_redacted():
    flt = SecretRedactingFilter()
    log, buf = _capture_logger(flt)
    log.info("Authorization: Bearer ghp_supersecret_xxxxxxxxxxxx")
    out = buf.getvalue()
    assert "ghp_supersecret" not in out
    assert "Bearer ***" in out


def test_authorization_header_in_dict_repr_is_redacted():
    """httpx logs request headers as dict reprs at DEBUG. The shape is
    `{'authorization': 'Bearer foo', ...}` or with the value bare."""
    flt = SecretRedactingFilter()
    log, buf = _capture_logger(flt)
    log.debug("request headers: {'authorization': 'Bearer ghp_value12345'}")
    out = buf.getvalue()
    assert "ghp_value12345" not in out
    assert "Bearer ***" in out


def test_x_api_key_header_is_redacted():
    flt = SecretRedactingFilter()
    log, buf = _capture_logger(flt)
    log.debug("x-api-key: sk-ant-api03-fake_value_xxxxxxxxxxxx")
    out = buf.getvalue()
    assert "sk-ant-api03-fake_value" not in out
    assert "x-api-key: ***" in out.lower()


# ---------- env-value scrubbing ----------


def test_exact_env_value_is_redacted(monkeypatch: pytest.MonkeyPatch):
    """If GITHUB_TOKEN's literal value appears anywhere in a log line —
    via a format arg, a third-party traceback string, whatever — it's
    replaced with the REDACTED sentinel."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_AAAABBBBCCCCDDDD")
    flt = SecretRedactingFilter()
    log, buf = _capture_logger(flt)
    log.warning("upstream rejected our token: 'ghp_AAAABBBBCCCCDDDD'")
    out = buf.getvalue()
    assert "ghp_AAAABBBBCCCCDDDD" not in out
    assert "***REDACTED***" in out


def test_short_env_value_not_treated_as_secret(monkeypatch: pytest.MonkeyPatch):
    """A 4-char placeholder isn't a real token; the filter shouldn't
    scrub it (and risk hiding legitimate occurrences of short strings)."""
    monkeypatch.setenv("GITHUB_TOKEN", "xxx")  # under the length floor
    flt = SecretRedactingFilter()
    log, buf = _capture_logger(flt)
    log.warning("token xxx reported as invalid")
    assert "xxx" in buf.getvalue()


def test_anthropic_and_gemini_keys_scrubbed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-value-xxxxxxxxx")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFakeGoogleKey1234567")
    flt = SecretRedactingFilter()
    log, buf = _capture_logger(flt)
    log.info("Claude error with sk-ant-fake-value-xxxxxxxxx")
    log.info("Gemini error with AIzaFakeGoogleKey1234567")
    out = buf.getvalue()
    assert "sk-ant-fake-value" not in out
    assert "AIzaFakeGoogleKey" not in out
    assert out.count("***REDACTED***") == 2


def test_gemini_project_env_not_treated_as_secret(monkeypatch: pytest.MonkeyPatch):
    """`GT_GEMINI_PROJECT` carries a GCP project ID — visible context, not a
    credential. ADC tokens live on disk, never in env. Guard against a future
    well-meaning addition to _SECRET_ENV_VARS hiding diagnostic context."""
    monkeypatch.setenv("GT_GEMINI_PROJECT", "my-research-project-1234")
    flt = SecretRedactingFilter()
    log, buf = _capture_logger(flt)
    log.info("using Gemini project my-research-project-1234")
    out = buf.getvalue()
    assert "my-research-project-1234" in out
    assert "REDACTED" not in out


# ---------- format-args handling ----------


def test_args_substituted_into_message_are_scrubbed():
    """If the secret arrives via %-args, getMessage() must produce the
    formatted string AND the filter must clear `args` so handlers don't
    re-format and undo the scrub."""
    flt = SecretRedactingFilter()
    log, buf = _capture_logger(flt)
    log.warning("Trying %s with %s", "Bearer ghp_fakeXXXXXXXXXXX", "fallback")
    out = buf.getvalue()
    assert "ghp_fakeXXXXXXXXXXX" not in out
    assert "Bearer ***" in out
    assert "fallback" in out


# ---------- non-secret messages untouched ----------


def test_innocuous_message_passes_through():
    flt = SecretRedactingFilter()
    log, buf = _capture_logger(flt)
    log.info("ingested 42 commits from owner/repo")
    out = buf.getvalue().strip()
    assert out == "ingested 42 commits from owner/repo"


# ---------- install_secret_redaction is idempotent ----------


def test_install_is_idempotent():
    """Repeated install calls don't stack filters (would cause double
    redaction in pathological cases and slow logging)."""
    first = install_secret_redaction()
    second = install_secret_redaction()
    assert first is second
    root = logging.getLogger()
    n_filters = sum(isinstance(f, SecretRedactingFilter) for f in root.filters)
    assert n_filters == 1


# ---------- third-party logger capping ----------


def test_cap_noisy_loggers_silences_httpx_debug():
    cap_noisy_loggers()
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("anthropic").getEffectiveLevel() >= logging.WARNING


def test_filter_swallows_message_formatting_errors():
    """Buggy log calls (`log.info("got %s and %s", 1)` — too few args)
    raise inside `getMessage()`. The filter must let those records
    through; downstream is the standard logging error path. We must
    never block a record from being emitted."""
    flt = SecretRedactingFilter()
    log, buf = _capture_logger(flt)
    log.info("got %s and %s", 1)  # intentionally malformed
    # No exception raised through us; the standard logging error
    # surface kicks in at the handler. Buf may contain a traceback —
    # we just need the filter to have returned cleanly.
    # Sanity: the filter accepted the record (didn't drop it).
    assert (
        flt.filter(
            logging.LogRecord(
                "x",
                logging.INFO,
                __file__,
                0,
                "got %s and %s",
                (1,),
                None,
            )
        )
        is True
    )
