"""Embed-time chunk prefix (contextual retrieval).

These tests are pure-functional over `q.ChunkRow`; no embedder, no SQL.
The pipeline-level test (forced re-embed when EMBED_TEXT_VERSION moves)
lives in `test_pipeline_embed_version.py`.
"""

from __future__ import annotations

from github_twin.embed.prefix import build_header, prefix_chunk
from github_twin.store import queries as q


def _chunk(kind: str, text: str, ctx: dict | None, summary: str | None = None) -> q.ChunkRow:
    return q.ChunkRow(
        id=1,
        artifact_id=1,
        kind=kind,
        text=text,
        context=ctx or {},
        embed_model=None,
        summary=summary,
    )


# ---------- code with LLM summary ----------


def test_code_chunk_header_includes_summary_when_present():
    c = _chunk(
        "code",
        "def handle(req):\n    return _dispatch(req)",
        {
            "path": "src/router.py",
            "symbol_name": "handle",
            "node_kind": "function_definition",
            "leading_doc": "Validate auth headers and dispatch.",
        },
        summary="Routes HTTP requests to authorized handlers after header validation.",
    )
    out = prefix_chunk(c)
    # Summary line lands between the location header and the leading_doc.
    expected_head = (
        "# src/router.py :: handle (function_definition)\n"
        "# Routes HTTP requests to authorized handlers after header validation.\n"
        "# Validate auth headers and dispatch.\n\n"
    )
    assert out.startswith(expected_head)
    assert out.endswith("def handle(req):\n    return _dispatch(req)")


def test_code_chunk_summary_without_leading_doc():
    c = _chunk(
        "code",
        "def f(): pass",
        {"path": "x.py", "symbol_name": "f", "node_kind": "function_definition"},
        summary="A no-op stub.",
    )
    out = prefix_chunk(c)
    assert out == "# x.py :: f (function_definition)\n# A no-op stub.\n\ndef f(): pass"


def test_line_window_fallback_includes_summary():
    c = _chunk(
        "code",
        "some text",
        {"path": "data.yaml", "language": "yaml"},
        summary="Build matrix for CI jobs.",
    )
    out = prefix_chunk(c)
    assert out.startswith("# data.yaml\n# Build matrix for CI jobs.\n\n")


def test_commit_message_chunk_includes_summary():
    c = _chunk(
        "commit_message",
        "fix: handle empty input\n\nRoot cause: ...",
        {"repo": "me/x", "commit_sha": "abcdef1234567890"},
        summary="Fixes a NPE when the request body is empty.",
    )
    out = prefix_chunk(c)
    assert out.startswith(
        "# commit me/x@abcdef1\n# Fixes a NPE when the request body is empty.\n\n"
    )
    assert out.endswith("fix: handle empty input\n\nRoot cause: ...")


# ---------- code (AST: full header with leading_doc) ----------


def test_code_ast_chunk_header_includes_symbol_and_doc():
    c = _chunk(
        "code",
        "def handle(req):\n    return _dispatch(req)",
        {
            "path": "src/router.py",
            "symbol_name": "handle",
            "node_kind": "function_definition",
            "leading_doc": "Validate auth headers and dispatch.",
        },
    )
    out = prefix_chunk(c)
    assert out.startswith("# src/router.py :: handle (function_definition)\n")
    assert "# Validate auth headers and dispatch." in out
    assert out.endswith("def handle(req):\n    return _dispatch(req)")
    # Blank line between header and code.
    assert "\n\ndef handle" in out


def test_code_ast_chunk_no_leading_doc():
    c = _chunk(
        "code",
        "def f(): pass",
        {"path": "x.py", "symbol_name": "f", "node_kind": "function_definition"},
    )
    out = prefix_chunk(c)
    assert out.startswith("# x.py :: f (function_definition)\n\n")
    # Single header line, then blank, then code.
    lines = out.splitlines()
    assert lines[0] == "# x.py :: f (function_definition)"
    assert lines[1] == ""
    assert lines[2] == "def f(): pass"


def test_code_line_window_fallback_path_only_header():
    """When symbol_name is missing (line-window fallback chunks), header is
    just `# {path}` — enough for path-aware queries to retrieve."""
    c = _chunk(
        "code",
        "some text\nmore text",
        {"path": "data.yaml", "language": "yaml", "start_line": 10, "end_line": 50},
    )
    out = prefix_chunk(c)
    assert out.startswith("# data.yaml\n\n")
    assert "data.yaml :: " not in out


def test_code_without_path_skips_header():
    """No metadata at all → no synthetic header. Don't pollute the embed
    space with a useless `# <unknown>` placeholder."""
    c = _chunk("code", "raw", {})
    assert prefix_chunk(c) == "raw"


# ---------- file (file-at-HEAD; same shape as code) ----------


def test_file_chunk_uses_same_header_as_code():
    c = _chunk(
        "file",
        "func Handle() {}",
        {
            "path": "internal/server.go",
            "symbol_name": "Handle",
            "node_kind": "function_declaration",
            "leading_doc": "Handle does the thing.",
        },
    )
    out = prefix_chunk(c)
    assert "# internal/server.go :: Handle (function_declaration)" in out
    assert "Handle does the thing." in out


# ---------- code_rule (distilled patterns) ----------


def test_code_rule_chunk_uses_same_header_as_code():
    c = _chunk(
        "code_rule",
        "Use `Resource.eval` for effectful initialization.",
        {"symbol_name": "resource_eval_pattern", "node_kind": "rule"},
    )
    # symbol_name without path: path falls back to "<unknown>" so we still
    # emit a header (rules embed alongside code; the symbol info is the
    # useful bit).
    out = prefix_chunk(c)
    assert out.startswith("# <unknown> :: resource_eval_pattern (rule)")


# ---------- commit_message ----------


def test_commit_message_header():
    c = _chunk(
        "commit_message",
        "fix: handle empty input\n\nRoot cause: ...",
        {"repo": "me/x", "commit_sha": "abcdef1234567890", "source_url": "https://..."},
    )
    out = prefix_chunk(c)
    assert out.startswith("# commit me/x@abcdef1\n\n")
    assert out.endswith("fix: handle empty input\n\nRoot cause: ...")


def test_commit_message_without_metadata_skips_header():
    c = _chunk("commit_message", "msg", {})
    assert prefix_chunk(c) == "msg"


# ---------- review_comment ----------


def test_review_comment_header_with_path():
    c = _chunk(
        "review_comment",
        "Should we use Resource instead?",
        {"repo": "me/x", "pr_number": 42, "path": "src/Foo.scala", "pr_title": "Add Foo"},
    )
    out = prefix_chunk(c)
    assert out.startswith("# review on me/x #42 (src/Foo.scala)\n\n")
    assert out.endswith("Should we use Resource instead?")


def test_review_comment_header_without_path():
    """PR-level review comments (no file context) get the repo + PR header
    alone — no awkward `(None)` suffix."""
    c = _chunk(
        "review_comment",
        "LGTM",
        {"repo": "me/x", "pr_number": 7, "pr_title": "Title"},
    )
    out = prefix_chunk(c)
    assert out.startswith("# review on me/x #7\n\n")
    assert "None" not in out


# ---------- pr_summary, rule: opt out ----------


def test_pr_summary_chunk_is_unchanged():
    c = _chunk("pr_summary", "Some PR title\n\nbody body body", {"pr_number": 1})
    assert prefix_chunk(c) == "Some PR title\n\nbody body body"


def test_rule_chunk_is_unchanged():
    c = _chunk("rule", "Rules text here.", {"source_examples": []})
    assert prefix_chunk(c) == "Rules text here."


# ---------- build_header is callable on its own ----------


def test_build_header_returns_string_with_trailing_blank_line():
    c = _chunk("commit_message", "msg", {"repo": "me/x", "commit_sha": "deadbeefcafe"})
    h = build_header(c)
    assert h.endswith("\n\n")
    assert "deadbee" in h
