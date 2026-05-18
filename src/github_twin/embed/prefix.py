"""Embed-time chunk prefix (contextual retrieval).

Builds a short, deterministic header from `chunk.context` and prepends
it to `chunk.text` *before* the embedder sees it. The point is to give
the vector index per-chunk identity information (file path, symbol
name, node kind, leading docstring/comment) that the raw chunk body
doesn't always express. Built using metadata our AST chunker already
produces — no LLM calls, no extra index-time cost beyond a re-embed.

Headers are not stored in `chunk.text`, so BM25 (which indexes
`chunk.text` via the external-content FTS5 table) is unchanged and
re-running embed deterministically re-derives them.

Bump `EMBED_TEXT_VERSION` in `pipeline.py` whenever this file changes
the header in a way that would shift vectors; the pipeline detects the
bump and re-embeds the whole corpus on the next `gt embed`.
"""

from __future__ import annotations

from typing import Any

from github_twin.store import queries as q

# `pr_summary` text already starts with the PR title; a header would
# duplicate context with no embedding benefit. `rule` texts are
# LLM-distilled NL summaries — already context-rich. Both opt out.
_NO_PREFIX_KINDS = frozenset({"pr_summary", "rule"})


def build_header(chunk: q.ChunkRow) -> str:
    """Return a header string (with trailing blank line) for `chunk`,
    or `""` if no header applies. Keep this stable: any change here
    should come with a bump to EMBED_TEXT_VERSION."""
    if chunk.kind in _NO_PREFIX_KINDS:
        return ""
    ctx = chunk.context or {}
    if chunk.kind in ("code", "code_rule", "file"):
        return _code_header(ctx, summary=chunk.summary)
    if chunk.kind == "commit_message":
        base = _commit_message_header(ctx)
        if not base:
            # Without a base header there's nowhere natural to attach the
            # summary line; skip rather than emit a header-less prefix.
            return ""
        # Inject the summary line between the commit reference and the
        # body for a consistent "# header / # summary / blank / body" shape.
        repo_line, _, _ = base.partition("\n")
        sum_line = _commit_message_summary_line(chunk.summary)
        return f"{repo_line}\n{sum_line}\n" if sum_line else base
    if chunk.kind == "review_comment":
        return _review_comment_header(ctx)
    if chunk.kind == "note":
        return _note_header(ctx)
    return ""


def _code_header(ctx: dict[str, Any], summary: str | None = None) -> str:
    path = ctx.get("path")
    symbol = ctx.get("symbol_name")
    node_kind = ctx.get("node_kind")
    leading_doc = ctx.get("leading_doc")
    lines: list[str] = []
    if symbol and node_kind:
        # AST chunk: rich header, plus optional doc + LLM summary.
        head = f"# {path or '<unknown>'} :: {symbol} ({node_kind})"
        lines.append(head)
        if summary:
            lines.append(f"# {summary}")
        if leading_doc:
            lines.append(f"# {leading_doc}")
    elif path:
        # Line-window fallback: path-only. Cheap context, still useful for
        # path-aware queries ("where do we open the sqlite connection").
        lines.append(f"# {path}")
        if summary:
            lines.append(f"# {summary}")
    elif summary:
        # No path metadata at all but a summary exists — still worth emitting.
        lines.append(f"# {summary}")
    else:
        return ""
    return "\n".join(lines) + "\n\n"


def _commit_message_summary_line(summary: str | None) -> str:
    if not summary:
        return ""
    return f"# {summary}\n"


def _commit_message_header(ctx: dict[str, Any]) -> str:
    repo = ctx.get("repo")
    sha = ctx.get("commit_sha")
    if not repo and not sha:
        return ""
    short = sha[:7] if sha else "?"
    return f"# commit {repo or '?'}@{short}\n\n"


def _note_header(ctx: dict[str, Any]) -> str:
    """Scratch-note prefix: `# note: {title}` (or path fallback). Lets
    NL queries land on note chunks by topic rather than requiring a
    keyword inside the body."""
    title = ctx.get("title") or ctx.get("path") or ""
    if not title:
        return ""
    return f"# note: {title}\n\n"


def _review_comment_header(ctx: dict[str, Any]) -> str:
    repo = ctx.get("repo")
    pr_number = ctx.get("pr_number")
    path = ctx.get("path")
    # Many review comments are PR-level (no path). Fall through to a
    # repo + PR header in that case.
    parts: list[str] = []
    if repo:
        parts.append(repo)
    if pr_number is not None:
        parts.append(f"#{pr_number}")
    if not parts:
        # We have neither — skip the header entirely rather than emit a
        # useless "# review" line.
        return ""
    head = "# review on " + " ".join(parts)
    if path:
        head += f" ({path})"
    return head + "\n\n"


def prefix_chunk(chunk: q.ChunkRow) -> str:
    """Combine the header (if any) with chunk.text."""
    return build_header(chunk) + chunk.text
