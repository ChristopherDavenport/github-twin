"""Chunkers for commits, review comments, and commit messages.

Code chunks are kept small (≤ 80 lines) so retrieval surfaces concrete snippets
rather than entire files. We split a unified diff into per-file hunks and only
keep the added lines (what *I* wrote / changed). Removed lines are noise for
style retrieval.

Review-comment chunks embed the comment body itself; the diff hunk is preserved
verbatim in `context` so the retriever can show the agent what I was reacting to.
"""

from __future__ import annotations

import fnmatch
import logging
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from tree_sitter import Node

from github_twin.process.grammars import LanguageGrammar, grammar_for_language
from github_twin.process.language import language_for_path

log = logging.getLogger(__name__)

MAX_CODE_CHUNK_LINES = 80
MIN_CODE_CHUNK_LINES = 3
MIN_COMMIT_MESSAGE_LEN = 20
# File-at-HEAD chunking: overlap successive windows by this many lines so a
# function or class doesn't get truncated across a window boundary.
FILE_CHUNK_OVERLAP = 10
# PR summary chunks (P3 predict_review_outcome) — embed title + first part
# of body. PR descriptions get long; 2000 chars captures intent without
# pushing the embedder anywhere near its context cap.
MAX_PR_BODY_CHARS = 2000
# AST-parse guards. Tree-sitter's TypeScript / TSX grammars can spend
# unbounded CPU on pathological inputs (generated, minified, or otherwise
# adversarial code) inside `ts_parser__do_all_potential_reductions`, and
# the Python binding holds the GIL across `Parser.parse`, so one bad file
# wedges every ingest worker. The size cap is the first line of defense
# (most blowups are giant generated files); the timeout is the backstop.
MAX_AST_PARSE_BYTES = 512 * 1024
AST_PARSE_TIMEOUT_MICROS = 5_000_000


def _set_parser_timeout(parser: Any, micros: int) -> None:
    # `Parser.timeout_micros` is the only timeout API that works for
    # bytestring sources in tree-sitter 0.25; the `progress_callback`
    # alternative segfaults on bytestrings. The setter is deprecated and
    # warns on every call — suppress it locally so we don't spam stderr
    # once per parse during a full ingest.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parser.timeout_micros = micros


@dataclass(frozen=True)
class CodeChunk:
    text: str
    language: str
    path: str
    context: dict[str, Any]


@dataclass(frozen=True)
class CommitMessageChunk:
    text: str
    context: dict[str, Any]


@dataclass(frozen=True)
class PRSummaryChunk:
    text: str
    context: dict[str, Any]


def is_excluded_path(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def _split_unified_diff(diff: str) -> Iterator[tuple[str, list[str]]]:
    """Yield (new_path, [body_lines]) for each `diff --git` section.

    new_path is the post-rename target (from the `+++ b/...` line). We skip
    binary diffs and file-deletion diffs.
    """
    current_path: str | None = None
    body: list[str] = []
    in_binary = False

    for raw_line in diff.splitlines():
        if raw_line.startswith("diff --git"):
            if current_path and body and not in_binary:
                yield current_path, body
            current_path = None
            body = []
            in_binary = False
            continue
        if raw_line.startswith("Binary files "):
            in_binary = True
            continue
        if raw_line.startswith("+++ "):
            # `+++ b/path` or `+++ /dev/null`
            tail = raw_line[4:].strip()
            current_path = None if tail == "/dev/null" else tail.removeprefix("b/")
            continue
        if raw_line.startswith("--- ") or raw_line.startswith("index "):
            continue
        if current_path is not None and not in_binary:
            body.append(raw_line)

    if current_path and body and not in_binary:
        yield current_path, body


def _added_blocks(hunk_body: list[str], max_lines: int) -> Iterator[list[str]]:
    """Yield runs of added (`+`) lines from a diff body, splitting on size."""
    block: list[str] = []
    for line in hunk_body:
        if line.startswith("+") and not line.startswith("+++"):
            block.append(line[1:])
            if len(block) >= max_lines:
                yield block
                block = []
        else:
            if len(block) >= MIN_CODE_CHUNK_LINES:
                yield block
            block = []
    if len(block) >= MIN_CODE_CHUNK_LINES:
        yield block


@dataclass(frozen=True)
class _DiffHunk:
    """One @@-block of a unified diff, reduced to post-image lines and the
    1-based line numbers (within `post_image`) that came from `+` lines.

    The post-image is parseable on its own only when the hunk happens to
    contain whole declarable units — common for adds and small reshapes,
    not so for deep edits inside a big function. The AST chunker tries to
    parse it and falls back to the line-block path when the parser yields
    no useful nodes.
    """

    path: str
    post_image: str
    added_lines: frozenset[int]


def _iter_diff_hunks(diff: str) -> Iterator[_DiffHunk]:
    """Yield one _DiffHunk per @@ block. Drops binary diffs and file-deletion
    diffs (target == /dev/null). Hunk headers in the body are dropped from
    the post-image."""
    current_path: str | None = None
    in_binary = False
    in_hunk = False
    post_image: list[str] = []
    added: set[int] = set()
    line_in_post = 0

    def flush() -> _DiffHunk | None:
        if not current_path or in_binary or not in_hunk:
            return None
        if not post_image:
            return None
        return _DiffHunk(
            path=current_path,
            post_image="\n".join(post_image),
            added_lines=frozenset(added),
        )

    for raw in diff.splitlines():
        if raw.startswith("diff --git"):
            h = flush()
            if h is not None:
                yield h
            current_path = None
            in_binary = False
            in_hunk = False
            post_image = []
            added = set()
            line_in_post = 0
            continue
        if raw.startswith("Binary files "):
            in_binary = True
            continue
        if raw.startswith("+++ "):
            tail = raw[4:].strip()
            current_path = None if tail == "/dev/null" else tail.removeprefix("b/")
            continue
        if raw.startswith("--- ") or raw.startswith("index "):
            continue
        if raw.startswith("@@"):
            h = flush()
            if h is not None:
                yield h
            in_hunk = True
            post_image = []
            added = set()
            line_in_post = 0
            continue
        if current_path is None or in_binary or not in_hunk:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            line_in_post += 1
            post_image.append(raw[1:])
            added.add(line_in_post)
        elif raw.startswith("-"):
            continue
        elif raw.startswith("\\"):
            # "\ No newline at end of file" — metadata, not content.
            continue
        else:
            # Context line (may start with " " or be entirely empty).
            line_in_post += 1
            post_image.append(raw[1:] if raw.startswith(" ") else raw)

    h = flush()
    if h is not None:
        yield h


def chunk_diff(
    diff: str,
    *,
    repo: str,
    sha: str,
    source_url: str | None,
    exclude_patterns: Iterable[str] = (),
) -> Iterator[CodeChunk]:
    """Walk a unified diff, yielding one code chunk per changed unit.

    For languages with a registered grammar, parses each hunk's post-image
    and emits AST nodes (function, class, ...) that overlap the added
    lines. Falls back to the original `_added_blocks` flow per-path when
    no grammar applies, or per-hunk when the parser yields no nodes
    intersecting the added region.
    """
    excludes = tuple(exclude_patterns)

    # Group hunks by path so the fallback decision can be made on a
    # per-file basis (mirroring the pre-AST contract: one fallback flow
    # per file, not per hunk).
    by_path: dict[str, list[_DiffHunk]] = {}
    order: list[str] = []
    for hunk in _iter_diff_hunks(diff):
        if excludes and is_excluded_path(hunk.path, excludes):
            continue
        if hunk.path not in by_path:
            by_path[hunk.path] = []
            order.append(hunk.path)
        by_path[hunk.path].append(hunk)

    for path in order:
        hunks = by_path[path]
        lang = language_for_path(path)
        if lang is None:
            continue
        grammar = grammar_for_language(lang)
        ast_chunks: list[CodeChunk] = []
        if grammar is not None:  # narrows for mypy
            for hunk in hunks:
                ast_chunks.extend(
                    _chunk_hunk_ast(
                        hunk,
                        grammar=grammar,
                        lang=lang,
                        repo=repo,
                        sha=sha,
                        source_url=source_url,
                    )
                )
        if ast_chunks:
            yield from ast_chunks
            continue
        # Fall back to the legacy line-block flow. The fallback consumes
        # the union of all hunk bodies for the file (matches pre-AST
        # behavior).
        body_lines: list[str] = []
        for hunk in hunks:
            for line_idx, text in enumerate(hunk.post_image.splitlines(), start=1):
                prefix = "+" if line_idx in hunk.added_lines else " "
                body_lines.append(prefix + text)
        for block in _added_blocks(body_lines, MAX_CODE_CHUNK_LINES):
            yield CodeChunk(
                text="\n".join(block),
                language=lang,
                path=path,
                context={
                    "repo": repo,
                    "path": path,
                    "language": lang,
                    "commit_sha": sha,
                    "source_url": source_url,
                },
            )


def _chunk_hunk_ast(
    hunk: _DiffHunk,
    *,
    grammar: LanguageGrammar,
    lang: str,
    repo: str,
    sha: str,
    source_url: str | None,
) -> Iterator[CodeChunk]:
    """Parse one hunk's post-image and yield CodeChunks for matched AST
    nodes that overlap the hunk's added lines.

    Yields nothing on parse failure or when no matched node intersects
    `hunk.added_lines` — callers handle the fallback.
    """
    if not hunk.added_lines:
        return
    try:
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language as _get_language
    except ImportError:
        log.warning("tree-sitter unavailable; skipping AST chunk_diff")
        return

    encoded = hunk.post_image.encode("utf-8", errors="replace")
    if len(encoded) > MAX_AST_PARSE_BYTES:
        log.warning(
            "tree-sitter skip (oversize hunk) path=%s bytes=%d > cap=%d",
            hunk.path,
            len(encoded),
            MAX_AST_PARSE_BYTES,
        )
        return

    try:
        ts_lang = _get_language(grammar.parser_name)
        parser = Parser(ts_lang)
        _set_parser_timeout(parser, AST_PARSE_TIMEOUT_MICROS)
        tree = parser.parse(encoded)
    except Exception as e:  # pragma: no cover - defensive against parser bugs
        log.warning("tree-sitter parse failed for hunk path=%s: %s", hunk.path, e)
        return
    if tree is None:
        log.warning(
            "tree-sitter timeout (>%dms) parsing hunk path=%s bytes=%d",
            AST_PARSE_TIMEOUT_MICROS // 1000,
            hunk.path,
            len(encoded),
        )
        return

    src_lines = hunk.post_image.splitlines()
    # Significant adds = non-blank `+` lines. Blank `+` lines appear
    # between new defs (e.g. the empty line between two added methods)
    # and shouldn't force the enclosing class/object to emit when the
    # actual content already shows up in its children.
    significant = frozenset(
        a for a in hunk.added_lines if a - 1 < len(src_lines) and src_lines[a - 1].strip()
    )
    if not significant:
        return

    emit_nodes: list[Node] = []

    def walk(node: Node) -> set[int]:
        """Returns the set of significant added lines claimed by chunkable
        nodes in this subtree (the current node included if it emits).
        A node emits only when it covers an added line that no chunkable
        descendant of it already claims — so a method change emits the
        method, not the class wrapping it; a class header change still
        emits the class because no descendant overlaps the header line."""
        claimed: set[int] = set()
        for child in node.children:
            claimed |= walk(child)
        if node.type in grammar.chunk_node_kinds and grammar.should_emit(node):
            sl = node.start_point[0] + 1
            el = node.end_point[0] + 1
            in_range = {a for a in significant if sl <= a <= el}
            if in_range - claimed:
                emit_nodes.append(node)
                return claimed | in_range
        return claimed

    walk(tree.root_node)
    if not emit_nodes:
        return

    seen: set[tuple[int, int]] = set()
    for node in sorted(emit_nodes, key=lambda n: (n.start_point[0], n.end_point[0])):
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        key = (start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        text = "\n".join(src_lines[start_line - 1 : end_line])
        if not text.strip():
            continue
        yield CodeChunk(
            text=text,
            language=lang,
            path=hunk.path,
            context={
                "repo": repo,
                "path": hunk.path,
                "language": lang,
                "commit_sha": sha,
                "source_url": source_url,
                "node_kind": node.type,
                "symbol_name": grammar.symbol_name(node),
                "leading_doc": grammar.leading_doc(node),
            },
        )


def chunk_file(
    content: str,
    *,
    repo: str,
    path: str,
    source_url: str | None = None,
    head_sha: str | None = None,
    exclude_patterns: Iterable[str] = (),
) -> Iterator[CodeChunk]:
    """Yield code chunks for a full-file source blob.

    For languages with a registered tree-sitter grammar, walks the AST and
    emits one chunk per declarable unit (function / method / class /
    decorated def, ...). Falls back to line-window chunking for unknown
    languages, parser failures, or files whose AST yields no chunkable
    nodes. Skips excluded paths and files shorter than MIN_CODE_CHUNK_LINES.
    """
    excludes = tuple(exclude_patterns)
    if excludes and is_excluded_path(path, excludes):
        return
    lang = language_for_path(path)
    if lang is None:
        return
    lines = content.splitlines()
    if len(lines) < MIN_CODE_CHUNK_LINES:
        return

    grammar = grammar_for_language(lang)
    if grammar is not None:
        emitted = False
        for chunk in _chunk_file_ast(
            content,
            grammar=grammar,
            lang=lang,
            repo=repo,
            path=path,
            source_url=source_url,
            head_sha=head_sha,
        ):
            emitted = True
            yield chunk
        if emitted:
            return
        # Parser produced no chunkable nodes (e.g., a file of only imports);
        # fall through to line-windows so we still index something.

    yield from _chunk_file_line_windows(
        lines,
        lang=lang,
        repo=repo,
        path=path,
        source_url=source_url,
        head_sha=head_sha,
    )


def _chunk_file_line_windows(
    lines: list[str],
    *,
    lang: str,
    repo: str,
    path: str,
    source_url: str | None,
    head_sha: str | None,
) -> Iterator[CodeChunk]:
    """Sliding-window chunker. Used as fallback when no AST grammar applies."""
    step = MAX_CODE_CHUNK_LINES - FILE_CHUNK_OVERLAP
    start = 0
    while start < len(lines):
        end = min(start + MAX_CODE_CHUNK_LINES, len(lines))
        window = lines[start:end]
        if len(window) >= MIN_CODE_CHUNK_LINES:
            yield CodeChunk(
                text="\n".join(window),
                language=lang,
                path=path,
                context={
                    "repo": repo,
                    "path": path,
                    "language": lang,
                    "start_line": start + 1,
                    "end_line": end,
                    "head_sha": head_sha,
                    "source_url": source_url,
                },
            )
        if end == len(lines):
            break
        start += step


def _chunk_file_ast(
    content: str,
    *,
    grammar: LanguageGrammar,
    lang: str,
    repo: str,
    path: str,
    source_url: str | None,
    head_sha: str | None,
) -> Iterator[CodeChunk]:
    """Parse `content` with `grammar`'s tree-sitter parser and yield one
    CodeChunk per matching AST node.

    Returns silently (yielding nothing) on parser failure — callers detect
    the empty stream and fall back to the line-window chunker.
    """
    try:
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language
    except ImportError:
        log.warning("tree-sitter unavailable; falling back for path=%s", path)
        return

    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) > MAX_AST_PARSE_BYTES:
        log.warning(
            "tree-sitter skip (oversize file) path=%s bytes=%d > cap=%d",
            path,
            len(encoded),
            MAX_AST_PARSE_BYTES,
        )
        return

    try:
        ts_lang = get_language(grammar.parser_name)
        parser = Parser(ts_lang)
        _set_parser_timeout(parser, AST_PARSE_TIMEOUT_MICROS)
        tree = parser.parse(encoded)
    except Exception as e:  # pragma: no cover - defensive against parser bugs
        log.warning("tree-sitter parse failed for path=%s: %s", path, e)
        return
    if tree is None:
        log.warning(
            "tree-sitter timeout (>%dms) parsing path=%s bytes=%d",
            AST_PARSE_TIMEOUT_MICROS // 1000,
            path,
            len(encoded),
        )
        return

    src_lines = content.splitlines()
    for node in _walk_chunk_nodes(tree.root_node, grammar):
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        text = "\n".join(src_lines[start_line - 1 : end_line])
        if not text.strip():
            continue
        symbol = grammar.symbol_name(node)
        yield CodeChunk(
            text=text,
            language=lang,
            path=path,
            context={
                "repo": repo,
                "path": path,
                "language": lang,
                "start_line": start_line,
                "end_line": end_line,
                "head_sha": head_sha,
                "source_url": source_url,
                "node_kind": node.type,
                "symbol_name": symbol,
                "leading_doc": grammar.leading_doc(node),
            },
        )


def _walk_chunk_nodes(root: Node, grammar: LanguageGrammar) -> Iterator[Node]:
    """Depth-first walk yielding nodes whose type is in
    `grammar.chunk_node_kinds` AND for which `grammar.should_emit(node)`
    is true. When an emitting node says `descend_into_match(node) is
    False`, its subtree is skipped after the match emits — used to avoid
    double-emitting the inner def/class of Python's `decorated_definition`.

    `should_emit` lets contextual chunk kinds (TS's `variable_declarator`,
    which is only meaningful when its initializer is an `arrow_function`)
    live in the registry without being emitted blindly."""
    kinds = grammar.chunk_node_kinds
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in kinds and grammar.should_emit(node):
            yield node
            if not grammar.descend_into_match(node):
                continue
        # Children pushed in reverse so DFS visits them in source order.
        for child in reversed(node.children):
            stack.append(child)


def chunk_pr_summary(
    *,
    title: str,
    body: str | None,
    repo: str,
    pr_number: int,
    source_url: str | None = None,
) -> PRSummaryChunk | None:
    """Build one embeddable text per PR: title + truncated body. Returns
    None for PRs with no usable text at all (very rare)."""
    title = (title or "").strip()
    body_text = (body or "").strip()[:MAX_PR_BODY_CHARS]
    if not title and not body_text:
        return None
    text = f"{title}\n\n{body_text}" if title and body_text else title or body_text
    return PRSummaryChunk(
        text=text,
        context={
            "repo": repo,
            "pr_number": pr_number,
            "pr_title": title,
            "url": source_url,
        },
    )


def chunk_commit_message(
    message: str,
    *,
    repo: str,
    sha: str,
    source_url: str | None,
) -> CommitMessageChunk | None:
    msg = (message or "").strip()
    if len(msg) < MIN_COMMIT_MESSAGE_LEN:
        return None
    return CommitMessageChunk(
        text=msg,
        context={"repo": repo, "commit_sha": sha, "source_url": source_url},
    )
