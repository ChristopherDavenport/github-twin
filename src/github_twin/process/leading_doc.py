"""Extract the leading doc-comment or docstring for an AST chunk node.

Two extraction patterns cover every grammar we support today:

1. **Inside-body docstring** — Python. The first statement of the
   function/class body is an `expression_statement` wrapping a string
   literal. Used by `extract_python_docstring`.

2. **Preceding-comment block** — Scala, JavaScript, TypeScript, Go,
   Rust. Doc comments live as one or more consecutive comment-typed
   siblings immediately before the declaration. Used by
   `extract_preceding_comments`.

Both helpers return a trimmed, truncated string suitable for inclusion
in an embedding prefix, or None when nothing usable is present. Cap at
`MAX_LEADING_DOC_CHARS` so retrieval prefixes stay bounded.
"""

from __future__ import annotations

from collections.abc import Iterator

from tree_sitter import Node

MAX_LEADING_DOC_CHARS = 240

# Comment node types we treat as doc material. Different tree-sitter
# grammars use different names — covering them all here means each
# grammar's leading_doc callback just calls extract_preceding_comments.
_COMMENT_TYPES = frozenset({"comment", "line_comment", "block_comment", "doc_comment"})

# Wrapper nodes that don't themselves "own" a doc comment but whose
# prev_sibling is where the doc actually sits. JSDoc on `export class X`
# attaches above the `export_statement`, not above the `class_declaration`
# nested inside. Same story for Scala packages and TS decorators.
_DOC_WRAPPER_TYPES = frozenset(
    {
        "export_statement",
        "decorator",
        "decorated_definition",  # python uses this; we cover it via _leading_doc override
    }
)


def _truncate(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if len(text) <= MAX_LEADING_DOC_CHARS:
        return text
    return text[:MAX_LEADING_DOC_CHARS].rstrip() + "…"


def _clean_comment(raw: str) -> str:
    """Strip common comment markers (// /* */ # /// /** */) and leading
    whitespace per line. Tree-sitter returns comment text including the
    delimiters, which would dilute the embedding if we left them in."""
    lines = raw.splitlines()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        # Block-comment delimiters and per-line stars.
        if s.startswith("/**"):
            s = s[3:]
        elif s.startswith("/*"):
            s = s[2:]
        if s.endswith("*/"):
            s = s[:-2]
        s = s.lstrip("*").lstrip()
        # Line-comment delimiters.
        if s.startswith("///"):
            s = s[3:].lstrip()
        elif s.startswith("//"):
            s = s[2:].lstrip()
        elif s.startswith("#"):
            s = s[1:].lstrip()
        if s:
            out.append(s)
    return " ".join(out)


def extract_preceding_comments(node: Node) -> str | None:
    """Walk `node.prev_sibling` while it's a comment-typed node; return
    the concatenated, marker-stripped text or None.

    Tries the node itself first, then climbs through known wrapper nodes
    (`export_statement`, etc.) so that JSDoc on `export class X` is
    still claimed by the inner `class_declaration`. Stops at the first
    non-comment sibling at each level."""
    for anchor in _candidate_anchors(node):
        comments = _collect_comments(anchor)
        if comments is not None:
            return comments
    return None


def _candidate_anchors(node: Node) -> Iterator[Node]:
    """Yield `node`, then each ancestor whose type is a doc-wrapper.
    The wrapper rule lets us find comments above `export_statement` /
    `decorator` / `decorated_definition` when the inner declaration's
    own prev_sibling is a keyword instead of a comment."""
    yield node
    parent = node.parent
    while parent is not None and parent.type in _DOC_WRAPPER_TYPES:
        yield parent
        parent = parent.parent


def _collect_comments(anchor: Node) -> str | None:
    parts: list[str] = []
    cur = anchor.prev_sibling
    while cur is not None and cur.type in _COMMENT_TYPES:
        parts.append(_clean_comment(_decode(cur)))
        cur = cur.prev_sibling
    if not parts:
        return None
    parts.reverse()
    return _truncate(" ".join(p for p in parts if p))


def extract_python_docstring(node: Node) -> str | None:
    """Python: walk into the node's `body` block; if the first non-comment
    statement is a bare `string` (or an `expression_statement` wrapping
    one — older grammars use that shape), return its literal text.

    `tree-sitter-language-pack` 0.4 returns the string directly; earlier
    grammars wrap it in expression_statement. Handle both."""
    body = node.child_by_field_name("body")
    if body is None:
        return None
    for child in body.named_children:
        if child.type in _COMMENT_TYPES:
            continue
        target = child
        if child.type == "expression_statement" and child.named_child_count >= 1:
            target = child.named_children[0]
        if target.type == "string":
            raw = _decode(target)
            # `string` children expose string_content as a separate node;
            # prefer that when present so we skip the surrounding quotes.
            content = None
            for sub in target.named_children:
                if sub.type == "string_content":
                    content = _decode(sub)
                    break
            return _truncate(content if content is not None else _unwrap_string(raw))
        return None
    return None


def _decode(node: Node) -> str:
    """Local `Node.text` decoder. Tree-sitter stubs type `text` as
    `bytes | None`; nodes from a parse never have None text but mypy
    can't prove that. Returns `""` for the unreachable-in-practice
    None branch."""
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _unwrap_string(raw: str) -> str:
    """Drop python triple/single quotes around a docstring literal.
    Tree-sitter gives us the literal including the surrounding quotes."""
    for q in ('"""', "'''", '"', "'"):
        if raw.startswith(q) and raw.endswith(q) and len(raw) >= 2 * len(q):
            return raw[len(q) : -len(q)]
    return raw


def first_nonempty(*candidates: str | None) -> str | None:
    """Pick the first non-None, non-blank value. Used by grammars whose
    leading-doc story spans two extractors (e.g. Python: docstring OR a
    preceding comment block)."""
    for c in candidates:
        if c and c.strip():
            return c
    return None


# Re-exported for callers that want to compose extractors themselves.
__all__ = [
    "MAX_LEADING_DOC_CHARS",
    "extract_preceding_comments",
    "extract_python_docstring",
    "first_nonempty",
]
