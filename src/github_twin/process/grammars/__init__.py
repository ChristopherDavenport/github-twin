"""Tree-sitter grammar registry for language-aware chunking.

Each supported language registers a `LanguageGrammar` describing:

- The `tree-sitter-language-pack` parser name.
- The set of AST node types we treat as chunkable units (functions,
  classes, impl blocks, ...).
- A `symbol_name` extractor that turns one of those nodes into a
  human-readable identifier (e.g. `def calculate_total` → `calculate_total`).
- An optional `descend_into_match` predicate that says whether to recurse
  into a matched node looking for nested chunks. Default: always recurse,
  so a class still emits its methods as separate chunks. The Python
  grammar overrides this for `decorated_definition` to avoid double-emitting
  the wrapped function.

Lookup is keyed on the canonical language tag emitted by
`language_for_path`, so the grammar registry plugs in below the existing
extension → language mapping without duplicating it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tree_sitter import Node


def _default_descend(_node: Node) -> bool:
    return True


def _default_should_emit(_node: Node) -> bool:
    return True


def _default_leading_doc(_node: Node) -> str | None:
    return None


@dataclass(frozen=True)
class LanguageGrammar:
    language: str
    parser_name: str
    chunk_node_kinds: frozenset[str]
    symbol_name: Callable[[Node], str | None]
    descend_into_match: Callable[[Node], bool] = field(default=_default_descend)
    # Optional post-filter: invoked on every node whose type is in
    # `chunk_node_kinds`. Returning False suppresses emission while still
    # allowing the walk to descend into the node's children. Used for
    # contextual chunkable kinds (e.g. TypeScript's variable_declarator,
    # which is only meaningful when it binds an arrow_function).
    should_emit: Callable[[Node], bool] = field(default=_default_should_emit)
    # Returns a short leading doc-comment or docstring for the node, or
    # None. The chunker stores the result in `chunk.context["leading_doc"]`
    # and the embed-time prefix splices it into the header. Default: None.
    leading_doc: Callable[[Node], str | None] = field(default=_default_leading_doc)


_REGISTRY: dict[str, LanguageGrammar] = {}


def register(grammar: LanguageGrammar) -> None:
    _REGISTRY[grammar.language] = grammar


def grammar_for_language(language: str | None) -> LanguageGrammar | None:
    if language is None:
        return None
    return _REGISTRY.get(language)


def decode_text(node: Node | None) -> str:
    """Safe `node.text.decode(...)` for tree-sitter nodes.

    Tree-sitter's type stubs declare `Node.text` as `bytes | None` because
    detached nodes (constructed outside a parse) have no source text. In
    practice every node we encounter came from `Parser.parse(...)` so
    `text` is non-None — but rather than scattering `assert ... is not
    None` everywhere, this helper centralizes the safety check and gives
    every grammar module the same `Node | None -> str` shape.

    Returns `""` on missing nodes / missing text. Symbol extractors that
    care about absence already check the `child_by_field_name(...) is
    None` branch above this call.
    """
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


# Importing the per-language modules triggers their register() calls.
from github_twin.process.grammars import go as _go  # noqa: E402,F401
from github_twin.process.grammars import javascript as _js  # noqa: E402,F401
from github_twin.process.grammars import python as _python  # noqa: E402,F401
from github_twin.process.grammars import rust as _rust  # noqa: E402,F401
from github_twin.process.grammars import scala as _scala  # noqa: E402,F401
from github_twin.process.grammars import typescript as _ts  # noqa: E402,F401
