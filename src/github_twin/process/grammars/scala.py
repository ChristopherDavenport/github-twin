"""Scala tree-sitter grammar.

Chunkable nodes cover the common top-level declarations: `object`, `class`
(including `case class`), `trait`, concrete `def` methods/functions, and
abstract `def` declarations (used inside traits and abstract classes). All
expose a `name` field on the AST, so the symbol_name extractor matches the
Python path.

The same descend-default applies as Python: we recurse into matched
nodes, so a `class Foo { def bar = ... }` emits separately as the class
chunk and the method chunk. No node type wraps a duplicate of a different
chunk kind (the way Python's `decorated_definition` does), so the default
`descend_into_match` is sufficient.

Upstream grammar: https://github.com/tree-sitter/tree-sitter-scala —
bundled inside `tree-sitter-language-pack` so no extra dependency.
"""

from __future__ import annotations

from tree_sitter import Node

from github_twin.process.grammars import LanguageGrammar, decode_text, register
from github_twin.process.leading_doc import extract_preceding_comments


def _symbol_name(node: Node) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return decode_text(name)
    return None


register(
    LanguageGrammar(
        language="scala",
        parser_name="scala",
        chunk_node_kinds=frozenset(
            {
                "object_definition",
                "class_definition",
                "trait_definition",
                "function_definition",
                "function_declaration",
            }
        ),
        symbol_name=_symbol_name,
        leading_doc=extract_preceding_comments,
    )
)
