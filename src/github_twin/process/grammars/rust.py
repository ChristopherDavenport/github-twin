"""Rust tree-sitter grammar.

Chunkable nodes are top-level item declarations: functions, impl blocks,
structs, enums, and traits. `impl_item` contains nested `function_item`
children — under the default descent rule both emit, which gives the
expected multi-granularity retrieval (the whole impl block AND each
method, mirroring how Python `class_definition` works).

`impl_item` does not have a `name` field; the implementing-type lives on
the `type` field. The fallback in `_symbol_name` handles that.
"""

from __future__ import annotations

from tree_sitter import Node

from github_twin.process.grammars import LanguageGrammar, decode_text, register
from github_twin.process.leading_doc import extract_preceding_comments


def _symbol_name(node: Node) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return decode_text(name)
    # impl_item: `impl Trait for Type` or `impl Type` — surface the
    # implementing-type identifier.
    type_node = node.child_by_field_name("type")
    if type_node is not None:
        return decode_text(type_node)
    return None


register(
    LanguageGrammar(
        language="rust",
        parser_name="rust",
        chunk_node_kinds=frozenset(
            {
                "function_item",
                "impl_item",
                "struct_item",
                "enum_item",
                "trait_item",
            }
        ),
        symbol_name=_symbol_name,
        leading_doc=extract_preceding_comments,
    )
)
