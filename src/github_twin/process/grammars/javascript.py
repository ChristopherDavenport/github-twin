"""JavaScript tree-sitter grammar.

Chunkable nodes are named declarations: regular and async functions,
generator functions, classes, and class methods (including constructors
and static methods).

Top-level arrow functions bound to a `variable_declarator` are NOT
chunked here — they nest one level deep (`lexical_declaration >
variable_declarator > arrow_function`) and recognising them as units
requires more grammar gymnastics than the corpus value justifies right
now. Arrow functions still appear inside other emitted chunks.
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
        language="javascript",
        parser_name="javascript",
        chunk_node_kinds=frozenset(
            {
                "function_declaration",
                "generator_function_declaration",
                "class_declaration",
                "method_definition",
            }
        ),
        symbol_name=_symbol_name,
        leading_doc=extract_preceding_comments,
    )
)
