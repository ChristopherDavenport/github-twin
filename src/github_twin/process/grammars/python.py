"""Python tree-sitter grammar.

Chunkable nodes are functions, classes, and decorated definitions.
Methods inside a class still emit as their own chunks (the class itself
also emits, so retrieval can match either the whole class or a single
method). For `decorated_definition` we skip recursing into the inner
function/class wrapped by the decorator — that inner node would be a
near-duplicate chunk.
"""

from __future__ import annotations

from tree_sitter import Node

from github_twin.process.grammars import LanguageGrammar, decode_text, register
from github_twin.process.leading_doc import (
    extract_preceding_comments,
    extract_python_docstring,
    first_nonempty,
)

_NAMED_KINDS = {"function_definition", "class_definition"}


def _symbol_name(node: Node) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return decode_text(name)
    # decorated_definition holds the real def/class as a child.
    for c in node.children:
        if c.type in _NAMED_KINDS:
            inner = c.child_by_field_name("name")
            if inner is not None:
                return decode_text(inner)
    return None


def _descend(node: Node) -> bool:
    # The wrapped def/class inside a decorated_definition has the same
    # name and a sub-range of the same lines — skip it to avoid double-emit.
    return node.type != "decorated_definition"


def _leading_doc(node: Node) -> str | None:
    # decorated_definition wraps the real def/class — look at the inner
    # for its docstring, but still claim any preceding `#` comments.
    inner = node
    if node.type == "decorated_definition":
        for c in node.children:
            if c.type in _NAMED_KINDS:
                inner = c
                break
    return first_nonempty(extract_python_docstring(inner), extract_preceding_comments(node))


register(
    LanguageGrammar(
        language="python",
        parser_name="python",
        chunk_node_kinds=frozenset(
            {"function_definition", "class_definition", "decorated_definition"}
        ),
        symbol_name=_symbol_name,
        descend_into_match=_descend,
        leading_doc=_leading_doc,
    )
)
