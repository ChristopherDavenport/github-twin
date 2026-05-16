"""TypeScript tree-sitter grammar.

Covers the JS chunkable set plus TS-only declarations: interfaces, type
aliases, and abstract classes. All expose a `name` field via the AST.

Top-level `const foo = () => { ... }` is also chunkable. Arrow functions
nest one level inside `lexical_declaration > variable_declarator >
arrow_function`, so the chunkable unit is the `variable_declarator` and
`should_emit` filters it to:
  - the value child is an `arrow_function`, and
  - the declarator is at module top level (its lexical_declaration's
    parent is the `program` node).

Restricting to top-level avoids double-emitting closures that already
appear inside an enclosing function/class chunk.

The grammar is also registered under the `tsx` language tag (the parser
name `tsx` is what `tree-sitter-language-pack` uses for `.tsx` files);
we deliberately keep these as separate registry entries so the
extension → grammar mapping stays predictable, but they share the same
node-kind set, symbol_name extractor, and should_emit predicate.
"""

from __future__ import annotations

from tree_sitter import Node

from github_twin.process.grammars import LanguageGrammar, decode_text, register
from github_twin.process.leading_doc import extract_preceding_comments

_NODE_KINDS = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "variable_declarator",
    }
)


def _symbol_name(node: Node) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return decode_text(name)
    return None


def _is_top_level_arrow_declarator(node: Node) -> bool:
    """True when `node` is a `variable_declarator` binding an
    `arrow_function` at module top level.

    Top-level here means: the declarator's enclosing
    `lexical_declaration` lives directly under `program`, optionally
    wrapped in an `export_statement`. Anything nested inside a function
    body, statement block, or class is rejected — those chunks already
    surface via their enclosing function/class chunk.
    """
    if node.type != "variable_declarator":
        return False
    value = node.child_by_field_name("value")
    if value is None or value.type != "arrow_function":
        return False
    parent = node.parent
    if parent is None or parent.type != "lexical_declaration":
        return False
    enclosing = parent.parent
    if enclosing is not None and enclosing.type == "export_statement":
        enclosing = enclosing.parent
    return enclosing is not None and enclosing.type == "program"


def _should_emit(node: Node) -> bool:
    # variable_declarator is gated to the top-level-arrow case; every
    # other registered kind always emits.
    if node.type == "variable_declarator":
        return _is_top_level_arrow_declarator(node)
    return True


register(
    LanguageGrammar(
        language="typescript",
        parser_name="typescript",
        chunk_node_kinds=_NODE_KINDS,
        symbol_name=_symbol_name,
        should_emit=_should_emit,
        leading_doc=extract_preceding_comments,
    )
)


register(
    LanguageGrammar(
        language="tsx",
        parser_name="tsx",
        chunk_node_kinds=_NODE_KINDS,
        symbol_name=_symbol_name,
        should_emit=_should_emit,
        leading_doc=extract_preceding_comments,
    )
)
