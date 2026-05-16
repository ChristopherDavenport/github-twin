"""Go tree-sitter grammar.

The chunkable set is:
- `function_declaration` — `func F(...) { ... }`
- `method_declaration` — `func (r *T) M(...) { ... }`
- `type_spec` — the per-type declaration inside a `type ( ... )` block
  OR a single `type X struct/...`. We register `type_spec` (not the
  outer `type_declaration`) because the name lives on the spec, and a
  multi-type block is more useful split into per-type chunks anyway.

For `method_declaration`, `symbol_name` returns `Receiver.Method` so a
search for "Server.Handle" lands on the right method even when the
codebase has several types with `Handle` methods. Pointer / value /
type-only receivers all reduce to the bare type name; generic receivers
(`func (s *Server[T]) M()`) likewise surface the type identifier without
the type-parameter list.
"""

from __future__ import annotations

from tree_sitter import Node

from github_twin.process.grammars import LanguageGrammar, decode_text, register
from github_twin.process.leading_doc import extract_preceding_comments


def _first_type_identifier(node: Node) -> Node | None:
    """Walk a `parameter_declaration` subtree and return the first
    `type_identifier` child. Handles pointer_type and generic_type
    wrappers transparently."""
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "type_identifier":
            return n
        for c in reversed(n.children):
            stack.append(c)
    return None


def _method_receiver_type(method: Node) -> str | None:
    """Return the receiver type identifier text for a `method_declaration`,
    or None if the AST doesn't have the expected shape (parser error,
    incomplete code, etc.)."""
    for c in method.children:
        if c.type != "parameter_list":
            continue
        # The first parameter_list of a method_declaration IS the
        # receiver (the args list comes after the method name).
        for inner in c.children:
            if inner.type == "parameter_declaration":
                t = _first_type_identifier(inner)
                if t is not None:
                    return decode_text(t)
        return None
    return None


def _symbol_name(node: Node) -> str | None:
    name = node.child_by_field_name("name")
    if name is None:
        return None
    method_name = decode_text(name)
    if node.type != "method_declaration":
        return method_name
    receiver = _method_receiver_type(node)
    return f"{receiver}.{method_name}" if receiver else method_name


register(
    LanguageGrammar(
        language="go",
        parser_name="go",
        chunk_node_kinds=frozenset({"function_declaration", "method_declaration", "type_spec"}),
        symbol_name=_symbol_name,
        leading_doc=extract_preceding_comments,
    )
)
