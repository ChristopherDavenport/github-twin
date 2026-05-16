"""Tests for the tree-sitter grammar registry."""

from __future__ import annotations

import tree_sitter
from tree_sitter_language_pack import get_language

from github_twin.process.grammars import grammar_for_language


def _parse_python(src: str):
    lang = get_language("python")
    parser = tree_sitter.Parser(lang)
    return parser.parse(src.encode())


def test_grammar_lookup_known_language():
    g = grammar_for_language("python")
    assert g is not None
    assert g.language == "python"
    assert "function_definition" in g.chunk_node_kinds
    assert "class_definition" in g.chunk_node_kinds
    assert "decorated_definition" in g.chunk_node_kinds


def test_grammar_lookup_unknown_language_returns_none():
    assert grammar_for_language("zig") is None
    assert grammar_for_language("nonexistent") is None
    assert grammar_for_language(None) is None


def test_python_symbol_name_for_function():
    g = grammar_for_language("python")
    tree = _parse_python("def greet(name):\n    return name\n")
    # Walk to the function_definition.
    func = next(c for c in tree.root_node.children if c.type == "function_definition")
    assert g.symbol_name(func) == "greet"


def test_python_symbol_name_for_class():
    g = grammar_for_language("python")
    tree = _parse_python("class Widget:\n    pass\n")
    cls = next(c for c in tree.root_node.children if c.type == "class_definition")
    assert g.symbol_name(cls) == "Widget"


def test_python_symbol_name_for_decorated_def():
    g = grammar_for_language("python")
    tree = _parse_python("@cache\ndef compute():\n    return 42\n")
    dec = next(c for c in tree.root_node.children if c.type == "decorated_definition")
    assert g.symbol_name(dec) == "compute"


def test_python_does_not_descend_into_decorated_def():
    """The wrapped def inside a decorated_definition has identical name and
    a sub-range; descending would emit a duplicate chunk."""
    g = grammar_for_language("python")
    tree = _parse_python("@cache\ndef compute():\n    return 42\n")
    dec = next(c for c in tree.root_node.children if c.type == "decorated_definition")
    assert g.descend_into_match(dec) is False


def test_python_descends_into_class_to_find_methods():
    g = grammar_for_language("python")
    tree = _parse_python("class W:\n    def m(self):\n        return 1\n")
    cls = next(c for c in tree.root_node.children if c.type == "class_definition")
    assert g.descend_into_match(cls) is True


# ---------- Phase 4 grammars ----------


def _parse(lang_name: str, src: str):
    lang = get_language(lang_name)
    parser = tree_sitter.Parser(lang)
    return parser.parse(src.encode())


def test_javascript_grammar_lookup_and_symbol_names():
    g = grammar_for_language("javascript")
    assert g is not None
    assert {"function_declaration", "class_declaration", "method_definition"} <= g.chunk_node_kinds
    tree = _parse("javascript", "function plain(){}\nclass C { greet(){} }\n")
    fn = next(c for c in tree.root_node.children if c.type == "function_declaration")
    cls = next(c for c in tree.root_node.children if c.type == "class_declaration")
    assert g.symbol_name(fn) == "plain"
    assert g.symbol_name(cls) == "C"


def test_typescript_grammar_covers_interfaces_and_type_aliases():
    g = grammar_for_language("typescript")
    assert g is not None
    assert "interface_declaration" in g.chunk_node_kinds
    assert "type_alias_declaration" in g.chunk_node_kinds
    src = "interface IThing { a: number }\ntype Alias = number\n"
    tree = _parse("typescript", src)
    iface = next(c for c in tree.root_node.children if c.type == "interface_declaration")
    alias = next(c for c in tree.root_node.children if c.type == "type_alias_declaration")
    assert g.symbol_name(iface) == "IThing"
    assert g.symbol_name(alias) == "Alias"


def test_typescript_and_tsx_share_node_kinds():
    """`.tsx` files use the `tsx` parser; both grammars target the same
    chunkable node set so retrieval semantics stay consistent."""
    ts = grammar_for_language("typescript")
    tsx = grammar_for_language("tsx")
    assert ts is not None
    assert tsx is not None
    assert ts.chunk_node_kinds == tsx.chunk_node_kinds


def test_go_grammar_uses_type_spec_for_naming():
    """type_declaration in Go wraps one or more type_specs; the name lives
    on the spec, so we register the spec — not the declaration — as the
    chunkable unit."""
    g = grammar_for_language("go")
    assert g is not None
    assert "type_spec" in g.chunk_node_kinds
    assert "function_declaration" in g.chunk_node_kinds
    assert "method_declaration" in g.chunk_node_kinds
    tree = _parse("go", "package p\ntype Server struct {}\nfunc (s *Server) M() int { return 1 }\n")
    # Find the type_spec under type_declaration.
    type_spec = None
    method = None

    def find(node):
        nonlocal type_spec, method
        if node.type == "type_spec" and type_spec is None:
            type_spec = node
        if node.type == "method_declaration" and method is None:
            method = node
        for c in node.children:
            find(c)

    find(tree.root_node)
    assert g.symbol_name(type_spec) == "Server"
    # Method symbol_name is prefixed with the receiver type to disambiguate
    # codebases that have many `Handle` / `String` / `Close` methods.
    assert g.symbol_name(method) == "Server.M"


def test_go_grammar_method_receiver_prefix_handles_all_receiver_forms():
    """Pointer, value, and type-only receivers should all reduce to the
    bare type name (no `*`, no parameter name)."""
    g = grammar_for_language("go")
    tree = _parse(
        "go",
        "package p\n"
        "func (s *Server) PtrM() int { return 1 }\n"
        "func (s Server) ValM() int { return 1 }\n"
        "func (Empty) Anon() int { return 1 }\n",
    )
    methods = []

    def find(node):
        if node.type == "method_declaration":
            methods.append(node)
        for c in node.children:
            find(c)

    find(tree.root_node)
    syms = [g.symbol_name(m) for m in methods]
    assert syms == ["Server.PtrM", "Server.ValM", "Empty.Anon"]


def test_go_grammar_function_declaration_unprefixed():
    """Plain `func F()` is NOT prefixed — receiver prefix is only for methods."""
    g = grammar_for_language("go")
    tree = _parse("go", "package p\nfunc Plain() int { return 1 }\n")
    fn = None

    def find(node):
        nonlocal fn
        if node.type == "function_declaration":
            fn = node
        for c in node.children:
            find(c)

    find(tree.root_node)
    assert g.symbol_name(fn) == "Plain"


def test_rust_grammar_impl_item_uses_type_field():
    """impl_item has no `name` field; the implementing type is on `type`.
    The Rust grammar's symbol_name extractor falls back to that field."""
    g = grammar_for_language("rust")
    assert g is not None
    assert {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
    } <= g.chunk_node_kinds
    tree = _parse("rust", "pub struct Widget;\nimpl Widget { pub fn new() -> Self { Widget } }\n")
    struct = next(c for c in tree.root_node.children if c.type == "struct_item")
    impl = next(c for c in tree.root_node.children if c.type == "impl_item")
    assert g.symbol_name(struct) == "Widget"
    assert g.symbol_name(impl) == "Widget"


def test_rust_grammar_function_item_symbol_name():
    g = grammar_for_language("rust")
    tree = _parse("rust", "pub fn compute(x: i32) -> i32 { x + 1 }\n")
    fn = next(c for c in tree.root_node.children if c.type == "function_item")
    assert g.symbol_name(fn) == "compute"


def test_typescript_should_emit_gates_variable_declarator():
    """`variable_declarator` is registered as chunkable but only emits when
    it binds an arrow_function at module top level."""
    g = grammar_for_language("typescript")
    assert "variable_declarator" in g.chunk_node_kinds

    def declarator_in(src: str):
        tree = _parse("typescript", src)

        def find(node):
            if node.type == "variable_declarator":
                return node
            for c in node.children:
                r = find(c)
                if r is not None:
                    return r
            return None

        return find(tree.root_node)

    # Top-level arrow → emits.
    d = declarator_in("const plain = () => 1;\n")
    assert d is not None
    assert g.should_emit(d) is True

    # Non-arrow initializer → suppressed.
    d = declarator_in("const x = 42;\n")
    assert d is not None
    assert g.should_emit(d) is False

    # Function expression (not arrow) → suppressed.
    d = declarator_in("const f = function() { return 1; };\n")
    assert d is not None
    assert g.should_emit(d) is False

    # Nested arrow (inside a function) → suppressed; the enclosing
    # function already emits.
    d = declarator_in("function outer() {\n  const inner = () => 1;\n}\n")
    assert d is not None  # this is the `inner` declarator
    assert d.child_by_field_name("name").text.decode() == "inner"
    assert g.should_emit(d) is False
