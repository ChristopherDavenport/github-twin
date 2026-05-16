"""Tests for `chunk_file` — file-at-HEAD windowing (O-C)."""

from __future__ import annotations

from github_twin.process.chunkers import (
    FILE_CHUNK_OVERLAP,
    MAX_CODE_CHUNK_LINES,
    MIN_CODE_CHUNK_LINES,
    chunk_file,
)


def _make_source(n_lines: int) -> str:
    return "\n".join(f"line_{i}" for i in range(n_lines))


def test_chunk_file_short_file_yields_one_chunk():
    text = _make_source(10)
    chunks = list(chunk_file(text, repo="org/r", path="src/x.py"))
    assert len(chunks) == 1
    assert chunks[0].language == "python"
    assert chunks[0].path == "src/x.py"
    assert chunks[0].text.count("\n") == 9
    assert chunks[0].context["start_line"] == 1
    assert chunks[0].context["end_line"] == 10
    assert chunks[0].context["repo"] == "org/r"


def test_chunk_file_long_file_windows_with_overlap():
    n = MAX_CODE_CHUNK_LINES * 3
    text = _make_source(n)
    chunks = list(chunk_file(text, repo="org/r", path="src/x.py"))
    # Windows step by (MAX - overlap), so ceil(n / step) ~= 3-4 chunks.
    assert len(chunks) >= 3
    # First window covers lines 1..MAX.
    assert chunks[0].context["start_line"] == 1
    assert chunks[0].context["end_line"] == MAX_CODE_CHUNK_LINES
    # Second window starts (MAX - overlap) lines in.
    step = MAX_CODE_CHUNK_LINES - FILE_CHUNK_OVERLAP
    assert chunks[1].context["start_line"] == step + 1
    # No window exceeds MAX lines.
    for ck in chunks:
        assert ck.text.splitlines() and len(ck.text.splitlines()) <= MAX_CODE_CHUNK_LINES


def test_chunk_file_skips_too_short():
    text = _make_source(MIN_CODE_CHUNK_LINES - 1)
    assert list(chunk_file(text, repo="org/r", path="x.py")) == []


def test_chunk_file_skips_unknown_language():
    text = _make_source(20)
    assert list(chunk_file(text, repo="org/r", path="notes.unknownext")) == []


def test_chunk_file_respects_excludes():
    text = _make_source(20)
    # `**/vendor/**` matches `pkg/vendor/...` but not root-level `vendor/...`
    # (fnmatch's `**` is just `*` and requires at least the literal `/vendor/`
    # to appear with something on each side). That's the same semantics
    # `chunk_diff` uses today, so we test the matching one.
    out = list(
        chunk_file(
            text,
            repo="org/r",
            path="pkg/vendor/whatever/x.py",
            exclude_patterns=["**/vendor/**"],
        )
    )
    assert out == []


def test_chunk_file_carries_head_sha_in_context():
    chunks = list(
        chunk_file(
            _make_source(20),
            repo="org/r",
            path="x.py",
            head_sha="deadbeef",
            source_url="https://gh/org/r/blob/deadbeef/x.py",
        )
    )
    assert chunks[0].context["head_sha"] == "deadbeef"
    assert chunks[0].context["source_url"].endswith("/x.py")


# ---------- AST chunking (tree-sitter) ----------


PY_AST_FILE = """\
def foo():
    return 1


class Bar:
    def baz(self):
        return 2

    def qux(self, n):
        return n + 1


@my_decorator
def quux():
    x = 1
    return x
"""


def test_chunk_file_ast_python_emits_per_unit_chunks():
    """Python files with a registered grammar produce one chunk per
    function/class/decorated-def — not arbitrary line windows."""
    chunks = list(chunk_file(PY_AST_FILE, repo="org/r", path="src/x.py"))
    kinds = [(c.context["node_kind"], c.context["symbol_name"]) for c in chunks]
    assert ("function_definition", "foo") in kinds
    assert ("class_definition", "Bar") in kinds
    assert ("function_definition", "baz") in kinds
    assert ("function_definition", "qux") in kinds
    assert ("decorated_definition", "quux") in kinds
    # No duplicate inner function_definition for the decorated def.
    assert sum(1 for k, s in kinds if s == "quux") == 1
    # Every AST chunk carries start_line/end_line.
    for c in chunks:
        assert c.context["start_line"] >= 1
        assert c.context["end_line"] >= c.context["start_line"]


def test_chunk_file_ast_handles_malformed_python_via_tree_sitter_recovery():
    """tree-sitter is error-tolerant; partially broken files still yield
    well-formed nodes around the error region. The chunker should not raise."""
    broken = (
        "def good_one():\n"
        "    return 1\n"
        "\n"
        "def busted(:\n"  # syntax error in this function's parameters
        "    return 2\n"
        "\n"
        "class Healthy:\n"
        "    def m(self):\n"
        "        return 3\n"
    )
    chunks = list(chunk_file(broken, repo="org/r", path="b.py"))
    syms = {c.context.get("symbol_name") for c in chunks}
    # The healthy regions should still surface.
    assert "good_one" in syms
    assert "Healthy" in syms
    assert "m" in syms


def test_chunk_file_unsupported_language_falls_back_to_line_windows():
    """Languages without a registered grammar use the line-window path,
    which produces chunks without node_kind/symbol_name."""
    text = _make_source(120)  # > MAX so windowing happens
    chunks = list(chunk_file(text, repo="org/r", path="src/x.go"))
    assert len(chunks) >= 1
    # No AST metadata on fallback chunks.
    for c in chunks:
        assert "node_kind" not in c.context
        assert "symbol_name" not in c.context


def test_chunk_file_python_with_no_chunkable_nodes_falls_back():
    """A Python file consisting only of imports + assignments has no
    function_definition / class_definition nodes; the chunker should fall
    through to line-windows so the file is still indexed."""
    text = "\n".join(["import os"] * 5 + ["X = 1"] * 5)
    chunks = list(chunk_file(text, repo="org/r", path="conf.py"))
    assert len(chunks) == 1
    assert "node_kind" not in chunks[0].context


# ---------- Phase 4: per-language AST chunking ----------


JS_FILE = """\
function plain() {
  return 1;
}

async function asyncFn() {
  return 2;
}

class Widget {
  constructor() {
    this.x = 1;
  }
  greet() {
    return 'hi';
  }
}
"""


def test_chunk_file_ast_javascript_per_unit():
    chunks = list(chunk_file(JS_FILE, repo="r", path="src/a.js"))
    pairs = {(c.context["node_kind"], c.context["symbol_name"]) for c in chunks}
    assert ("function_declaration", "plain") in pairs
    assert ("function_declaration", "asyncFn") in pairs
    assert ("class_declaration", "Widget") in pairs
    assert ("method_definition", "constructor") in pairs
    assert ("method_definition", "greet") in pairs


TS_FILE = """\
function foo(x: number): number {
  return x + 1;
}

interface IThing {
  a: number;
  b: string;
}

type Alias = { a: number };

class Widget<T> {
  constructor(public name: T) {}
  greet(): string {
    return 'hi';
  }
}
"""


def test_chunk_file_ast_typescript_per_unit_includes_interface_and_type_alias():
    chunks = list(chunk_file(TS_FILE, repo="r", path="src/a.ts"))
    pairs = {(c.context["node_kind"], c.context["symbol_name"]) for c in chunks}
    assert ("function_declaration", "foo") in pairs
    assert ("interface_declaration", "IThing") in pairs
    assert ("type_alias_declaration", "Alias") in pairs
    assert ("class_declaration", "Widget") in pairs
    assert ("method_definition", "greet") in pairs


TS_ARROW_FILE = """\
export const plain = () => 1;

export const asyncOne = async (x: number): Promise<number> => {
  return x + 1;
};

const NUMERIC = 42;
const fnExpr = function() { return 3; };

function outer() {
  const nested = () => 99;
  return nested;
}

class C {
  m(): string { return 'x'; }
}
"""


def test_chunk_file_ast_typescript_emits_top_level_arrow_declarators():
    """Top-level `const foo = () => ...` chunks as a variable_declarator;
    non-arrow declarators and nested arrows are suppressed."""
    chunks = list(chunk_file(TS_ARROW_FILE, repo="r", path="src/a.ts"))
    pairs = {(c.context["node_kind"], c.context["symbol_name"]) for c in chunks}
    # Top-level arrow declarators emit.
    assert ("variable_declarator", "plain") in pairs
    assert ("variable_declarator", "asyncOne") in pairs
    # Non-arrow initializers do NOT emit as declarator chunks.
    declarator_syms = {sym for kind, sym in pairs if kind == "variable_declarator"}
    assert "NUMERIC" not in declarator_syms
    assert "fnExpr" not in declarator_syms
    # Nested arrow inside `outer` does NOT get its own declarator chunk
    # — `outer` itself already chunks.
    assert "nested" not in declarator_syms
    assert ("function_declaration", "outer") in pairs
    assert ("class_declaration", "C") in pairs


GO_FILE = """\
package main

import "fmt"

func Plain() int {
\treturn 1
}

type Server struct {
\tx int
}

func (s *Server) Method() int {
\treturn s.x
}

func (s Server) Other() string {
\treturn "hi"
}

type Greeter interface {
\tGreet() string
}
"""


def test_chunk_file_ast_go_emits_type_spec_and_methods():
    """type_spec (not type_declaration) is the chunk unit; both methods
    on Server emit individually. Method symbol_names are
    receiver-prefixed so `Server.Method` is distinguishable from any
    `*.Method` in another type."""
    chunks = list(chunk_file(GO_FILE, repo="r", path="x.go"))
    pairs = {(c.context["node_kind"], c.context["symbol_name"]) for c in chunks}
    assert ("function_declaration", "Plain") in pairs
    assert ("type_spec", "Server") in pairs
    assert ("type_spec", "Greeter") in pairs
    assert ("method_declaration", "Server.Method") in pairs
    assert ("method_declaration", "Server.Other") in pairs


RUST_FILE = """\
pub fn plain() -> i32 {
    1
}

pub struct Widget {
    name: String,
}

impl Widget {
    pub fn new(name: String) -> Self {
        Widget { name }
    }
    pub fn greet(&self) -> String {
        self.name.clone()
    }
}

pub enum Color {
    Red,
    Green,
    Blue,
}

pub trait Greeter {
    fn greet(&self) -> String;
}
"""


def test_chunk_file_ast_rust_emits_items_and_impl_methods():
    """impl_item descends into function_item children — both the impl
    chunk and its methods emit, matching the Python class behavior."""
    chunks = list(chunk_file(RUST_FILE, repo="r", path="x.rs"))
    pairs = {(c.context["node_kind"], c.context["symbol_name"]) for c in chunks}
    assert ("function_item", "plain") in pairs
    assert ("struct_item", "Widget") in pairs
    assert ("impl_item", "Widget") in pairs
    assert ("function_item", "new") in pairs
    assert ("function_item", "greet") in pairs
    assert ("enum_item", "Color") in pairs
    assert ("trait_item", "Greeter") in pairs
