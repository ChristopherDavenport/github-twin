from github_twin.process.chunkers import (
    MAX_CODE_CHUNK_LINES,
    chunk_commit_message,
    chunk_diff,
    is_excluded_path,
)
from github_twin.process.language import language_for_path

PY_DIFF = """diff --git a/foo.py b/foo.py
index 1234567..abcdefg 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,8 @@
 import os
-x = 1
+def hello(name: str) -> str:
+    return f"Hello, {name}!"
+
+
+def goodbye(name: str) -> str:
+    return f"Bye, {name}!"
 # tail
"""

MULTI_FILE_DIFF = (
    PY_DIFF
    + """diff --git a/bar.lock b/bar.lock
--- a/bar.lock
+++ b/bar.lock
@@ -1,1 +1,2 @@
+ignored
+lockfile content
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,1 +1,3 @@
+This is a README.
+(skipped as Markdown)
"""
)

BINARY_DIFF = """diff --git a/img.png b/img.png
Binary files a/img.png and b/img.png differ
"""

DELETION_DIFF = """diff --git a/gone.py b/gone.py
deleted file mode 100644
index 1234567..0000000
--- a/gone.py
+++ /dev/null
@@ -1,3 +0,0 @@
-x = 1
-y = 2
-z = 3
"""


def test_language_for_known_extensions():
    assert language_for_path("foo.py") == "python"
    assert language_for_path("src/main.go") == "go"
    assert language_for_path("a/b/main.rs") == "rust"
    assert language_for_path("x.ts") == "typescript"


def test_language_drops_non_code():
    # Markdown and JSON are dropped — not useful as style exemplars.
    assert language_for_path("README.md") is None
    assert language_for_path("package.json") is None
    # Unknown extensions return None instead of crashing.
    assert language_for_path("mystery.xyz") is None


def test_chunk_diff_emits_added_lines_only():
    """AST-aware chunk_diff yields one CodeChunk per touched function,
    not a single blob covering the run of `+` lines."""
    chunks = list(chunk_diff(PY_DIFF, repo="me/foo", sha="deadbeef", source_url=None))
    syms = {c.context.get("symbol_name") for c in chunks}
    assert syms == {"hello", "goodbye"}
    for c in chunks:
        assert c.language == "python"
        assert c.context["node_kind"] == "function_definition"
        # The pre-image line `x = 1` was deleted; it must not appear in
        # any emitted chunk.
        assert "x = 1" not in c.text
    # And the actual added function bodies do show up in their own chunks.
    hello = next(c for c in chunks if c.context["symbol_name"] == "hello")
    assert "Hello," in hello.text
    goodbye = next(c for c in chunks if c.context["symbol_name"] == "goodbye")
    assert "Bye," in goodbye.text


def test_chunk_diff_respects_excludes_and_drops_markdown():
    chunks = list(
        chunk_diff(
            MULTI_FILE_DIFF,
            repo="me/foo",
            sha="abc",
            source_url=None,
            exclude_patterns=["*.lock"],
        )
    )
    paths = {c.path for c in chunks}
    assert paths == {"foo.py"}  # .lock excluded; .md dropped by language


def test_chunk_diff_skips_binary_and_deletion():
    assert list(chunk_diff(BINARY_DIFF, repo="r", sha="s", source_url=None)) == []
    assert list(chunk_diff(DELETION_DIFF, repo="r", sha="s", source_url=None)) == []


def test_chunk_diff_size_cap():
    big_block = "\n".join(["+x = " + str(i) for i in range(MAX_CODE_CHUNK_LINES + 50)])
    diff = f"diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -0,0 +0,0 @@\n{big_block}\n"
    chunks = list(chunk_diff(diff, repo="r", sha="s", source_url=None))
    # Block of 130 added lines should be split into 2 chunks (≤80 each)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text.splitlines()) <= MAX_CODE_CHUNK_LINES


def test_is_excluded_path():
    pats = ["**/*.lock", "**/node_modules/**"]
    assert is_excluded_path("a/b/foo.lock", pats)
    assert is_excluded_path("a/node_modules/x/y.js", pats)
    assert not is_excluded_path("a/b/foo.py", pats)


def test_chunk_commit_message():
    assert chunk_commit_message("short", repo="r", sha="s", source_url=None) is None
    msg = "Improve handling of edge cases when configuration is malformed"
    chunk = chunk_commit_message(msg, repo="r", sha="s", source_url=None)
    assert chunk is not None
    assert chunk.text == msg
    assert chunk.context["repo"] == "r"


# ---------- AST-aware chunk_diff (Phase 3) ----------


SINGLE_FUNCTION_EDIT_DIFF = """diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,5 +1,5 @@
 def big_one(x):
     # comment
-    return x
+    return x + 1
     # trailing
"""


MULTI_FUNCTION_DIFF = """diff --git a/m.py b/m.py
--- a/m.py
+++ b/m.py
@@ -1,3 +1,9 @@
 def existing(x):
     return x

+def alpha():
+    return 1
+
+def beta():
+    return 2
"""


CONTEXT_ONLY_TOUCH_DIFF = """diff --git a/c.py b/c.py
--- a/c.py
+++ b/c.py
@@ -1,5 +1,6 @@
 def untouched(x):
     return x

+def newcomer():
+    return 99
"""


def test_chunk_diff_ast_emits_one_chunk_per_touched_function():
    chunks = list(chunk_diff(MULTI_FUNCTION_DIFF, repo="r", sha="s", source_url=None))
    syms = {c.context.get("symbol_name") for c in chunks}
    # Two new functions added; `existing` is context only and must NOT
    # show up as a chunk (no added lines fall inside it).
    assert syms == {"alpha", "beta"}
    for c in chunks:
        assert c.context["node_kind"] == "function_definition"


def test_chunk_diff_ast_edit_inside_function_yields_whole_function():
    """A one-line change inside a multi-line function should surface the
    whole function as a single chunk — the line-block path would have
    discarded this entirely (run length < MIN_CODE_CHUNK_LINES)."""
    chunks = list(chunk_diff(SINGLE_FUNCTION_EDIT_DIFF, repo="r", sha="s", source_url=None))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.context["symbol_name"] == "big_one"
    assert c.context["node_kind"] == "function_definition"
    # The chunk contains the new body, not the old one.
    assert "return x + 1" in c.text
    assert "return x\n" not in c.text


def test_chunk_diff_ast_skips_context_only_functions():
    """Only functions whose added lines intersect their range emit chunks."""
    chunks = list(chunk_diff(CONTEXT_ONLY_TOUCH_DIFF, repo="r", sha="s", source_url=None))
    syms = {c.context.get("symbol_name") for c in chunks}
    assert syms == {"newcomer"}


SCALA_DIFF = """diff --git a/Foo.scala b/Foo.scala
--- a/Foo.scala
+++ b/Foo.scala
@@ -1,5 +1,9 @@
 package com.example

 object Foo {
+  def added(x: Int): Int = x + 1
+
+  def alsoAdded(): String = "hi"
+
   def existing: Int = 7
 }
"""


def test_chunk_diff_ast_scala_per_function():
    """Scala grammar is registered; a diff adding two methods inside an
    object should yield one chunk per added method."""
    chunks = list(chunk_diff(SCALA_DIFF, repo="r", sha="s", source_url=None))
    syms = {c.context.get("symbol_name") for c in chunks}
    assert "added" in syms
    assert "alsoAdded" in syms
    # `existing` is untouched context inside the object — no chunk for it.
    assert "existing" not in syms
    for c in chunks:
        assert c.language == "scala"
        assert c.context["node_kind"] == "function_definition"


UNSUPPORTED_LANG_DIFF = """diff --git a/x.rb b/x.rb
--- a/x.rb
+++ b/x.rb
@@ -1,1 +1,5 @@
 module Foo
+  def self.added
+    1
+  end
+end
"""


def test_chunk_diff_unsupported_language_falls_back_to_line_blocks():
    """Ruby has no registered grammar, so chunk_diff should fall back to
    the legacy `+`-line block extractor and still emit something. (If we
    ever add a Ruby grammar, swap this fixture for another unregistered
    language.)"""
    chunks = list(chunk_diff(UNSUPPORTED_LANG_DIFF, repo="r", sha="s", source_url=None))
    assert len(chunks) >= 1
    for c in chunks:
        # Fallback chunks carry no AST metadata.
        assert "node_kind" not in c.context
        assert "symbol_name" not in c.context


METHOD_INSIDE_CLASS_DIFF = """diff --git a/c.py b/c.py
--- a/c.py
+++ b/c.py
@@ -1,4 +1,7 @@
 class Widget:
     def existing(self):
         return 1
+
+    def added(self):
+        return 2
"""


def test_chunk_diff_ast_suppresses_class_when_only_method_changed():
    """Deepest-ancestor rule: a new method inside a class emits the method
    chunk, not the wrapping class chunk — the class wrapper would be noise
    re-indexing context the diff didn't actually touch."""
    chunks = list(chunk_diff(METHOD_INSIDE_CLASS_DIFF, repo="r", sha="s", source_url=None))
    syms = {c.context.get("symbol_name") for c in chunks}
    kinds = {c.context.get("node_kind") for c in chunks}
    assert "added" in syms
    # The class itself must NOT emit — every added line is claimed by the
    # method.
    assert "Widget" not in syms
    assert "class_definition" not in kinds


CLASS_HEADER_CHANGE_DIFF = """diff --git a/h.py b/h.py
--- a/h.py
+++ b/h.py
@@ -1,3 +1,4 @@
 class Widget:
+    \"\"\"A small widget.\"\"\"
     def existing(self):
         return 1
"""


def test_chunk_diff_ast_emits_class_when_header_region_changed():
    """The flip side: when the added line is at the class level (above
    any method), the deepest chunkable ancestor IS the class, so the
    class emits."""
    chunks = list(chunk_diff(CLASS_HEADER_CHANGE_DIFF, repo="r", sha="s", source_url=None))
    syms = {c.context.get("symbol_name") for c in chunks}
    assert syms == {"Widget"}
    assert chunks[0].context["node_kind"] == "class_definition"


def test_chunk_diff_unparseable_python_falls_back_per_path():
    """If a Python hunk's post-image yields no chunkable AST nodes (e.g.,
    a deep edit with no visible function boundary), chunk_diff should
    fall back to the line-block flow for that path."""
    diff = (
        "diff --git a/d.py b/d.py\n"
        "--- a/d.py\n"
        "+++ b/d.py\n"
        "@@ -10,3 +10,5 @@\n"
        "     # before\n"
        "+    new_local_1 = 1\n"
        "+    new_local_2 = 2\n"
        "+    new_local_3 = 3\n"
        "     # after\n"
    )
    chunks = list(chunk_diff(diff, repo="r", sha="s", source_url=None))
    # Three added lines is ≥ MIN_CODE_CHUNK_LINES, so the line-block
    # fallback should emit one block.
    assert len(chunks) == 1
    assert "node_kind" not in chunks[0].context
