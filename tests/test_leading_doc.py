"""Leading-doc extraction for AST chunks.

End-to-end via `chunk_file`: feed a tiny source file, walk the produced
chunks, assert `context["leading_doc"]` is populated for the docs we'd
expect a human reader to see. Tree-sitter is bundled (required dep), so
these tests run without skips.
"""

from __future__ import annotations

from github_twin.process.chunkers import chunk_file


def _doc_for(symbol: str, chunks: list) -> str | None:
    for c in chunks:
        if c.context.get("symbol_name") == symbol:
            return c.context.get("leading_doc")
    return None


# ---------- python: inside-body docstring ----------


def test_python_function_docstring_lifted():
    src = '''def handle_request(req):
    """Validate auth headers and dispatch to the right handler."""
    return _dispatch(req)


def no_doc():
    return 1
'''
    chunks = list(chunk_file(src, repo="me/x", path="src/router.py"))
    assert (
        _doc_for("handle_request", chunks)
        == "Validate auth headers and dispatch to the right handler."
    )
    assert _doc_for("no_doc", chunks) is None


def test_python_class_docstring_lifted():
    src = '''class Worker:
    """A worker that polls the queue and runs jobs."""
    def run(self):
        return 1
'''
    chunks = list(chunk_file(src, repo="me/x", path="src/worker.py"))
    assert _doc_for("Worker", chunks) == "A worker that polls the queue and runs jobs."


def test_python_decorated_function_docstring_lifted():
    src = '''import functools

@functools.cache
def expensive():
    """Compute something pricey."""
    return 42
'''
    chunks = list(chunk_file(src, repo="me/x", path="src/cache.py"))
    assert _doc_for("expensive", chunks) == "Compute something pricey."


# ---------- go: preceding line-comments ----------


def test_go_doc_comments_lifted():
    src = """package main

// Handle does the thing.
// It also does the other thing.
func Handle() int {
    return 1
}

func NoDoc() int {
    return 0
}
"""
    chunks = list(chunk_file(src, repo="me/x", path="main.go"))
    doc = _doc_for("Handle", chunks)
    assert doc is not None
    assert "Handle does the thing" in doc
    assert "It also does the other thing" in doc
    assert _doc_for("NoDoc", chunks) is None


def test_go_block_comment_lifted():
    src = """package main

/* Process consumes the channel. */
func Process(ch chan int) {
    return
}
"""
    chunks = list(chunk_file(src, repo="me/x", path="main.go"))
    assert _doc_for("Process", chunks) == "Process consumes the channel."


# ---------- rust: /// and /** style ----------


def test_rust_doc_comments_lifted():
    src = """/// Returns the answer.
/// Always 42.
pub fn answer() -> u32 {
    42
}
"""
    chunks = list(chunk_file(src, repo="me/x", path="src/lib.rs"))
    doc = _doc_for("answer", chunks)
    assert doc is not None
    assert "Returns the answer" in doc
    assert "Always 42" in doc


# ---------- scala: /** ... */ ----------


def test_scala_doc_comments_lifted():
    src = """package x

/** A circuit breaker for http4s clients. */
class CircuitedClient {
  def call(): Int = 1
}
"""
    chunks = list(chunk_file(src, repo="me/x", path="src/main/scala/X.scala"))
    doc = _doc_for("CircuitedClient", chunks)
    assert doc is not None
    assert "circuit breaker" in doc


# ---------- javascript: /** JSDoc */ ----------


def test_javascript_jsdoc_lifted():
    src = """/**
 * Resize an image to the given dimensions.
 * Returns the new buffer.
 */
function resize(buf, w, h) {
  return buf;
}
"""
    chunks = list(chunk_file(src, repo="me/x", path="src/img.js"))
    doc = _doc_for("resize", chunks)
    assert doc is not None
    assert "Resize an image to the given dimensions" in doc


# ---------- typescript ----------


def test_typescript_jsdoc_lifted():
    src = """/** Build an HTTP client. */
export class HttpClient {
  send(req: Request): Response { return new Response(); }
}
"""
    chunks = list(chunk_file(src, repo="me/x", path="src/http.ts"))
    doc = _doc_for("HttpClient", chunks)
    assert doc is not None
    assert "Build an HTTP client" in doc


# ---------- absence: no false-positive on unrelated preceding code ----------


def test_no_doc_when_previous_sibling_is_not_a_comment():
    src = """package main

func A() int { return 1 }

func B() int { return 2 }
"""
    chunks = list(chunk_file(src, repo="me/x", path="main.go"))
    # B's previous sibling is A's function_declaration, not a comment.
    assert _doc_for("B", chunks) is None


# ---------- truncation ----------


def test_long_docstring_is_truncated():
    long = "X " * 500
    src = f'''def foo():
    """{long}"""
    return 1
'''
    chunks = list(chunk_file(src, repo="me/x", path="src/foo.py"))
    doc = _doc_for("foo", chunks)
    assert doc is not None
    assert len(doc) <= 241  # MAX_LEADING_DOC_CHARS + ellipsis
    assert doc.endswith("…")
