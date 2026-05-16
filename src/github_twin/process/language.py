"""Language inference from file paths.

Pygments has the most comprehensive extension map. We only return canonical
short names (the first alias) and skip lexers that aren't really programming
languages (text, json, html-templates, etc. for the *purpose* of code-style RAG).
"""

from __future__ import annotations

from pathlib import PurePosixPath

from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound

# Lexer name → canonical language tag. Everything else: use the first lexer alias.
_OVERRIDES = {
    "Python": "python",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "TSX": "tsx",
    "JSX": "jsx",
    "Go": "go",
    "Rust": "rust",
    "Ruby": "ruby",
    "Java": "java",
    "Kotlin": "kotlin",
    "Swift": "swift",
    "C": "c",
    "C++": "cpp",
    "C#": "csharp",
    "Bash": "shell",
    "Zsh": "shell",
    "PowerShell": "powershell",
    "SQL": "sql",
    "HTML": "html",
    "CSS": "css",
    "SCSS": "scss",
    "YAML": "yaml",
    "TOML": "toml",
}

# Non-code lexers we explicitly drop. These commonly match but aren't useful
# as style exemplars and tend to dominate review-comment language counts.
_DROP_LEXERS = {"Text only", "JSON", "JSON5", "Markdown", "reStructuredText", "INI"}


def language_for_path(path: str) -> str | None:
    """Return a short language tag for a file path, or None if it isn't code we'd index."""
    if not path:
        return None
    p = PurePosixPath(path)
    name = p.name
    try:
        lexer = get_lexer_for_filename(name)
    except ClassNotFound:
        return None
    if lexer.name in _DROP_LEXERS:
        return None
    if lexer.name in _OVERRIDES:
        return _OVERRIDES[lexer.name]
    if lexer.aliases:
        return str(lexer.aliases[0])
    return str(lexer.name).lower()
