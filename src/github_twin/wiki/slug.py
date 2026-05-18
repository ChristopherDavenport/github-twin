"""Stable slugs for vault filenames.

Each entity gets a deterministic, filesystem-safe filename so re-exports
land on the same path and content-hash idempotency works. Rule slugs
embed an 8-char content hash because two rule texts that normalize to
the same prefix (or rules whose first 60 chars are identical) would
otherwise collide.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# repo slugs keep underscores so the `owner__name` separator survives the
# normalize pass intact; everything else (dots, parens, etc.) still collapses.
_NON_ALNUM_KEEP_UNDERSCORE = re.compile(r"[^a-z0-9_]+")


def _normalize(text: str) -> str:
    return _NON_ALNUM.sub("-", text.lower()).strip("-")


def rule_slug(rule_text: str) -> str:
    """`{normalized-first-60-chars}-{sha1(text)[:8]}`. Stable across runs
    because both inputs are pure functions of the rule body."""
    base = _normalize(rule_text)[:60].rstrip("-")
    digest = hashlib.sha1(rule_text.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}" if base else digest


def profile_slug(login: str) -> str:
    """Profiles file under `profiles/{slug}.md`. Lowercase + non-alnum
    collapsed; never empty (callers pass a login or target name)."""
    norm = _normalize(login)
    return norm or "unnamed"


def file_page_relpath(repo: str, path: str) -> Path:
    """Vault-relative path for one per-file page. Mirrors the source
    repo + path so Obsidian's file tree maps 1:1 onto the codebase:
    `files/{owner__name}/{path}.md`. Filename includes the original
    file extension (e.g. `Foo.scala.md`) so two same-stem files in
    different languages don't collide and the language stays visible
    in the file tree."""
    return Path("files") / repo_slug(repo) / (path + ".md")


def repo_slug(full_name: str) -> str:
    """`owner/name` -> `owner__name`. Two underscores survive the
    normalize pass (regex preserves `_`) so the owner is visually
    separated from the repo name in filenames and `[[wikilinks]]`.
    Other non-alnum characters in either segment collapse to `-`."""
    base = full_name.lower().replace("/", "__")
    return _NON_ALNUM_KEEP_UNDERSCORE.sub("-", base).strip("-_")
