"""Walk an existing vault to find files we previously generated.

Files we wrote carry `generated: true` in their YAML frontmatter. The
export orchestrator uses this to prune stale auto-files (a rule that
got merged into another cluster and no longer exists, an author who left
the org, etc.) without ever touching hand-written notes.

The parser is minimal: we read at most the first ~30 lines of each
`.md`, look for a `---` fence, and key-value scan until the closing
`---`. No full YAML parser pulled in for this — the frontmatter we
write is flat scalars only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

GENERATED_MARKER = "generated: true"


def parse_frontmatter(text: str) -> dict[str, str]:
    """Return the flat string key/value map between the opening and
    closing `---` fences. Empty dict if no fence pair is found in the
    leading region of the document.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return out
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return {}


def format_frontmatter(fields: dict[str, Any]) -> str:
    """Render `fields` as a YAML frontmatter block (opening + closing
    `---`). Values are coerced to strings; `True`/`False` lowercased so
    Obsidian's boolean detection picks them up; None values dropped.
    """
    lines = ["---"]
    for key, value in fields.items():
        if value is None:
            continue
        rendered = ("true" if value else "false") if isinstance(value, bool) else str(value)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def list_generated_files(root: Path) -> set[Path]:
    """Recursively find every `.md` under `root` that carries our
    `generated: true` frontmatter marker.

    Hand-written notes in `scratch/` (or anywhere else) lack the marker
    and are returned exclusively by `iter_scratch_notes` instead.
    Missing root → empty set so first-run callers don't need to special-case.
    """
    if not root.exists():
        return set()
    out: set[Path] = set()
    for path in root.rglob("*.md"):
        try:
            head = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fm = parse_frontmatter(head)
        if fm.get("generated", "").lower() == "true":
            out.add(path)
    return out


def iter_scratch_notes(scratch_dir: Path) -> list[Path]:
    """Return every `.md` under `scratch_dir`, sorted. Skips files that
    carry `generated: true` (defense-in-depth — `scratch/` should never
    hold generated files, but ingesting one would loop).
    """
    if not scratch_dir.exists():
        return []
    out: list[Path] = []
    for path in sorted(scratch_dir.rglob("*.md")):
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fm = parse_frontmatter(body)
        if fm.get("generated", "").lower() == "true":
            continue
        out.append(path)
    return out
