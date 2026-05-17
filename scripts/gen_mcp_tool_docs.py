#!/usr/bin/env python3
"""Generate docs/mcp-tools.md from the @mcp.tool() decorated functions in
src/github_twin/mcp_server/server.py.

This is a one-shot script — not wired into CI. Re-run it whenever the tool
surface changes:

    uv run python scripts/gen_mcp_tool_docs.py

Static AST parsing means we capture signatures + docstrings deterministically
without spawning the MCP server, importing optional deps, or pulling a DB
connection. The cost is that the JSON schemas in the output are derived from
Python type annotations, not FastMCP's runtime schema generation — close
enough for a human-facing reference; the MCP registry sees the live schemas
when clients call `tools/list`.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_PATH = REPO_ROOT / "src" / "github_twin" / "mcp_server" / "server.py"
OUTPUT_PATH = REPO_ROOT / "docs" / "mcp-tools.md"


@dataclass
class Param:
    name: str
    annotation: str
    default: str | None  # rendered source ("None", "'all'", "5"), or None when required


@dataclass
class Tool:
    name: str
    summary: str
    description: str
    params: list[Param]
    returns: str


def _decorator_is_mcp_tool(dec: ast.expr) -> bool:
    """Match `@mcp.tool()` and `@mcp.tool` regardless of call form."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    return (
        isinstance(target, ast.Attribute)
        and target.attr == "tool"
        and isinstance(target.value, ast.Name)
        and target.value.id == "mcp"
    )


def _format_annotation(node: ast.expr | None) -> str:
    if node is None:
        return "Any"
    return ast.unparse(node)


def _format_default(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _split_docstring(doc: str) -> tuple[str, str]:
    """Return (summary, body). Summary is the first non-empty line."""
    stripped = doc.strip()
    if not stripped:
        return "", ""
    parts = stripped.split("\n\n", 1)
    summary = parts[0].strip()
    body = parts[1].strip() if len(parts) > 1 else ""
    return summary, body


def _extract_tools(tree: ast.AST) -> list[Tool]:
    tools: list[Tool] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(_decorator_is_mcp_tool(d) for d in node.decorator_list):
            continue

        doc = ast.get_docstring(node) or ""
        summary, body = _split_docstring(doc)

        args = node.args
        # Build defaults right-to-left: trailing positional defaults pad
        # from the end of the args list.
        defaults_padded: list[ast.expr | None] = [None] * (
            len(args.args) - len(args.defaults)
        ) + list(args.defaults)

        params = [
            Param(
                name=a.arg,
                annotation=_format_annotation(a.annotation),
                default=_format_default(d),
            )
            for a, d in zip(args.args, defaults_padded, strict=True)
            if a.arg != "self"
        ]

        tools.append(
            Tool(
                name=node.name,
                summary=summary,
                description=body,
                params=params,
                returns=_format_annotation(node.returns),
            )
        )
    return tools


def _escape_cell(text: str) -> str:
    """Escape `|` for GFM table cells. Backticks alone don't protect pipes."""
    return text.replace("|", "\\|")


def _render_param_table(params: list[Param]) -> str:
    if not params:
        return "_(no parameters)_\n"
    lines = [
        "| Parameter | Type | Default |",
        "| --- | --- | --- |",
    ]
    for p in params:
        annotation = _escape_cell(p.annotation)
        default = "_required_" if p.default is None else f"`{_escape_cell(p.default)}`"
        lines.append(f"| `{p.name}` | `{annotation}` | {default} |")
    return "\n".join(lines) + "\n"


def _render_tool(tool: Tool) -> str:
    parts: list[str] = [f"## `{tool.name}`\n"]
    if tool.summary:
        parts.append(tool.summary + "\n")
    parts.append(f"**Returns:** `{tool.returns}`\n")
    parts.append(_render_param_table(tool.params))
    if tool.description:
        parts.append("**Details:**\n\n```\n" + tool.description + "\n```\n")
    return "\n".join(parts)


def render(tools: list[Tool]) -> str:
    header = (
        "# MCP tool reference\n\n"
        "Auto-generated from `src/github_twin/mcp_server/server.py` via\n"
        "`scripts/gen_mcp_tool_docs.py`. Do not edit by hand — re-run the script\n"
        "after editing the server module.\n\n"
        f"Tools registered: **{len(tools)}**.\n\n"
        "## Table of contents\n\n"
        + "\n".join(f"- [`{t.name}`](#{t.name.replace('_', '-')})" for t in tools)
        + "\n\n---\n\n"
    )
    body = "\n---\n\n".join(_render_tool(t) for t in tools)
    return header + body


def main() -> int:
    if not SERVER_PATH.exists():
        print(f"server.py not found at {SERVER_PATH}", file=sys.stderr)
        return 1
    tree = ast.parse(SERVER_PATH.read_text())
    tools = _extract_tools(tree)
    if not tools:
        print("No @mcp.tool() functions detected.", file=sys.stderr)
        return 1
    OUTPUT_PATH.write_text(render(tools) + "\n")
    print(f"wrote {len(tools)} tools to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
