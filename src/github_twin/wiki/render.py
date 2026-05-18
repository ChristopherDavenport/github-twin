"""Render functions for the wiki vault.

Each `render_*` returns the **full file body** (frontmatter + markdown),
ready for byte-equality comparison against an existing file on disk.

Cross-references use Obsidian `[[wikilinks]]` for in-vault navigation
plus GitHub permalinks back to source. The frontmatter is intentionally
flat (no nested YAML), parseable by the minimal scanner in `scan.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from github_twin.wiki.scan import format_frontmatter
from github_twin.wiki.slug import file_page_relpath, profile_slug, repo_slug


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _frontmatter(fields: dict[str, Any]) -> str:
    """Wrap `fields` plus the canonical generator-stamp keys."""
    base = {"generated": True, "source": "github-twin"}
    base.update(fields)
    return format_frontmatter(base)


def render_rule(rule: dict[str, Any], *, target_name: str, kind: str) -> str:
    """Render one distilled rule. `rule` is a row from `q.list_rules`
    (keys: rule, language, examples, cluster_size, repos, urls).

    `kind` is the chunk discriminator (`'rule'` for review-derived,
    `'code_rule'` for commit-derived) so the page header can label the
    rule's provenance accurately.
    """
    rule_text = (rule.get("rule") or "").strip()
    language = rule.get("language") or "unspecified"
    cluster_size = rule.get("cluster_size") or 0
    examples = rule.get("examples") or []
    urls = rule.get("urls") or []
    repos = rule.get("repos") or []

    label = "Code pattern" if kind == "code_rule" else "Reviewer convention"
    fm = _frontmatter(
        {
            "type": "rule",
            "rule_kind": kind,
            "language": language,
            "target": target_name,
            "cluster_size": cluster_size,
        }
    )

    lines = [fm, "", f"# {label}", "", f"**{rule_text}**", ""]
    if cluster_size:
        lines.append(f"_Distilled from {cluster_size} examples._")
        lines.append("")
    lines.append(f"- Language: `{language}`")
    if repos:
        lines.append("- Repos: " + ", ".join(f"[[{repo_slug(r)}]]" for r in repos))
    lines.append("")

    if examples:
        lines.append("## Examples")
        lines.append("")
        for i, ex in enumerate(examples[:5]):
            quote = str(ex).strip().replace("\n", " ")
            if len(quote) > 300:
                quote = quote[:300].rstrip() + "…"
            link = ""
            if i < len(urls):
                link = f" ([source]({urls[i]}))"
            lines.append(f"> {quote}{link}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_profile(
    *,
    login: str,
    profile_md: str,
    target_name: str,
    n_samples: int,
    generated_at: str | None,
) -> str:
    """Render a developer voice/style profile. `profile_md` is the
    LLM-synthesized Markdown body from `developer_profile`."""
    fm = _frontmatter(
        {
            "type": "profile",
            "author": login,
            "target": target_name,
            "n_samples": n_samples,
            "generated_at": generated_at or _now_iso(),
        }
    )
    lines = [
        fm,
        "",
        f"# {login}",
        "",
        f"_Voice & review profile distilled from {n_samples} review comments._",
        "",
        profile_md.strip(),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_profile_placeholder(
    *,
    login: str,
    target_name: str,
    n_samples: int,
    reason: str,
) -> str:
    """A profile page that records why no LLM synthesis ran (no key, no
    samples, etc.). Useful so the vault shape stays predictable across
    runs even when API access is intermittent."""
    fm = _frontmatter(
        {
            "type": "profile",
            "author": login,
            "target": target_name,
            "n_samples": n_samples,
            "placeholder": True,
        }
    )
    body = (
        f"_No profile has been synthesized yet ({reason})._\n\n"
        f"Run `gt sync` with an LLM backend configured to populate this page."
    )
    return "\n".join([fm, "", f"# {login}", "", body, ""]).rstrip() + "\n"


def render_repo(
    *,
    target_name: str,
    overview: dict[str, Any],
) -> str:
    """Render the repo overview page. `overview` is `q.repo_overview`'s
    return shape (keys: full_name, counts, top_paths, top_authors,
    default_branch, head_sha, pushed_at, archived, fork)."""
    full_name = overview["full_name"]
    counts = overview.get("counts", {})
    top_paths = overview.get("top_paths", [])
    top_authors = overview.get("top_authors", [])

    fm = _frontmatter(
        {
            "type": "repo",
            "repo": full_name,
            "target": target_name,
            "default_branch": overview.get("default_branch"),
            "head_sha": overview.get("head_sha"),
            "pushed_at": overview.get("pushed_at"),
            "archived": overview.get("archived"),
            "fork": overview.get("fork"),
        }
    )
    lines = [fm, "", f"# {full_name}", ""]
    if overview.get("archived"):
        lines.append("> :warning: Archived")
        lines.append("")
    if overview.get("fork"):
        lines.append("> :information_source: Fork")
        lines.append("")
    lines.append(f"- GitHub: <https://github.com/{full_name}>")
    if overview.get("default_branch"):
        lines.append(f"- Default branch: `{overview['default_branch']}`")
    if overview.get("head_sha"):
        lines.append(f"- HEAD (last indexed): `{overview['head_sha'][:12]}`")
    if overview.get("pushed_at"):
        lines.append(f"- Last pushed: `{overview['pushed_at']}`")
    lines.append("")

    if counts:
        lines.append("## Indexed artifacts")
        lines.append("")
        for kind in sorted(counts):
            lines.append(f"- {kind}: {counts[kind]}")
        lines.append("")

    if top_authors:
        lines.append("## Top contributors")
        lines.append("")
        for login, n in top_authors:
            lines.append(f"- [[{profile_slug(login)}|{login}]] — {n} artifacts")
        lines.append("")

    if top_paths:
        lines.append("## Top files (by chunk count)")
        lines.append("")
        # Wikilink into the per-file page; the file page itself carries
        # the GitHub link + per-chunk summaries. Using `.with_suffix("")`
        # so the wikilink target is the vault-relative stem (Obsidian
        # convention).
        for path, n in top_paths:
            page = file_page_relpath(full_name, path).with_suffix("")
            lines.append(f"- [[{page.as_posix()}|{path}]] — {n} chunks")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_file(
    *,
    target_name: str,
    repo: str,
    path: str,
    language: str | None,
    chunks: list[dict[str, Any]],
) -> str:
    """Per-file page rendering. `chunks` is an ordered list of dicts
    with keys: `symbol_name`, `node_kind`, `summary`, `start_line`,
    `end_line` — all derived from `chunk.context` + the AST chunker's
    metadata. Body is summary-driven: code text itself stays in
    GitHub, the vault carries the LLM's NL distillation.
    """
    fm = _frontmatter(
        {
            "type": "file",
            "repo": repo,
            "path": path,
            "target": target_name,
            "language": language,
            "chunk_count": len(chunks),
        }
    )
    gh_url = f"https://github.com/{repo}/blob/HEAD/{path}"
    repo_link = f"[[{repo_slug(repo)}|{repo}]]"
    lines: list[str] = [
        fm,
        "",
        f"# {path}",
        "",
        f"{repo_link} · {language or 'unknown'} · {len(chunks)} chunks · [GitHub]({gh_url})",
        "",
        "## Chunks",
        "",
    ]
    for c in chunks:
        symbol = c.get("symbol_name")
        node_kind = c.get("node_kind")
        # Per-chunk heading: prefer `symbol (node_kind)`, fall back to
        # whichever metadata is available, finally to `chunk` so the
        # outline is always navigable.
        if symbol and node_kind:
            heading = f"`{symbol}` ({node_kind})"
        elif symbol:
            heading = f"`{symbol}`"
        elif node_kind:
            heading = f"({node_kind})"
        else:
            heading = "chunk"
        # Append line range when both ends are known — gives a visible
        # anchor without dragging in the source body.
        start = c.get("start_line")
        end = c.get("end_line")
        if start is not None and end is not None:
            heading += f" — lines {start}–{end}"
        lines.append(f"### {heading}")
        lines.append("")
        summary = (c.get("summary") or "").strip()
        if summary:
            lines.append(summary)
        else:
            lines.append("_(no summary yet — run `gt summarize` to populate)_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_index(
    *,
    targets: list[dict[str, Any]],
    counts: dict[str, int],
) -> str:
    """Top-level vault entry page. `targets` is a list of
    `{kind, name, id}`; `counts` aggregates the file totals
    (`rules`, `profiles`, `repos`).
    """
    fm = _frontmatter({"type": "index", "generated_at": _now_iso()})
    lines = [
        fm,
        "",
        "# github-twin wiki",
        "",
        "Auto-generated from the SQLite corpus. Edit the source data (run "
        "`gt sync`, `gt distill`, drop notes in [[scratch/README]]) — do "
        "not hand-edit files marked `generated: true`.",
        "",
        "## Targets",
        "",
    ]
    for tg in targets:
        lines.append(f"- **{tg['kind']}** `{tg['name']}` (id {tg['id']})")
    lines.append("")
    lines.append("## Sections")
    lines.append("")
    lines.append(f"- [[rules/_index|Rules]] ({counts.get('rules', 0)} files)")
    lines.append(f"- [[profiles/_index|Profiles]] ({counts.get('profiles', 0)} files)")
    lines.append(f"- [[repos/_index|Repos]] ({counts.get('repos', 0)} files)")
    lines.append(f"- Files ({counts.get('files', 0)} files; navigate via repo pages)")
    lines.append("- [[scratch/README|Scratch (round-trip notes)]]")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_section_index(
    *,
    section: str,
    entries: list[tuple[str, str]],
) -> str:
    """Per-section index page. `entries` is `[(label, slug), ...]` where
    `slug` is the bare filename minus `.md`. Used for `rules/_index.md`
    and the rest so Obsidian users have a quick jump-list."""
    fm = _frontmatter({"type": "section_index", "section": section})
    lines = [fm, "", f"# {section.title()}", ""]
    if not entries:
        lines.append(f"_No {section} indexed yet._")
        lines.append("")
    for label, slug in entries:
        lines.append(f"- [[{slug}|{label}]]")
    return "\n".join(lines).rstrip() + "\n"


def render_scratch_readme() -> str:
    """Written once into `<vault>/scratch/README.md`. Explains the
    round-trip and what the user can drop in. We DO mark this with
    `generated: true` so the next `gt wiki export` can refresh it if we
    change the explanation — but the scratch ingester skips it (path is
    `README.md`, hash is stable, but more importantly the frontmatter
    marker excludes it from `iter_scratch_notes`).
    """
    fm = _frontmatter({"type": "scratch_readme"})
    body = (
        "# Scratch — the round-trip inbox\n"
        "\n"
        "Drop any `.md` file in this folder and it will be ingested as a "
        "`kind='note'` artifact on the next `gt sync`. From then on it shows "
        "up in hybrid search, find_review_comments, find_code, and every "
        "other MCP retrieval tool.\n"
        "\n"
        "## Why this exists\n"
        "\n"
        "Following Karpathy's LLM-knowledge-base pattern: any analysis the "
        "model writes (notes, follow-ups, glossaries, decision logs) belongs "
        "back in the corpus so the next query has access to it. The vault "
        "becomes the durable memory layer; the SQLite DB is the index that "
        "makes it searchable.\n"
        "\n"
        "## Tips\n"
        "\n"
        "- Filenames are arbitrary; the artifact `external_id` is the SHA-256 "
        "  of the file contents, so editing a note creates a new chunk row "
        "  and deletes the old one cleanly.\n"
        "- Hand-write or have the LLM write; both work.\n"
        "- Delete a file → the next `gt sync` removes its artifact + chunks "
        "  + vectors.\n"
        "- Auto-generated files anywhere in the vault carry "
        "  `generated: true` in their YAML frontmatter and are excluded from "
        "  this round-trip path; you can hand-edit one by stripping that "
        "  flag (or moving it out of `scratch/`).\n"
    )
    return fm + "\n\n" + body
