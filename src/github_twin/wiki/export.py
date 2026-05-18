"""Orchestrator: materialize the SQLite corpus as a markdown vault.

`export_wiki` is idempotent — it computes the full expected file set,
writes only files whose body differs (preserves mtime for Obsidian's
file watcher), and prunes generated files that fell out of the expected
set. Hand-edited files (no `generated: true` frontmatter) are never
touched.

Profile pages are LLM-backed. When `profile_llm` is provided we call
`developer_profile` (which caches in `developer_profile_cache`), so
repeat exports hit the cache and don't burn tokens. When it's `None`
(e.g. tests, no API key, --skip-llm), we still emit a placeholder page
for each author so the vault shape stays predictable.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from github_twin.config import Config
from github_twin.eval.llm import TextLLM
from github_twin.store import queries as q
from github_twin.target import Target, load_targets
from github_twin.wiki.render import (
    render_file,
    render_index,
    render_profile,
    render_profile_placeholder,
    render_repo,
    render_rule,
    render_scratch_readme,
    render_section_index,
)
from github_twin.wiki.scan import list_generated_files, parse_frontmatter
from github_twin.wiki.slug import file_page_relpath, profile_slug, repo_slug, rule_slug

Reporter = Callable[[str], None]


def _noop(_: str) -> None:
    return None


def resolve_vault_root(cfg: Config, out: Path | None = None) -> Path:
    """Pick the vault root: explicit `out` > `cfg.wiki.out` > default
    under `cfg.paths.data_dir / 'wiki'`."""
    if out is not None:
        return Path(out)
    if cfg.wiki.out is not None:
        return Path(cfg.wiki.out)
    return cfg.paths.data_dir / "wiki"


def _rules_for_target(conn: sqlite3.Connection, *, target_id: int) -> list[tuple[Path, str]]:
    """Return `[(relative_path, body), ...]` for every rule under `target_id`,
    across both rule kinds. Relative path layout:
    `rules/{language or _unspecified}/{slug}.md`.
    """
    out: list[tuple[Path, str]] = []
    target_row = q.get_target_by_id(conn, target_id)
    target_name = target_row["name"] if target_row else f"target-{target_id}"
    for kind in ("rule", "code_rule"):
        rules = q.list_rules(conn, target_id=target_id, chunk_kind=kind, limit=10_000)
        for rule in rules:
            language = rule.get("language") or "_unspecified"
            slug = rule_slug(rule.get("rule") or "")
            rel = Path("rules") / language / f"{slug}.md"
            body = render_rule(rule, target_name=target_name, kind=kind)
            out.append((rel, body))
    return out


def _profile_for_author(
    conn: sqlite3.Connection,
    *,
    target: Target,
    author_login: str | None,
    profile_llm: TextLLM | None,
    cfg: Config,
) -> tuple[Path, str] | None:
    """Build the profile entry for one author. Returns None if no
    review-comment samples exist (no point emitting an empty page).

    For org-mode targets `author_login` is the GitHub login; for
    user-mode it's `None` and the page is keyed by the target name.
    """
    from github_twin.mcp_server.tools import developer_profile  # avoid cycle at module load

    display_login = author_login or target.name
    rel = Path("profiles") / f"{profile_slug(display_login)}.md"

    if profile_llm is None:
        # No LLM — surface whatever is already cached, else placeholder.
        # We don't know the cache key without _profile_cache_key, so just
        # write a placeholder; a later run with an LLM will replace it.
        n_samples = len(
            q.recent_review_comments(
                conn,
                author_login=author_login,
                target_id=target.id,
                limit=cfg.summarize.profile_max_tokens,
            )
        )
        if n_samples == 0:
            return None
        body = render_profile_placeholder(
            login=display_login,
            target_name=target.name,
            n_samples=n_samples,
            reason="no LLM backend configured for wiki export",
        )
        return rel, body

    result = developer_profile(
        conn,
        profile_llm,
        author_login=author_login,
        target=target.name,
        n_samples=50,
    )
    if not result.get("profile_md"):
        return None
    body = render_profile(
        login=display_login,
        profile_md=result["profile_md"],
        target_name=target.name,
        n_samples=int(result.get("n_samples", 0)),
        generated_at=result.get("generated_at"),
    )
    return rel, body


def _profiles_for_target(
    conn: sqlite3.Connection,
    *,
    target: Target,
    profile_llm: TextLLM | None,
    cfg: Config,
    report: Reporter,
) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    if target.is_user:
        entry = _profile_for_author(
            conn,
            target=target,
            author_login=None,
            profile_llm=profile_llm,
            cfg=cfg,
        )
        if entry is not None:
            out.append(entry)
        return out
    # org-mode / repo-mode: one page per author with enough samples
    assert target.id is not None
    authors = q.list_authors_for_target(conn, target_id=target.id, min_review_comments=1)
    for login, _count in authors:
        entry = _profile_for_author(
            conn,
            target=target,
            author_login=login,
            profile_llm=profile_llm,
            cfg=cfg,
        )
        if entry is not None:
            out.append(entry)
    if authors:
        report(f"  profiles: {len(out)} (from {len(authors)} authors with review comments)")
    return out


def _files_for_target(
    conn: sqlite3.Connection, *, target_id: int, target_name: str
) -> list[tuple[Path, str]]:
    """Emit one page per distinct (repo, path) under `target_id`.

    All chunks for a given file are grouped in source order (chunk.id
    is the order they were ingested in). Each row carries the metadata
    the AST chunker stores: `symbol_name`, `node_kind`, `summary`, and
    the line range from `chunk.context_json`. Single SQL query +
    in-Python group-by — keeps idempotent re-exports cheap even at
    1000+ files.
    """
    rows = conn.execute(
        "SELECT a.repo, "
        "json_extract(c.context_json, '$.path') AS path, "
        "c.id AS chunk_id, c.language, c.symbol_name, c.node_kind, c.summary, "
        "json_extract(c.context_json, '$.start_line') AS start_line, "
        "json_extract(c.context_json, '$.end_line') AS end_line "
        "FROM chunk c JOIN artifact a ON a.id = c.artifact_id "
        "WHERE c.kind IN ('code', 'file') "
        "AND a.repo IS NOT NULL "
        "AND json_extract(c.context_json, '$.path') IS NOT NULL "
        "AND a.target_id = ? "
        "ORDER BY a.repo, json_extract(c.context_json, '$.path'), c.id",
        (target_id,),
    ).fetchall()

    out: list[tuple[Path, str]] = []
    if not rows:
        return out

    current_repo: str | None = None
    current_path: str | None = None
    current_language: str | None = None
    current_chunks: list[dict[str, object]] = []

    def _flush() -> None:
        if current_repo is None or current_path is None:
            return
        rel = file_page_relpath(current_repo, current_path)
        body = render_file(
            target_name=target_name,
            repo=current_repo,
            path=current_path,
            language=current_language,
            chunks=current_chunks,
        )
        out.append((rel, body))

    for r in rows:
        if r["repo"] != current_repo or r["path"] != current_path:
            _flush()
            current_repo = r["repo"]
            current_path = r["path"]
            current_language = r["language"]
            current_chunks = []
        current_chunks.append(
            {
                "symbol_name": r["symbol_name"],
                "node_kind": r["node_kind"],
                "summary": r["summary"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
            }
        )
    _flush()
    return out


def _repos_for_target(conn: sqlite3.Connection, *, target: Target) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    if target.id is None:
        return out
    for repo_row in q.list_repos(
        conn, target_id=target.id, include_archived=True, include_forks=True
    ):
        full_name = repo_row["full_name"]
        overview = q.repo_overview(conn, target_id=target.id, full_name=full_name)
        rel = Path("repos") / f"{repo_slug(full_name)}.md"
        body = render_repo(target_name=target.name, overview=overview)
        out.append((rel, body))
    return out


def _section_indexes(
    rules: list[tuple[Path, str]],
    profiles: list[tuple[Path, str]],
    repos: list[tuple[Path, str]],
) -> list[tuple[Path, str]]:
    """Emit `_index.md` per section listing every entry. Obsidian uses
    these as natural jump-pages; the index renders even when the
    section directory has subfolders (rules are by-language)."""

    def _entries(paths: list[Path]) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        for p in sorted(paths):
            # label = filename minus extension; slug = bare stem for the wikilink
            items.append((p.stem.replace("-", " "), p.stem))
        return items

    out: list[tuple[Path, str]] = []
    out.append(
        (
            Path("rules") / "_index.md",
            render_section_index(
                section="rules",
                entries=_entries([p for p, _ in rules]),
            ),
        )
    )
    out.append(
        (
            Path("profiles") / "_index.md",
            render_section_index(
                section="profiles",
                entries=_entries([p for p, _ in profiles]),
            ),
        )
    )
    out.append(
        (
            Path("repos") / "_index.md",
            render_section_index(
                section="repos",
                entries=_entries([p for p, _ in repos]),
            ),
        )
    )
    return out


def export_wiki(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    out: Path | None = None,
    target: int | str | None = None,
    profile_llm: TextLLM | None = None,
    report: Reporter = _noop,
) -> dict[str, int]:
    """Materialize the corpus as a markdown vault. Returns
    `{written, unchanged, removed}`.

    `target=None` exports every target; otherwise narrows to one
    (matches by id when int, by name when str). `profile_llm=None`
    skips LLM synthesis and emits placeholder profile pages.
    """
    vault_root = resolve_vault_root(cfg, out)
    report(f"vault root: {vault_root}")

    if target is None:
        targets = load_targets(conn)
    else:
        all_targets = load_targets(conn)
        if isinstance(target, int):
            targets = [t for t in all_targets if t.id == target]
        else:
            targets = [t for t in all_targets if t.name == target]
        if not targets:
            raise ValueError(f"No target matches {target!r}")

    rules_all: list[tuple[Path, str]] = []
    profiles_all: list[tuple[Path, str]] = []
    repos_all: list[tuple[Path, str]] = []
    files_all: list[tuple[Path, str]] = []

    for tg in targets:
        assert tg.id is not None
        report(f"target: {tg.kind} {tg.name} (id {tg.id})")
        rules_all.extend(_rules_for_target(conn, target_id=tg.id))
        profiles_all.extend(
            _profiles_for_target(
                conn,
                target=tg,
                profile_llm=profile_llm,
                cfg=cfg,
                report=report,
            )
        )
        repos_all.extend(_repos_for_target(conn, target=tg))
        files_all.extend(_files_for_target(conn, target_id=tg.id, target_name=tg.name))

    expected: dict[Path, str] = {}
    for rel, body in rules_all + profiles_all + repos_all + files_all:
        expected[vault_root / rel] = body
    for rel, body in _section_indexes(rules_all, profiles_all, repos_all):
        expected[vault_root / rel] = body

    counts = {
        "rules": len(rules_all),
        "profiles": len(profiles_all),
        "repos": len(repos_all),
        "files": len(files_all),
    }
    expected[vault_root / "index.md"] = render_index(
        targets=[{"kind": t.kind, "name": t.name, "id": t.id} for t in targets],
        counts=counts,
    )
    # scratch/README.md is treated like any other generated file so the
    # explainer can evolve over time. The ingester ignores it via
    # frontmatter scan.
    expected[vault_root / "scratch" / "README.md"] = render_scratch_readme()

    existing = list_generated_files(vault_root)

    written = 0
    unchanged = 0
    adopted = 0
    for path, body in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                current = path.read_text(encoding="utf-8")
            except OSError:
                current = ""
            if current == body:
                unchanged += 1
                continue
            # Hand-edit guard: a file that no longer carries
            # `generated: true` has been adopted by the user. Leave it
            # alone — even when the DB has a fresher canonical body.
            fm = parse_frontmatter(current)
            if fm.get("generated", "").lower() != "true":
                adopted += 1
                continue
        path.write_text(body, encoding="utf-8")
        written += 1

    removed = 0
    expected_paths = set(expected.keys())
    for stale in existing - expected_paths:
        try:
            stale.unlink()
            removed += 1
        except OSError:
            pass

    report(
        f"wiki: {written} written, {unchanged} unchanged, "
        f"{removed} removed, {adopted} adopted (hand-edited)"
    )
    return {
        "written": written,
        "unchanged": unchanged,
        "removed": removed,
        "adopted": adopted,
    }
