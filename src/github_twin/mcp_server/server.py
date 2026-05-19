"""MCP server entry point for github-twin.

Tools exposed:
  - find_review_comments(diff_hunk, language?, repo?, author_login?, scope?, k=5)
  - find_style_examples(query, language?, repo?, author_login?, scope?, k=5)
  - find_code(query, language?, repo?, path_glob?, k=5)   [org-mode: files at HEAD]
  - find_applicable_rules(query, language?, repo?, author_login?, k=5)
  - predict_review_outcome(diff_or_summary, ..., k=20)
  - summarize_review_patterns(language?, limit=20)
  - house_rules(language?, repo?, author_login?, scope?, limit=50)     -> Markdown block of distilled rules
  - developer_profile(author_login?, language?, repo?, scope?, n_samples=50, force_refresh=False)  -> cached voice profile
  - sync(since?)  -> runs ingest + embed, returns counts

`scope` on the retrieval tools is sugar over `repo` / `author_login`
filters: `"personal"` / `"project"` / `"all"` (default). Explicit
kwargs always win over scope.

Runs over stdio so it can be wired into Claude Code via ~/.claude.json.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import anyio
from anyio.from_thread import run as run_from_thread
from mcp.server.fastmcp import Context, FastMCP
from opentelemetry.trace import Span

from github_twin.config import Config, load_config
from github_twin.embed import make_embedder
from github_twin.eval.llm import make_text_llm
from github_twin.mcp_server import bootstrap as bootstrap_mod
from github_twin.mcp_server import tools as t
from github_twin.mcp_server.bootstrap import BootstrapSpec
from github_twin.mcp_server.tools import Scope
from github_twin.observability import set_safe_attributes, tracer
from github_twin.pipeline import run_embed, run_ingest
from github_twin.store import queries as q
from github_twin.store.db import db_session
from github_twin.store.query_expansion import expander_from_config
from github_twin.store.vector_store import make_vector_store

log = logging.getLogger(__name__)


def _record_result_count(span: Span, result: Any) -> None:
    """Tag the active span with how much the tool returned, for quick
    "did this tool find anything?" filtering in trace UIs."""
    if isinstance(result, list):
        span.set_attribute("gh_twin.result.count", len(result))
    elif isinstance(result, dict):
        # `predict_review_outcome` returns a structured dict; the most useful
        # post-hoc dimensions are the prediction and how confident it was.
        if "prediction" in result:
            span.set_attribute("gh_twin.result.prediction", str(result.get("prediction")))
        if "confidence" in result:
            span.set_attribute("gh_twin.result.confidence", float(result["confidence"]))
        if "n_pulled" in result:
            span.set_attribute("gh_twin.result.n_pulled", int(result["n_pulled"]))


def run(config_path: Path | None = None) -> None:
    cfg = load_config(config_path)
    # The MCP server holds its DB connection for the process lifetime,
    # so we wrap the whole loop in `db_session` to guarantee a clean
    # `close()` if the server exits normally (Claude Code disconnects,
    # SIGTERM, etc.). FastMCP's `mcp.run()` blocks until the transport
    # tears down; control returns and the `finally` in `db_session`
    # runs the close().
    with db_session(cfg.paths.db_path, cfg.embed.dim) as conn:
        _serve(cfg, conn)


def _serve(cfg: Config, conn: sqlite3.Connection) -> None:
    embedder = make_embedder(cfg.embed)
    store = make_vector_store(conn, backend=cfg.vector_store.backend, dim=cfg.embed.dim)
    expander = expander_from_config(cfg)
    # Captured once at startup so each tool call falls back to the same
    # default unless the caller passes an explicit `recency_half_life_days`.
    cfg_recency = cfg.retrieval.recency_half_life_days

    def _resolve_recency(override: float | None) -> float | None:
        """None from the caller falls through to the cfg default; any
        explicit value (including 0, which disables decay) wins."""
        return override if override is not None else cfg_recency

    # LLM used by `developer_profile`. Lazy: only constructed if the tool
    # is called (FastMCP's `@mcp.tool` registration doesn't invoke the
    # function until a client calls it), so a server that never sees a
    # profile request never imports the LLM SDK.
    _llm_cache: dict[str, Any] = {}

    def _get_llm() -> Any:
        if "llm" not in _llm_cache:
            _llm_cache["llm"] = make_text_llm(
                claude_model=cfg.summarize.claude_model,
                gemini_model=cfg.summarize.gemini_model,
                ollama_model=cfg.summarize.ollama_model,
                prefer=cfg.summarize.backend,
            )
        return _llm_cache["llm"]

    mcp = FastMCP("github-twin")

    @mcp.tool()
    def find_review_comments(
        diff_hunk: str,
        language: str | None = None,
        repo: str | None = None,
        author_login: str | None = None,
        target: str | None = None,
        scope: Scope = "all",
        k: int = 5,
        recency_half_life_days: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find past review comments on diffs similar to `diff_hunk`.

        Args:
            diff_hunk: The new code under review (a unified-diff hunk works best).
            language: Optional language filter (e.g. 'python', 'go', 'typescript').
            repo: Optional 'owner/name' filter (org-mode).
            author_login: Optional GH login to narrow to a single reviewer.
            target: Optional target name. Unset = search across every target
                (coalesce + dedup). Each returned hit carries its target.
            scope: 'personal' resolves to the unique user-mode target;
                'project' to the unique repo-mode target; 'all' (default)
                is unscoped. Explicit kwargs win.
            k: Max results to return.
            recency_half_life_days: Override the server's recency decay
                half-life for this call. Unset = use the configured
                `retrieval.recency_half_life_days` default (often None,
                meaning no decay).
        """
        with tracer().start_as_current_span("mcp.tool.find_review_comments") as span:
            set_safe_attributes(
                span,
                **{
                    "gh_twin.tool.k": k,
                    "gh_twin.tool.scope": scope,
                    "gh_twin.tool.target": target,
                    "gh_twin.tool.diff_hunk_chars": len(diff_hunk or ""),
                    "gh_twin.filter.language": language,
                    "gh_twin.filter.repo": repo,
                    "gh_twin.filter.author_login": author_login,
                },
            )
            result = t.find_review_comments(
                conn,
                embedder,
                store,
                diff_hunk=diff_hunk,
                language=language,
                repo=repo,
                author_login=author_login,
                target=target,
                scope=scope,
                k=k,
                expander=expander,
                recency_half_life_days=_resolve_recency(recency_half_life_days),
            )
            _record_result_count(span, result)
            return result

    @mcp.tool()
    def find_style_examples(
        query: str,
        language: str | None = None,
        repo: str | None = None,
        author_login: str | None = None,
        target: str | None = None,
        scope: Scope = "all",
        k: int = 5,
        recency_half_life_days: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find code that matches a description, for style reference.

        Args:
            query: Natural-language description of what you're trying to write.
            language: Optional language filter.
            repo: Optional 'owner/name' filter.
            author_login: Optional GH login to scope to one author.
            target: Optional target name. Unset = coalesce across every target.
            scope: 'personal' / 'project' / 'all' — see find_review_comments.
            k: Max results to return.
            recency_half_life_days: Override the server's recency decay
                half-life for this call. Unset = use the configured default.
        """
        with tracer().start_as_current_span("mcp.tool.find_style_examples") as span:
            set_safe_attributes(
                span,
                **{
                    "gh_twin.tool.k": k,
                    "gh_twin.tool.scope": scope,
                    "gh_twin.tool.target": target,
                    "gh_twin.tool.query_chars": len(query or ""),
                    "gh_twin.filter.language": language,
                    "gh_twin.filter.repo": repo,
                    "gh_twin.filter.author_login": author_login,
                },
            )
            result = t.find_style_examples(
                conn,
                embedder,
                store,
                query=query,
                language=language,
                repo=repo,
                author_login=author_login,
                target=target,
                scope=scope,
                k=k,
                expander=expander,
                recency_half_life_days=_resolve_recency(recency_half_life_days),
            )
            _record_result_count(span, result)
            return result

    @mcp.tool()
    def find_code(
        query: str,
        language: str | None = None,
        repo: str | None = None,
        path_glob: str | None = None,
        node_kind: str | None = None,
        target: str | None = None,
        k: int = 5,
        recency_half_life_days: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find source snippets at HEAD across every indexed target.

        Args:
            query: Natural-language description or code-shape to match.
            language: Optional language filter (e.g. 'scala', 'go').
            repo: Optional 'owner/name' filter.
            path_glob: Optional fnmatch glob applied to file path.
            node_kind: Optional tree-sitter AST node type.
            target: Optional target name. Unset = coalesce across all
                targets with cross-target dedup. Each hit carries its target.
            k: Max results to return.
            recency_half_life_days: Accepted for API symmetry; effectively
                a no-op for file-at-HEAD chunks (those artifacts are
                excluded from recency decay).
        """
        with tracer().start_as_current_span("mcp.tool.find_code") as span:
            set_safe_attributes(
                span,
                **{
                    "gh_twin.tool.k": k,
                    "gh_twin.tool.target": target,
                    "gh_twin.tool.query_chars": len(query or ""),
                    "gh_twin.filter.language": language,
                    "gh_twin.filter.repo": repo,
                    "gh_twin.filter.path_glob": path_glob,
                    "gh_twin.filter.node_kind": node_kind,
                },
            )
            result = t.find_code(
                conn,
                embedder,
                store,
                query=query,
                language=language,
                repo=repo,
                path_glob=path_glob,
                node_kind=node_kind,
                target=target,
                k=k,
                expander=expander,
                recency_half_life_days=_resolve_recency(recency_half_life_days),
            )
            _record_result_count(span, result)
            return result

    @mcp.tool()
    def find_applicable_rules(
        query: str,
        language: str | None = None,
        repo: str | None = None,
        author_login: str | None = None,
        target: str | None = None,
        k: int = 5,
        recency_half_life_days: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find distilled code-pattern rules that apply to a coding task.

        Args:
            query: What you're about to write or change.
            language: Optional language filter.
            repo: Optional 'owner/name' filter.
            author_login: Optional GH login to scope to personal style.
            target: Optional target name. Unset = coalesce across all targets.
            k: Max results to return.
            recency_half_life_days: Accepted for API symmetry; effectively
                a no-op for rule chunks (rule artifacts are excluded from
                recency decay since they're synthesized recently from
                heterogeneous-age sources).
        """
        with tracer().start_as_current_span("mcp.tool.find_applicable_rules") as span:
            set_safe_attributes(
                span,
                **{
                    "gh_twin.tool.k": k,
                    "gh_twin.tool.target": target,
                    "gh_twin.tool.query_chars": len(query or ""),
                    "gh_twin.filter.language": language,
                    "gh_twin.filter.repo": repo,
                    "gh_twin.filter.author_login": author_login,
                },
            )
            result = t.find_applicable_rules(
                conn,
                embedder,
                store,
                query=query,
                language=language,
                repo=repo,
                author_login=author_login,
                target=target,
                k=k,
                expander=expander,
                recency_half_life_days=_resolve_recency(recency_half_life_days),
            )
            _record_result_count(span, result)
            return result

    @mcp.tool()
    def predict_review_outcome(
        diff_or_summary: str,
        language: str | None = None,
        repo: str | None = None,
        author_login: str | None = None,
        target: str | None = None,
        k: int = 20,
    ) -> dict[str, Any]:
        """Predict how a candidate PR would be reviewed.

        Args:
            diff_or_summary: A diff, a PR title+body, or a free-form description.
            language: Optional language filter.
            repo: Optional 'owner/name' filter.
            author_login: Optional reviewer login.
            target: Optional target name. Unset = coalesce with dedup.
            k: How many similar past PRs to pull.
        """
        with tracer().start_as_current_span("mcp.tool.predict_review_outcome") as span:
            set_safe_attributes(
                span,
                **{
                    "gh_twin.tool.k": k,
                    "gh_twin.tool.target": target,
                    "gh_twin.tool.input_chars": len(diff_or_summary or ""),
                    "gh_twin.filter.language": language,
                    "gh_twin.filter.repo": repo,
                    "gh_twin.filter.author_login": author_login,
                },
            )
            result = t.predict_review_outcome(
                conn,
                embedder,
                store,
                diff_or_summary=diff_or_summary,
                language=language,
                repo=repo,
                author_login=author_login,
                target=target,
                k=k,
            )
            _record_result_count(span, result)
            return result

    @mcp.tool()
    def summarize_review_patterns(
        language: str | None = None,
        target: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return distilled review rules.

        Args:
            language: Optional filter (e.g. 'scala', 'go').
            target: Optional target name. Unset = coalesce across all targets.
            limit: Max rules to return.
        """
        with tracer().start_as_current_span("mcp.tool.summarize_review_patterns") as span:
            set_safe_attributes(
                span,
                **{
                    "gh_twin.tool.limit": limit,
                    "gh_twin.tool.target": target,
                    "gh_twin.filter.language": language,
                },
            )
            result = t.summarize_review_patterns(
                conn, language=language, target=target, limit=limit
            )
            _record_result_count(span, result)
            return result

    @mcp.tool()
    def house_rules(
        language: str | None = None,
        repo: str | None = None,
        author_login: str | None = None,
        target: str | None = None,
        scope: Scope = "all",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return all distilled rules as a single Markdown block.

        Args:
            language: Strongly recommended in single-language sessions.
            repo: Optional 'owner/name' filter.
            author_login: Optional GH login filter.
            target: Optional target name. Unset = coalesce across all targets.
            scope: 'personal' / 'project' / 'all' — see find_review_comments.
            limit: Max rules per source kind (review + code). Default 50.
        """
        with tracer().start_as_current_span("mcp.tool.house_rules") as span:
            set_safe_attributes(
                span,
                **{
                    "gh_twin.tool.limit": limit,
                    "gh_twin.tool.scope": scope,
                    "gh_twin.tool.target": target,
                    "gh_twin.filter.language": language,
                    "gh_twin.filter.repo": repo,
                    "gh_twin.filter.author_login": author_login,
                },
            )
            result = t.house_rules(
                conn,
                language=language,
                repo=repo,
                author_login=author_login,
                target=target,
                scope=scope,
                limit=limit,
            )
            span.set_attribute("gh_twin.result.review_rules", result["review_rules"])
            span.set_attribute("gh_twin.result.code_rules", result["code_rules"])
            return result

    @mcp.tool()
    def developer_profile(
        author_login: str | None = None,
        language: str | None = None,
        repo: str | None = None,
        target: str | None = None,
        scope: Scope = "all",
        n_samples: int = 50,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Synthesize a short Markdown profile of one developer's review voice.

        Args:
            author_login: GH login to profile. Required in org mode; in
                user mode, omit to profile the corpus owner.
            language: Optional language filter on review-comment chunks.
            repo: Optional 'owner/name' filter.
            target: Optional target name. Unset = coalesce samples across
                every target the author appears in (with dedup).
            scope: 'personal' resolves to the user-mode target — the
                right call when you want a strictly user-mode profile
                without org-side mirrors of the same commits.
            n_samples: Number of recent review comments.
            force_refresh: Bypass the cache and re-synthesize.
        """
        with tracer().start_as_current_span("mcp.tool.developer_profile") as span:
            set_safe_attributes(
                span,
                **{
                    "gh_twin.tool.author_login": author_login,
                    "gh_twin.tool.n_samples": n_samples,
                    "gh_twin.tool.force_refresh": force_refresh,
                    "gh_twin.tool.scope": scope,
                    "gh_twin.tool.target": target,
                    "gh_twin.filter.language": language,
                    "gh_twin.filter.repo": repo,
                },
            )
            result = t.developer_profile(
                conn,
                _get_llm(),
                author_login=author_login,
                language=language,
                repo=repo,
                target=target,
                scope=scope,
                n_samples=n_samples,
                max_tokens=cfg.summarize.profile_max_tokens,
                force_refresh=force_refresh,
            )
            span.set_attribute("gh_twin.result.n_samples", result["n_samples"])
            span.set_attribute("gh_twin.result.from_cache", result["from_cache"])
            return result

    @mcp.tool()
    def bootstrap_status() -> dict[str, Any]:
        """Report whether this DB is ready for retrieval.

        Returns:
            db_initialized: True (the DB file always exists at startup; the
                flag exists for symmetry with future remote backends).
            targets: list of {kind, name} for every target in the DB.
            stats: artifact / chunk / vector counts via `q.stats`.
            in_progress: True when a `bootstrap` call is currently running.
            recommendation: human-readable hint when the DB isn't usable yet;
                None when retrieval will work.
        """
        with tracer().start_as_current_span("mcp.tool.bootstrap_status") as span:
            payload = bootstrap_mod.status_payload(conn)
            set_safe_attributes(
                span,
                **{
                    "gh_twin.bootstrap.targets": len(payload["targets"]),
                    "gh_twin.bootstrap.vectors": payload["stats"].get("vectors", 0),
                    "gh_twin.bootstrap.in_progress": payload["in_progress"],
                },
            )
            return payload

    @mcp.tool()
    async def bootstrap(
        ctx: Context[Any, Any, Any],
        kind: str | None = None,
        name: str | None = None,
        path: str | None = None,
        keep_fork: bool = False,
        skip_sync: bool = False,
    ) -> dict[str, Any]:
        """One-shot setup: discover the target, ingest, embed.

        Auto-detects repo-mode from the MCP server's cwd when `kind` is
        unset. Streams phase-level progress notifications back to the
        client; the final return value carries the saved target + stats.

        Args:
            kind: 'user' | 'org' | 'repo' | None (auto-detect from path/cwd).
            name: user login, org login, or 'owner/name' (repo). Required
                for `kind='org'` and `kind='user'` when not in this user's
                token's identity; optional for `kind='repo'` when `path`
                resolves to a github.com working tree.
            path: filesystem path to walk for `.git/config`. Defaults to
                the MCP server's cwd.
            keep_fork: when the resolved repo is a fork, keep it instead
                of swapping to upstream (default: swap).
            skip_sync: stop after writing target/repo rows; caller invokes
                `sync` later. Use this when you want fast first-call
                feedback and will run the long ingest separately.
        """
        with tracer().start_as_current_span("mcp.tool.bootstrap") as span:
            set_safe_attributes(
                span,
                **{
                    "gh_twin.bootstrap.kind": kind,
                    "gh_twin.bootstrap.name": name,
                    "gh_twin.bootstrap.skip_sync": skip_sync,
                    "gh_twin.bootstrap.keep_fork": keep_fork,
                },
            )
            spec = BootstrapSpec(
                kind=kind,
                name=name,
                path=Path(path) if path else None,
                keep_fork=keep_fork,
                skip_sync=skip_sync,
            )
            step = [0]

            def sync_report(msg: str) -> None:
                # Called from the worker thread spawned by to_thread.run_sync;
                # from_thread.run schedules the async progress notification on
                # the main event loop so MCP message ordering is preserved.
                step[0] += 1
                run_from_thread(ctx.report_progress, float(step[0]), None, msg)

            def work() -> dict[str, Any]:
                return bootstrap_mod.run_bootstrap(cfg, spec, report=sync_report)

            result = await anyio.to_thread.run_sync(work)
            span.set_attribute("gh_twin.bootstrap.ingested", bool(result.get("ingested")))
            span.set_attribute(
                "gh_twin.bootstrap.chunks_total", result.get("stats", {}).get("vectors", 0)
            )
            return result

    @mcp.tool()
    def sync(
        since: str | None = None,
        target: str | None = None,
    ) -> dict[str, Any]:
        """Incremental: pull new commits + review comments and embed them.

        Args:
            since: ISO date floor. If omitted, uses the stored sync cursor.
            target: Optional target name. Unset = run ingest for every
                target in the DB; summarize + embed are corpus-wide.
        """
        with tracer().start_as_current_span("mcp.tool.sync") as span:
            set_safe_attributes(
                span,
                **{"gh_twin.tool.since": since, "gh_twin.tool.target": target},
            )
            before = q.stats(conn)
            from github_twin.target import load_target as _load_target

            target_id: int | None = None
            if target is not None:
                t = _load_target(conn, name=target)
                if t is not None and t.id is not None:
                    target_id = t.id
            run_ingest(
                cfg,
                conn,
                since=since,
                target=target_id,
                report=lambda m: log.info("%s", m),
            )
            run_embed(cfg, conn, report=lambda m: log.info("%s", m))
            after = q.stats(conn)
            new_vectors = after["vectors"] - before["vectors"]
            span.set_attribute("gh_twin.sync.new_vectors", new_vectors)
            return {
                "added": {
                    "artifacts": _delta(before["artifacts"], after["artifacts"]),
                    "chunks": _delta(before["chunks"], after["chunks"]),
                    "vectors": new_vectors,
                },
                "totals": after,
            }

    mcp.run()


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys}


if __name__ == "__main__":  # pragma: no cover
    run()
