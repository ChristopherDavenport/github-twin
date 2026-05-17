"""Pure-Python implementations of the MCP tools, no MCP framework deps.

Keeping these separate from server.py means unit tests can exercise the
retrieval logic directly without spinning up an MCP transport.

Multi-target semantics:

- Every tool that filters by target accepts an optional `target` (target
  name). When unset, queries coalesce across every target in the DB and
  the SQL layer dedupes on `(artifact.kind, artifact.external_id,
  chunk_idx)` so a commit that exists under multiple targets contributes
  one hit.
- `_resolve_scope` translates `scope="personal"` to the unique user-mode
  target (sets both `target_id` and `author_login`) and `scope="project"`
  to the unique repo-mode target. Each narrowed lookup avoids the
  coalesce dedup since single-target reads are already unique.
- Returned hits carry `target_name` so callers can see which target a
  result came from in coalesced mode.
"""

from __future__ import annotations

import fnmatch
import json
import sqlite3
from typing import Any, Literal

from github_twin.distill.profile import sample_hash, synthesize_profile
from github_twin.embed import Embedder
from github_twin.eval.llm import TextLLM
from github_twin.observability import set_safe_attributes, tracer
from github_twin.store import queries as q
from github_twin.store.query_expansion import QueryExpander
from github_twin.store.vector_store import VectorSearchFilters, VectorStore, hybrid_search
from github_twin.target import AmbiguousTargetError, load_target

Scope = Literal["all", "personal", "project"]

CODE_SNIPPET_MAX_LINES = 40

DECISION_KINDS = ("approved", "changes_requested", "commented")


def _truncate(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... [+{len(lines) - max_lines} more lines]"


def _embed_one(embedder: Embedder, text: str) -> list[float]:
    """Run the embedder under a span so the OTel trace shows embed
    latency separately from retrieval latency."""
    with tracer().start_as_current_span("embedder.embed") as span:
        set_safe_attributes(
            span,
            **{
                "gh_twin.embed.input_chars": len(text),
                "gh_twin.embed.model": getattr(embedder, "model_id", "unknown"),
            },
        )
        return embedder.embed([text])[0]


def _hybrid(
    store: VectorStore,
    conn: sqlite3.Connection,
    *,
    query_vec: list[float],
    query_text: str,
    filters: VectorSearchFilters,
    k: int,
    expander: QueryExpander | None,
) -> list[q.SearchHit]:
    """`hybrid_search` wrapped in a span that records result count and
    top distance."""
    with tracer().start_as_current_span("retrieval.hybrid_search") as span:
        set_safe_attributes(
            span,
            **{
                "gh_twin.retrieval.chunk_kind": filters.chunk_kind,
                "gh_twin.retrieval.k": k,
                "gh_twin.retrieval.target_id": filters.target_id,
                "gh_twin.retrieval.expander": (
                    expander.backend_id if expander is not None else "off"
                ),
            },
        )
        hits = hybrid_search(
            store,
            conn,
            query_vec=query_vec,
            query_text=query_text,
            filters=filters,
            k=k,
            expander=expander,
        )
        span.set_attribute("gh_twin.retrieval.hits", len(hits))
        if hits:
            span.set_attribute("gh_twin.retrieval.top_distance", round(hits[0].distance, 6))
        return hits


def find_review_comments(
    conn: sqlite3.Connection,
    embedder: Embedder,
    store: VectorStore,
    *,
    diff_hunk: str,
    language: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
    target: str | None = None,
    scope: Scope = "all",
    k: int = 5,
    expander: QueryExpander | None = None,
) -> list[dict[str, Any]]:
    """Return up to k past review comments on diffs that look like `diff_hunk`.

    `target` narrows to one target by name; without it the search coalesces
    across every target with cross-target dedup. `scope="personal"` resolves
    to the unique user-mode target; `scope="project"` to the unique repo-mode
    target. Explicit kwargs always win over scope.
    """
    if not diff_hunk.strip():
        return []
    target_id, repo, author_login = _resolve_scope(
        conn, scope=scope, target=target, repo=repo, author_login=author_login
    )
    vec = _embed_one(embedder, diff_hunk)
    hits = _hybrid(
        store,
        conn,
        query_vec=vec,
        query_text=diff_hunk,
        filters=VectorSearchFilters(
            chunk_kind="review_comment",
            language=language,
            repo=repo,
            author_login=author_login,
            target_id=target_id,
        ),
        k=k,
        expander=expander,
    )
    out: list[dict[str, Any]] = []
    for h in hits:
        ctx = h.context or {}
        out.append(
            {
                "comment": h.text,
                "diff_hunk_context": _truncate(ctx.get("diff_hunk") or "", CODE_SNIPPET_MAX_LINES)
                if ctx.get("diff_hunk")
                else None,
                "path": ctx.get("path"),
                "language": ctx.get("language") or h.artifact_language,
                "pr_title": ctx.get("pr_title"),
                "repo": h.artifact_repo,
                "target": h.target_name,
                "url": ctx.get("url") or h.artifact_source_url,
                "distance": round(h.distance, 4),
            }
        )
    return out


def summarize_review_patterns(
    conn: sqlite3.Connection,
    *,
    language: str | None = None,
    target: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return distilled review rules and the comment quotes that motivated them.

    `target` narrows to one target's rules; without it returns rules
    coalesced across every target.
    """
    target_id = _lookup_target_id(conn, target)
    return q.list_rules(conn, language=language, limit=limit, target_id=target_id)


def find_style_examples(
    conn: sqlite3.Connection,
    embedder: Embedder,
    store: VectorStore,
    *,
    query: str,
    language: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
    target: str | None = None,
    scope: Scope = "all",
    k: int = 5,
    expander: QueryExpander | None = None,
) -> list[dict[str, Any]]:
    """Return up to k code chunks matching `query`. Multi-target aware:
    `target` narrows to one target; unset coalesces across all."""
    if not query.strip():
        return []
    target_id, repo, author_login = _resolve_scope(
        conn, scope=scope, target=target, repo=repo, author_login=author_login
    )
    vec = _embed_one(embedder, query)
    hits = _hybrid(
        store,
        conn,
        query_vec=vec,
        query_text=query,
        filters=VectorSearchFilters(
            chunk_kind="code",
            language=language,
            repo=repo,
            author_login=author_login,
            target_id=target_id,
        ),
        k=k,
        expander=expander,
    )
    out: list[dict[str, Any]] = []
    for h in hits:
        ctx = h.context or {}
        out.append(
            {
                "code_snippet": _truncate(h.text, CODE_SNIPPET_MAX_LINES),
                "language": ctx.get("language") or h.artifact_language,
                "path": ctx.get("path"),
                "repo": h.artifact_repo,
                "target": h.target_name,
                "commit_url": ctx.get("source_url") or h.artifact_source_url,
                "distance": round(h.distance, 4),
            }
        )
    return out


def _fetch_pr_metas(conn: sqlite3.Connection, artifact_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not artifact_ids:
        return {}
    placeholders = ",".join("?" * len(artifact_ids))
    rows = conn.execute(
        f"SELECT id, meta_json FROM artifact WHERE id IN ({placeholders})",
        artifact_ids,
    ).fetchall()
    return {r["id"]: (json.loads(r["meta_json"]) if r["meta_json"] else {}) for r in rows}


def _decision_for_hit(
    hit: q.SearchHit, meta: dict[str, Any], author_login: str | None
) -> str | None:
    """Pick the relevant decision for one PR hit."""
    if author_login is not None:
        for entry in meta.get("reviewer_decisions", []) or []:
            if entry.get("login") == author_login:
                state = (entry.get("state") or "").lower()
                if state in DECISION_KINDS:
                    return state
        return None
    if hit.artifact_decision and hit.artifact_decision in DECISION_KINDS:
        return hit.artifact_decision
    return None


def predict_review_outcome(
    conn: sqlite3.Connection,
    embedder: Embedder,
    store: VectorStore,
    *,
    diff_or_summary: str,
    language: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
    target: str | None = None,
    k: int = 20,
) -> dict[str, Any]:
    """Predict how a candidate PR would be reviewed.

    Intentionally bypasses `hybrid_search` and uses `store.search`
    directly: the inverse-distance weighting below is calibrated on
    raw L2 distance, and RRF's `1 - rrf_score` would silently change
    the prediction math.

    `target` narrows to one target's PRs; without it coalesces with
    dedup (so a PR ingested under multiple targets contributes one vote).
    """
    target_id = _lookup_target_id(conn, target)
    empty = {
        "prediction": "unknown",
        "confidence": 0.0,
        "weighted": {k: 0.0 for k in DECISION_KINDS},
        "n_pulled": 0,
        "n_with_decision": 0,
        "support": [],
    }
    if not diff_or_summary.strip():
        return empty
    vec = _embed_one(embedder, diff_or_summary)
    with tracer().start_as_current_span("retrieval.vector_search") as span:
        set_safe_attributes(
            span,
            **{
                "gh_twin.retrieval.chunk_kind": "pr_summary",
                "gh_twin.retrieval.k": k,
                "gh_twin.retrieval.target_id": target_id,
            },
        )
        hits = store.search(
            vec,
            filters=VectorSearchFilters(
                chunk_kind="pr_summary",
                language=language,
                repo=repo,
                target_id=target_id,
            ),
            k=k,
        )
        span.set_attribute("gh_twin.retrieval.hits", len(hits))
        if hits:
            span.set_attribute("gh_twin.retrieval.top_distance", round(hits[0].distance, 6))
    if not hits:
        return empty

    metas = _fetch_pr_metas(conn, [h.artifact_id for h in hits])
    weighted = {k: 0.0 for k in DECISION_KINDS}
    support: list[dict[str, Any]] = []
    n_with_decision = 0
    for h in hits:
        meta = metas.get(h.artifact_id, {})
        decision = _decision_for_hit(h, meta, author_login)
        if decision is not None:
            w = 1.0 / (1.0 + max(h.distance, 0.0))
            weighted[decision] += w
            n_with_decision += 1
        ctx = h.context or {}
        support.append(
            {
                "url": h.artifact_source_url,
                "title": ctx.get("pr_title") or meta.get("title"),
                "repo": h.artifact_repo,
                "target": h.target_name,
                "distance": round(h.distance, 4),
                "decision": decision,
            }
        )
    total = sum(weighted.values())
    if total == 0:
        return {**empty, "n_pulled": len(hits), "support": support}
    prediction = max(weighted, key=lambda kind: weighted[kind])
    return {
        "prediction": prediction,
        "confidence": round(weighted[prediction] / total, 4),
        "weighted": {k: round(v, 4) for k, v in weighted.items()},
        "n_pulled": len(hits),
        "n_with_decision": n_with_decision,
        "support": support,
    }


def find_applicable_rules(
    conn: sqlite3.Connection,
    embedder: Embedder,
    store: VectorStore,
    *,
    query: str,
    language: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
    target: str | None = None,
    k: int = 5,
    expander: QueryExpander | None = None,
) -> list[dict[str, Any]]:
    """Return up to k distilled code-pattern rules most relevant to `query`."""
    if not query.strip():
        return []
    target_id = _lookup_target_id(conn, target)
    vec = _embed_one(embedder, query)
    hits = _hybrid(
        store,
        conn,
        query_vec=vec,
        query_text=query,
        filters=VectorSearchFilters(
            chunk_kind="code_rule",
            language=language,
            repo=repo,
            author_login=author_login,
            target_id=target_id,
        ),
        k=k,
        expander=expander,
    )
    out: list[dict[str, Any]] = []
    for h in hits:
        ctx = h.context or {}
        out.append(
            {
                "rule": h.text,
                "language": ctx.get("language") or h.artifact_language,
                "examples": ctx.get("examples", []),
                "target": h.target_name,
                "distance": round(h.distance, 4),
            }
        )
    return out


def find_code(
    conn: sqlite3.Connection,
    embedder: Embedder,
    store: VectorStore,
    *,
    query: str,
    language: str | None = None,
    repo: str | None = None,
    path_glob: str | None = None,
    node_kind: str | None = None,
    target: str | None = None,
    k: int = 5,
    expander: QueryExpander | None = None,
) -> list[dict[str, Any]]:
    """Return up to k file-at-HEAD snippets matching `query`."""
    if not query.strip():
        return []
    target_id = _lookup_target_id(conn, target)
    vec = _embed_one(embedder, query)
    # Overscan when we'll post-filter by path glob.
    sql_k = k * 4 if path_glob else k
    hits = _hybrid(
        store,
        conn,
        query_vec=vec,
        query_text=query,
        filters=VectorSearchFilters(
            chunk_kind="file",
            language=language,
            repo=repo,
            node_kind=node_kind,
            target_id=target_id,
        ),
        k=sql_k,
        expander=expander,
    )
    out: list[dict[str, Any]] = []
    for h in hits:
        ctx = h.context or {}
        path = ctx.get("path")
        if path_glob and not (path and fnmatch.fnmatch(path, path_glob)):
            continue
        out.append(
            {
                "code_snippet": _truncate(h.text, CODE_SNIPPET_MAX_LINES),
                "language": ctx.get("language") or h.artifact_language,
                "path": path,
                "start_line": ctx.get("start_line"),
                "end_line": ctx.get("end_line"),
                "node_kind": ctx.get("node_kind"),
                "symbol_name": ctx.get("symbol_name"),
                "repo": h.artifact_repo,
                "target": h.target_name,
                "url": ctx.get("source_url") or h.artifact_source_url,
                "distance": round(h.distance, 4),
            }
        )
        if len(out) >= k:
            break
    return out


# ---------- Prompt-management surface (house_rules, developer_profile, scope) ----------


def _lookup_target_id(conn: sqlite3.Connection, target: str | None) -> int | None:
    """Translate a `target=NAME` parameter to an id. None means coalesce."""
    if target is None:
        return None
    t = load_target(conn, name=target)
    if t is None or t.id is None:
        raise ValueError(f"No target named {target!r} in this DB.")
    return t.id


def _resolve_scope(
    conn: sqlite3.Connection,
    *,
    scope: Scope,
    target: str | None,
    repo: str | None,
    author_login: str | None,
) -> tuple[int | None, str | None, str | None]:
    """Translate scope + target into `(target_id, repo, author_login)`.

    - `scope="personal"` resolves the unique user-mode target and fills
      both `target_id` (so coalesce-mode dedup doesn't reintroduce org-side
      copies of the user's own commits) and `author_login` (sugar over the
      author_login column).
    - `scope="project"` resolves the unique repo-mode target and fills
      `target_id` + `repo`.
    - `scope="all"` is a no-op pass-through.
    - Explicit `target` always wins; explicit `repo` / `author_login`
      kwargs win within their dimension.
    """
    target_id = _lookup_target_id(conn, target)
    if scope == "all":
        return target_id, repo, author_login
    if scope == "personal":
        try:
            t = load_target(conn, kind="user")
        except AmbiguousTargetError as exc:
            raise ValueError(f"scope='personal' is ambiguous: {exc}") from None
        if t is not None and t.id is not None:
            if target_id is None:
                target_id = t.id
            if author_login is None:
                author_login = t.name
    elif scope == "project":
        try:
            t = load_target(conn, kind="repo")
        except AmbiguousTargetError as exc:
            raise ValueError(f"scope='project' is ambiguous: {exc}") from None
        if t is not None and t.id is not None:
            if target_id is None:
                target_id = t.id
            if repo is None:
                repo = t.name
    return target_id, repo, author_login


def _render_house_rules(
    review_rules: list[dict[str, Any]],
    code_rules: list[dict[str, Any]],
) -> str:
    """Render two `list_rules` outputs as one Markdown block, grouped
    by language and section."""

    def by_language(rules: list[dict[str, Any]]) -> dict[str | None, list[dict[str, Any]]]:
        out: dict[str | None, list[dict[str, Any]]] = {}
        for r in rules:
            out.setdefault(r.get("language"), []).append(r)
        return out

    def render_section(title: str, rules: list[dict[str, Any]]) -> str:
        if not rules:
            return ""
        lines = [f"## {title} ({len(rules)} rules)\n"]
        grouped = by_language(rules)
        named = sorted(k for k in grouped if k)
        unknown = [k for k in grouped if not k]
        for lang in named + unknown:
            label = lang or "(language unspecified)"
            bucket = grouped[lang]
            lines.append(f"### {label} ({len(bucket)} rules)\n")
            for r in bucket:
                rule_text = (r.get("rule") or "").strip()
                if not rule_text:
                    continue
                size = r.get("cluster_size") or 0
                lines.append(f"- **{rule_text}**" + (f" (from {size} examples)" if size else ""))
                examples = r.get("examples") or []
                if examples:
                    quote = str(examples[0]).strip().replace("\n", " ")
                    if len(quote) > 200:
                        quote = quote[:200].rstrip() + "…"
                    lines.append(f"  > {quote}")
                urls = r.get("urls") or []
                if urls:
                    lines.append(f"  ([source]({urls[0]}))")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    blocks = []
    review_md = render_section("Reviewer conventions", review_rules)
    code_md = render_section("Code patterns", code_rules)
    if review_md:
        blocks.append(review_md)
    if code_md:
        blocks.append(code_md)
    if not blocks:
        return (
            "## House rules\n\n"
            "_No rules distilled yet. Run `gt distill` (review comments) "
            "or `gt distill --kind code` (commits) to populate this surface._\n"
        )
    return "\n".join(blocks)


def house_rules(
    conn: sqlite3.Connection,
    *,
    language: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
    target: str | None = None,
    scope: Scope = "all",
    limit: int = 50,
) -> dict[str, Any]:
    """Return all distilled rules as one Markdown block.

    Without `target`, rules coalesce across every target with dedup; with
    `target=NAME`, rules are restricted to that target.
    """
    target_id, repo, author_login = _resolve_scope(
        conn, scope=scope, target=target, repo=repo, author_login=author_login
    )
    review_rules = q.list_rules(
        conn,
        target_id=target_id,
        language=language,
        repo=repo,
        author_login=author_login,
        limit=limit,
        chunk_kind="rule",
    )
    code_rules = q.list_rules(
        conn,
        target_id=target_id,
        language=language,
        repo=repo,
        author_login=author_login,
        limit=limit,
        chunk_kind="code_rule",
    )
    markdown = _render_house_rules(review_rules, code_rules)
    return {
        "markdown": markdown,
        "review_rules": len(review_rules),
        "code_rules": len(code_rules),
    }


# Sentinel base used when caching the user-mode default profile (no
# explicit author_login passed).
_DEFAULT_PROFILE_LOGIN = "__target__"


def _profile_cache_key(
    *,
    author_login: str | None,
    language: str | None,
    repo: str | None,
    target_id: int | None = None,
) -> str:
    """Build a stable cache key over the (author, language, repo, target)
    filter tuple. Each non-None dimension expands to a `key=value`-joined
    suffix so a narrowed profile doesn't collide with an unscoped one."""
    base = author_login or _DEFAULT_PROFILE_LOGIN
    if language is None and repo is None and target_id is None:
        return base
    bits = []
    if language is not None:
        bits.append(f"lang={language}")
    if repo is not None:
        bits.append(f"repo={repo}")
    if target_id is not None:
        bits.append(f"target={target_id}")
    return base + "|" + "|".join(bits)


def developer_profile(
    conn: sqlite3.Connection,
    llm: TextLLM,
    *,
    author_login: str | None = None,
    language: str | None = None,
    repo: str | None = None,
    target: str | None = None,
    scope: Scope = "all",
    n_samples: int = 50,
    max_tokens: int = 600,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Synthesize a 2–3 paragraph voice/style profile for one author.

    Without `target`, samples coalesce across every target the author
    appears in (with dedup); with `target=NAME`, narrows to that target's
    history only. `scope="personal"` is the right call when you want
    "my own user-mode corpus" specifically — it narrows to the user-mode
    target so org-side mirrors of the same commits don't enter the sample.
    """
    target_id, repo, author_login = _resolve_scope(
        conn, scope=scope, target=target, repo=repo, author_login=author_login
    )
    comments = q.recent_review_comments(
        conn,
        author_login=author_login,
        repo=repo,
        language=language,
        target_id=target_id,
        limit=n_samples,
    )
    cache_key = _profile_cache_key(
        author_login=author_login,
        language=language,
        repo=repo,
        target_id=target_id,
    )

    if not comments:
        return {
            "profile_md": "",
            "n_samples": 0,
            "from_cache": False,
            "generated_at": None,
            "author_login": author_login,
            "language": language,
            "repo": repo,
            "target_id": target_id,
        }

    current_hash = sample_hash(comments)
    if not force_refresh:
        cached = q.get_cached_profile(conn, cache_key)
        if cached is not None and cached["sample_hash"] == current_hash:
            return {
                "profile_md": cached["profile_md"],
                "n_samples": cached["n_samples"],
                "from_cache": True,
                "generated_at": cached["generated_at"],
                "author_login": author_login,
                "language": language,
                "repo": repo,
                "target_id": target_id,
            }

    profile_md = synthesize_profile(llm, comments, max_tokens=max_tokens)
    q.set_cached_profile(
        conn,
        login=cache_key,
        profile_md=profile_md,
        sample_hash=current_hash,
        n_samples=len(comments),
    )
    refreshed = q.get_cached_profile(conn, cache_key)
    generated_at = refreshed["generated_at"] if refreshed else None
    return {
        "profile_md": profile_md,
        "n_samples": len(comments),
        "from_cache": False,
        "generated_at": generated_at,
        "author_login": author_login,
        "language": language,
        "repo": repo,
        "target_id": target_id,
    }
