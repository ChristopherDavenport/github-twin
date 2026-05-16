"""Pure-Python implementations of the MCP tools, no MCP framework deps.

Keeping these separate from server.py means unit tests can exercise the
retrieval logic directly without spinning up an MCP transport.
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
from github_twin.target import load_target

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
    latency separately from retrieval latency. The span is a noop
    when no OTel SDK is configured (the API ships a free noop tracer)."""
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
    """`hybrid_search` wrapped in a span that records the result count
    and top distance — the two attributes most useful for retrieving
    a query later in the trace UI."""
    with tracer().start_as_current_span("retrieval.hybrid_search") as span:
        set_safe_attributes(
            span,
            **{
                "gh_twin.retrieval.chunk_kind": filters.chunk_kind,
                "gh_twin.retrieval.k": k,
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
    scope: Scope = "all",
    k: int = 5,
    expander: QueryExpander | None = None,
) -> list[dict[str, Any]]:
    """Return up to k past review comments on diffs that look like `diff_hunk`.

    `scope` is sugar over `repo` / `author_login`: `"personal"` fills
    `author_login` from the user-mode target, `"project"` fills `repo`
    from a repo-mode target. Explicit kwargs always win over scope.
    """
    if not diff_hunk.strip():
        return []
    repo, author_login = _resolve_scope(conn, scope=scope, repo=repo, author_login=author_login)
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
                "url": ctx.get("url") or h.artifact_source_url,
                "distance": round(h.distance, 4),
            }
        )
    return out


def summarize_review_patterns(
    conn: sqlite3.Connection,
    *,
    language: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return distilled review rules and the comment quotes that motivated them."""
    return q.list_rules(conn, language=language, limit=limit)


def find_style_examples(
    conn: sqlite3.Connection,
    embedder: Embedder,
    store: VectorStore,
    *,
    query: str,
    language: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
    scope: Scope = "all",
    k: int = 5,
    expander: QueryExpander | None = None,
) -> list[dict[str, Any]]:
    """Return up to k code chunks matching `query`. In user mode, all rows are
    yours; in org mode, narrow with `author_login=...` or `scope="personal"`
    to scope to one person."""
    if not query.strip():
        return []
    repo, author_login = _resolve_scope(conn, scope=scope, repo=repo, author_login=author_login)
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
    """Pick the relevant decision for one PR hit.

    - If `author_login` is supplied, take that author's entry from
      `meta.reviewer_decisions` (org-mode shape).
    - Otherwise prefer the artifact-level `decision` column (user-mode).
    - Returns None if neither path yields a usable decision.
    """
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
    k: int = 20,
) -> dict[str, Any]:
    """Predict how a candidate PR would be reviewed, based on similar past PRs.

    Embeds `diff_or_summary` (a diff, a PR title+body, or a free-form
    description) and pulls the nearest `kind='pr_summary'` chunks. Each
    PR's decision is weighted by inverse distance (closer = more votes).
    Set `author_login` to scope to one reviewer's history (org-mode).

    Intentionally bypasses `hybrid_search` and goes straight to
    `store.search`: the inverse-distance weighting below is calibrated on
    raw L2 distance, and RRF's `1 - rrf_score` would silently change the
    prediction math.
    """
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
            },
        )
        hits = store.search(
            vec,
            filters=VectorSearchFilters(chunk_kind="pr_summary", language=language, repo=repo),
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
    k: int = 5,
    expander: QueryExpander | None = None,
) -> list[dict[str, Any]]:
    """Return up to k distilled code-pattern rules most relevant to `query`.

    Code rules are produced by `gt distill --kind code` (clustering of
    `chunk.kind='code'` blocks plus LLM synthesis). They're stored as
    `chunk.kind='code_rule'` and embedded the same way every other chunk
    is, so retrieval is identical to `find_review_comments`.

    Use this when an agent is about to write code and wants to mirror
    established patterns. Scope to `author_login` for personal style,
    `repo` for project style, or leave both unscoped for org-wide style.
    """
    if not query.strip():
        return []
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
    k: int = 5,
    expander: QueryExpander | None = None,
) -> list[dict[str, Any]]:
    """Return up to k file-at-HEAD snippets matching `query` (org-mode).

    `repo` pre-filters at SQL. `path_glob` post-filters in Python so we
    don't have to teach SQL fnmatch — overscan slightly to compensate.
    `node_kind` filters on the tree-sitter AST node type (e.g.
    "function_definition", "class_definition"); only applies to chunks
    produced by the AST chunker — line-window fallback chunks have
    `node_kind = NULL` and won't match.
    """
    if not query.strip():
        return []
    vec = _embed_one(embedder, query)
    # Overscan when we'll post-filter by path glob, so the final result set
    # still has a decent shot at hitting k.
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
                "url": ctx.get("source_url") or h.artifact_source_url,
                "distance": round(h.distance, 4),
            }
        )
        if len(out) >= k:
            break
    return out


# ---------- Prompt-management surface (house_rules, developer_profile, scope) ----------


def _resolve_scope(
    conn: sqlite3.Connection,
    *,
    scope: Scope,
    repo: str | None,
    author_login: str | None,
) -> tuple[str | None, str | None]:
    """Translate a named scope into concrete `(repo, author_login)`
    filters that `hybrid_search` already understands. Explicit
    `repo=` / `author_login=` kwargs always win — scope only fills
    fields the caller left unset.

    - `scope="personal"` defaults `author_login` to the target's name
      when the target is a single user; in org-mode the caller must
      pass `author_login=` explicitly.
    - `scope="project"` defaults `repo` to the target's name when the
      target is `kind='repo'`; in org-mode the caller picks one.
    - `scope="all"` is a no-op pass-through.
    """
    if scope == "all":
        return repo, author_login

    target = load_target(conn)
    if scope == "personal" and author_login is None and target is not None and target.is_user:
        author_login = target.name
    elif scope == "project" and repo is None and target is not None and target.is_repo:
        repo = target.name
    return repo, author_login


def _render_house_rules(
    review_rules: list[dict[str, Any]],
    code_rules: list[dict[str, Any]],
) -> str:
    """Render two `list_rules` outputs as one Markdown block, grouped
    by language and section. Designed to be pasted directly into an
    agent's system prompt or a `CLAUDE.md` memory file."""

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
        # Sort languages stably: named first (alphabetical), unknown last.
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
                # `cluster_size` is the number of source comments / commits.
                size = r.get("cluster_size") or 0
                lines.append(f"- **{rule_text}**" + (f" (from {size} examples)" if size else ""))
                # First example quote as evidence; trim aggressively.
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
    scope: Scope = "all",
    limit: int = 50,
) -> dict[str, Any]:
    """Return all distilled rules as one Markdown block.

    Two underlying calls:
      - `list_rules(chunk_kind='rule')`      → reviewer conventions
      - `list_rules(chunk_kind='code_rule')` → code patterns

    Output is shaped for direct paste into a system prompt: one
    Markdown section per source kind, language-bucketed within each,
    each rule with a one-line summary plus the first example quote
    and a source URL when available.

    `language` is the high-leverage filter — without it the block
    mixes idioms across every language the corpus touches, which is
    actively misleading inside a single-language session. `repo`,
    `author_login`, and `scope` mirror the retrieval tools so the
    static surface answers the same scope vocabulary.
    """
    repo, author_login = _resolve_scope(conn, scope=scope, repo=repo, author_login=author_login)
    review_rules = q.list_rules(
        conn,
        language=language,
        repo=repo,
        author_login=author_login,
        limit=limit,
        chunk_kind="rule",
    )
    code_rules = q.list_rules(
        conn,
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


# Sentinel login used when caching the user-mode default profile (no
# explicit author_login passed). `load_target(conn)["name"]` is the
# correct value for user-mode but doing the cache key on the literal
# argument keeps the cache logic simple — and in user mode the corpus
# IS the target's history.
_DEFAULT_PROFILE_LOGIN = "__target__"


def _profile_cache_key(
    *,
    author_login: str | None,
    language: str | None,
    repo: str | None,
) -> str:
    """Build a stable cache key over the (author, language, repo)
    filter tuple. When all filters are None the key is the bare
    `_DEFAULT_PROFILE_LOGIN` sentinel — preserves existing cache
    entries written before scope filters existed. Any non-None
    filter expands to a `key=value`-joined suffix."""
    base = author_login or _DEFAULT_PROFILE_LOGIN
    if language is None and repo is None:
        return base
    bits = []
    if language is not None:
        bits.append(f"lang={language}")
    if repo is not None:
        bits.append(f"repo={repo}")
    return base + "|" + "|".join(bits)


def developer_profile(
    conn: sqlite3.Connection,
    llm: TextLLM,
    *,
    author_login: str | None = None,
    language: str | None = None,
    repo: str | None = None,
    scope: Scope = "all",
    n_samples: int = 50,
    max_tokens: int = 600,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Synthesize a 2–3 paragraph voice/style profile for one author.

    Pulls the N most-recent review comments from the corpus (filtered
    by `author_login` / `repo` / `language` when supplied), hashes
    their chunk-ids, and checks `developer_profile_cache`. On a hash
    match the cached Markdown is returned with `from_cache=True`. On
    miss (or `force_refresh=True`) the synthesizer is invoked and the
    new profile is cached.

    The cache key folds in all filter dimensions, so scoping to a
    different language or repo produces (and caches) a distinct
    profile rather than clobbering the unscoped one.

    `n_samples` defaults to 50 — a balance between giving the LLM
    enough range to characterize the author and keeping the prompt
    under any backend's context cap (each comment is truncated to
    ~600 chars in `distill/profile.py`).
    """
    repo, author_login = _resolve_scope(conn, scope=scope, repo=repo, author_login=author_login)
    comments = q.recent_review_comments(
        conn,
        author_login=author_login,
        repo=repo,
        language=language,
        limit=n_samples,
    )
    cache_key = _profile_cache_key(author_login=author_login, language=language, repo=repo)

    if not comments:
        return {
            "profile_md": "",
            "n_samples": 0,
            "from_cache": False,
            "generated_at": None,
            "author_login": author_login,
            "language": language,
            "repo": repo,
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
    }
