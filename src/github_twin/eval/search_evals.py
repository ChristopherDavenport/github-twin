"""Tiered retrieval-quality eval — the "dogfood suite" surface.

Loads a YAML of queries, each with one or more `expect_any` clauses and a
`tier` priority. Runs each query through BM25, vector, and hybrid paths
independently and reports per-tier, per-backend pass rates. The
per-backend split is the load-bearing bit: it surfaces the case where a
hybrid pass rate hides one leg silently regressing (the failure mode
that motivated this whole eval).

Wire from `cli.py:eval_app` as `gt eval search <yaml>`.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from github_twin.embed.base import Embedder
from github_twin.store import queries as q
from github_twin.store.query_expansion import QueryExpander
from github_twin.store.vector_store import VectorSearchFilters, VectorStore, hybrid_search

Mode = Literal["bm25", "vector", "hybrid"]
ALL_MODES: tuple[Mode, ...] = ("bm25", "vector", "hybrid")

# Each tool maps to a chunk_kind. Mirrors `mcp_server/tools.py`.
_TOOL_TO_CHUNK_KIND: dict[str, str] = {
    "find_review_comments": "review_comment",
    "find_style_examples": "code",
    "find_code": "file",
    "find_applicable_rules": "code_rule",
}


@dataclass
class Expectation:
    """One clause inside a query's `expect_any` list.

    A hit satisfies an Expectation when *any* set field matches. A query
    passes when *any* Expectation in the list is satisfied by *any* of
    the top-k hits.
    """

    artifact_id: int | None = None
    chunk_id: int | None = None
    path_substr: str | None = None
    text_substr: str | None = None
    url_substr: str | None = None
    symbol_name: str | None = None
    node_kind: str | None = None


@dataclass
class SearchQuery:
    query: str
    tool: str
    tier: int  # 1 = critical, 2 = important, 3 = nice
    expect_any: list[Expectation]
    filters: dict[str, Any] = field(default_factory=dict)  # passes through to VectorSearchFilters
    label: str | None = None  # optional human tag for reporting


@dataclass
class QueryOutcome:
    query: SearchQuery
    mode: Mode
    passed: bool
    top_hits: list[q.SearchHit]  # truncated to k for the report


@dataclass
class SearchEvalReport:
    outcomes: list[QueryOutcome]
    k: int

    # ---- aggregates ----
    def pass_rate(self, *, tier: int | None = None, mode: Mode | None = None) -> tuple[int, int]:
        """Return (passed, total) filtered by tier and/or mode."""
        relevant = [
            o
            for o in self.outcomes
            if (tier is None or o.query.tier == tier) and (mode is None or o.mode == mode)
        ]
        return sum(1 for o in relevant if o.passed), len(relevant)

    def failures(self, *, tier: int, mode: Mode) -> list[QueryOutcome]:
        return [
            o for o in self.outcomes if o.query.tier == tier and o.mode == mode and not o.passed
        ]

    def tiers(self) -> list[int]:
        return sorted({o.query.tier for o in self.outcomes})


# ---------- loader ----------


def load_queries(path: Path) -> list[SearchQuery]:
    """Parse a YAML file into SearchQuery objects.

    Unknown keys raise — better to surface typos than silently drop them.
    """
    import yaml

    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: top-level YAML must be a list of queries")
    out: list[SearchQuery] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}#{i}: each query must be a mapping")
        out.append(_parse_query(item, where=f"{path}#{i}"))
    return out


def _parse_query(d: dict[str, Any], *, where: str) -> SearchQuery:
    allowed_top = {"query", "tool", "tier", "expect_any", "filters", "label"}
    extra = set(d) - allowed_top
    if extra:
        raise ValueError(f"{where}: unknown keys {sorted(extra)}")
    if "query" not in d or "tool" not in d:
        raise ValueError(f"{where}: 'query' and 'tool' are required")
    tool = d["tool"]
    if tool not in _TOOL_TO_CHUNK_KIND:
        raise ValueError(
            f"{where}: tool={tool!r} not recognized; expected one of {sorted(_TOOL_TO_CHUNK_KIND)}"
        )
    tier = int(d.get("tier", 2))
    if tier not in (1, 2, 3):
        raise ValueError(f"{where}: tier must be 1, 2, or 3")
    expectations_raw = d.get("expect_any") or []
    if not expectations_raw:
        raise ValueError(f"{where}: 'expect_any' must list at least one clause")
    expectations = [
        _parse_expectation(e, where=f"{where}.expect_any[{i}]")
        for i, e in enumerate(expectations_raw)
    ]
    return SearchQuery(
        query=d["query"],
        tool=tool,
        tier=tier,
        expect_any=expectations,
        filters=dict(d.get("filters") or {}),
        label=d.get("label"),
    )


def _parse_expectation(d: dict[str, Any], *, where: str) -> Expectation:
    allowed = {
        "artifact_id",
        "chunk_id",
        "path_substr",
        "text_substr",
        "url_substr",
        "symbol_name",
        "node_kind",
    }
    extra = set(d) - allowed
    if extra:
        raise ValueError(f"{where}: unknown keys {sorted(extra)}")
    if not any(k in d for k in allowed):
        raise ValueError(f"{where}: clause is empty (no match fields)")
    return Expectation(
        artifact_id=d.get("artifact_id"),
        chunk_id=d.get("chunk_id"),
        path_substr=d.get("path_substr"),
        text_substr=d.get("text_substr"),
        url_substr=d.get("url_substr"),
        symbol_name=d.get("symbol_name"),
        node_kind=d.get("node_kind"),
    )


# ---------- matching ----------


def _hit_matches(hit: q.SearchHit, expectation: Expectation) -> bool:
    if expectation.artifact_id is not None and hit.artifact_id == expectation.artifact_id:
        return True
    if expectation.chunk_id is not None and hit.chunk_id == expectation.chunk_id:
        return True
    ctx = hit.context or {}
    if expectation.path_substr is not None:
        path = (ctx.get("path") or "") or ""
        if expectation.path_substr in path:
            return True
    if expectation.text_substr is not None:
        text = hit.text or ""
        if expectation.text_substr.lower() in text.lower():
            return True
    if expectation.url_substr is not None:
        url = ctx.get("url") or hit.artifact_source_url or ""
        if expectation.url_substr in url:
            return True
    if expectation.symbol_name is not None and ctx.get("symbol_name") == expectation.symbol_name:
        return True
    return bool(expectation.node_kind is not None and ctx.get("node_kind") == expectation.node_kind)


def _query_passes(hits: list[q.SearchHit], expectations: list[Expectation]) -> bool:
    return any(_hit_matches(h, e) for h in hits for e in expectations)


# ---------- runner ----------


def _filters_for(query: SearchQuery) -> VectorSearchFilters:
    chunk_kind = _TOOL_TO_CHUNK_KIND[query.tool]
    f = query.filters
    return VectorSearchFilters(
        chunk_kind=chunk_kind,
        language=f.get("language"),
        repo=f.get("repo"),
        author_login=f.get("author_login"),
        node_kind=f.get("node_kind"),
    )


def run_query(
    conn: sqlite3.Connection,
    embedder: Embedder,
    store: VectorStore,
    *,
    query: SearchQuery,
    mode: Mode,
    k: int = 5,
    expander: QueryExpander | None = None,
) -> list[q.SearchHit]:
    filters = _filters_for(query)
    if mode == "bm25":
        return q.bm25_search(
            conn,
            query_text=query.query,
            chunk_kind=filters.chunk_kind,
            language=filters.language,
            repo=filters.repo,
            author_login=filters.author_login,
            node_kind=filters.node_kind,
            k=k,
            expander=expander,
        )
    vec = embedder.embed([query.query])[0]
    if mode == "vector":
        return store.search(vec, filters=filters, k=k)
    if mode == "hybrid":
        return hybrid_search(
            store,
            conn,
            query_vec=vec,
            query_text=query.query,
            filters=filters,
            k=k,
            expander=expander,
        )
    raise ValueError(f"unknown mode {mode!r}")


def evaluate_search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    store: VectorStore,
    queries: Iterable[SearchQuery],
    *,
    k: int = 5,
    modes: Iterable[Mode] = ALL_MODES,
    expander: QueryExpander | None = None,
) -> SearchEvalReport:
    """Run every query through every mode; build a flat outcome list."""
    queries = list(queries)
    modes = list(modes)
    outcomes: list[QueryOutcome] = []
    for sq in queries:
        for mode in modes:
            hits = run_query(
                conn,
                embedder,
                store,
                query=sq,
                mode=mode,
                k=k,
                expander=expander,
            )
            passed = _query_passes(hits, sq.expect_any)
            outcomes.append(QueryOutcome(query=sq, mode=mode, passed=passed, top_hits=hits))
    return SearchEvalReport(outcomes=outcomes, k=k)


# ---------- exposed for the CLI report ----------


def per_tool_pass_rate(report: SearchEvalReport, mode: Mode) -> dict[str, tuple[int, int]]:
    """Group outcomes by tool, return {tool: (passed, total)} for a given mode."""
    bucket: dict[str, list[QueryOutcome]] = defaultdict(list)
    for o in report.outcomes:
        if o.mode == mode:
            bucket[o.query.tool].append(o)
    return {tool: (sum(1 for o in os if o.passed), len(os)) for tool, os in bucket.items()}
