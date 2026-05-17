"""Retrieval-quality eval harness (`gt eval search`).

Tiny FakeEmbedder + 3 staged chunks lets us assert that:
  - bm25-only / vector-only / hybrid each pass the queries they should
  - per-tier pass rates are computed from the right outcomes
  - the YAML loader rejects unknown fields and missing required fields
  - Tier-1 misses produce a non-zero exit code in the renderer
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
import yaml
from rich.console import Console

from github_twin.eval.report import render_search_result
from github_twin.eval.search_evals import (
    Expectation,
    SearchEvalReport,
    SearchQuery,
    evaluate_search,
    load_queries,
)
from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.store.vector_store import SqliteVecStore
from tests.conftest import seed_target


class FakeEmbedder:
    dim = 4
    model_id = "fake"
    PATTERNS = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0, 0.0],
        "C": [0.0, 0.0, 1.0, 0.0],
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for s in texts:
            for k, v in self.PATTERNS.items():
                if k in s:
                    out.append(list(v))
                    break
            else:
                out.append([0.0, 0.0, 0.0, 1.0])
        return out


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "search_eval.sqlite", embed_dim=FakeEmbedder.dim)
    seed_target(db)
    yield db
    db.close()


def _seed_code(conn, *, path, text, vec, symbol=None, node=None):
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="commit",
        external_id=f"art-{path}-{symbol}",
        source_url=None,
        repo="me/x",
        language="python",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    cid = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="code",
        text=text,
        context={"path": path, "symbol_name": symbol, "node_kind": node, "language": "python"},
        language="python",
    )
    q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="fake")
    return cid


# ---------- loader ----------


def _write_yaml(tmp_path: Path, data) -> Path:
    p = tmp_path / "q.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_load_queries_parses_well_formed(tmp_path: Path):
    p = _write_yaml(
        tmp_path,
        [
            {
                "query": "test",
                "tool": "find_style_examples",
                "tier": 1,
                "expect_any": [{"path_substr": "foo.py"}],
            }
        ],
    )
    qs = load_queries(p)
    assert len(qs) == 1
    assert qs[0].tool == "find_style_examples"
    assert qs[0].tier == 1
    assert qs[0].expect_any[0].path_substr == "foo.py"


def test_load_queries_rejects_unknown_tool(tmp_path: Path):
    p = _write_yaml(
        tmp_path, [{"query": "x", "tool": "find_nope", "expect_any": [{"path_substr": "a"}]}]
    )
    with pytest.raises(ValueError, match="tool="):
        load_queries(p)


def test_load_queries_rejects_unknown_key(tmp_path: Path):
    p = _write_yaml(
        tmp_path,
        [
            {
                "query": "x",
                "tool": "find_style_examples",
                "expect_any": [{"path_substr": "a"}],
                "typo": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_queries(p)


def test_load_queries_requires_expect_any(tmp_path: Path):
    p = _write_yaml(tmp_path, [{"query": "x", "tool": "find_style_examples", "expect_any": []}])
    with pytest.raises(ValueError, match="expect_any"):
        load_queries(p)


def test_load_queries_rejects_empty_clause(tmp_path: Path):
    p = _write_yaml(
        tmp_path,
        [
            {
                "query": "x",
                "tool": "find_style_examples",
                "expect_any": [{}],
            }
        ],
    )
    with pytest.raises(ValueError, match="empty"):
        load_queries(p)


def test_load_queries_rejects_bad_tier(tmp_path: Path):
    p = _write_yaml(
        tmp_path,
        [
            {
                "query": "x",
                "tool": "find_style_examples",
                "tier": 7,
                "expect_any": [{"path_substr": "a"}],
            }
        ],
    )
    with pytest.raises(ValueError, match="tier"):
        load_queries(p)


# ---------- runner: bm25 vs vector vs hybrid ----------


def test_bm25_only_query_passes_bm25_fails_vector(conn):
    """A query whose embedding is far from the right chunk but whose text
    contains the right keyword must pass under BM25 / hybrid but not vector.

    To force a vector miss with k=2, stage three pattern-A decoys (all at
    distance 0 from the query) plus one pattern-C target. Vector returns
    the three A's; target falls outside the top-k.
    """
    target = _seed_code(
        conn,
        path="src/needle.py",
        text="contains unique_token_xyz here",
        vec=[0.0, 0.0, 1.0, 0.0],
    )
    # Decoys at the default no-pattern embedding so they sit at distance 0
    # from the (single-token, no-pattern) query and push target out of top-k.
    for i in range(3):
        _seed_code(conn, path=f"src/decoy{i}.py", text=f"decoy_{i}", vec=[0.0, 0.0, 0.0, 1.0])

    sq = SearchQuery(
        query="unique_token_xyz",  # no pattern → default vector; bm25 → needle
        tool="find_style_examples",
        tier=1,
        expect_any=[Expectation(chunk_id=target)],
    )
    report = evaluate_search(conn, FakeEmbedder(), SqliteVecStore(conn), [sq], k=2)

    by_mode = {o.mode: o.passed for o in report.outcomes}
    assert by_mode["bm25"] is True
    assert by_mode["hybrid"] is True
    assert by_mode["vector"] is False


def test_vector_only_query_passes_vector(conn):
    """Vector-close chunk with no lexical overlap still passes vector + hybrid."""
    target = _seed_code(
        conn,
        path="src/near.py",
        text="A near in vector space only",
        vec=[1.0, 0.0, 0.0, 0.0],
    )
    sq = SearchQuery(
        query="A something_no_match",
        tool="find_style_examples",
        tier=1,
        expect_any=[Expectation(chunk_id=target)],
    )
    report = evaluate_search(conn, FakeEmbedder(), SqliteVecStore(conn), [sq], k=5)
    by_mode = {o.mode: o.passed for o in report.outcomes}
    assert by_mode["vector"] is True
    assert by_mode["hybrid"] is True


# ---------- expectation matching ----------


def test_expect_any_path_substr_matches(conn):
    _seed_code(conn, path="src/deep/match.py", text="A code", vec=[1, 0, 0, 0])
    sq = SearchQuery(
        query="A code",
        tool="find_style_examples",
        tier=2,
        expect_any=[Expectation(path_substr="deep/match")],
    )
    rep = evaluate_search(conn, FakeEmbedder(), SqliteVecStore(conn), [sq], modes=("hybrid",))
    assert rep.outcomes[0].passed


def test_expect_any_symbol_name_matches(conn):
    _seed_code(
        conn,
        path="src/x.py",
        text="A code",
        vec=[1, 0, 0, 0],
        symbol="renew",
        node="function_definition",
    )
    sq = SearchQuery(
        query="A code",
        tool="find_style_examples",
        tier=2,
        expect_any=[Expectation(symbol_name="renew")],
    )
    rep = evaluate_search(conn, FakeEmbedder(), SqliteVecStore(conn), [sq], modes=("hybrid",))
    assert rep.outcomes[0].passed


# ---------- aggregation ----------


def test_evaluate_search_forwards_recency_to_hybrid_only(conn, monkeypatch):
    """`evaluate_search` must pass `recency_half_life_days` into the hybrid
    leg (so `gt eval search --recency-half-life-days=N` actually measures
    something) and must NOT pass it to bm25_search or store.search
    (per the eval's "each leg unweighted by design" contract).
    """
    from github_twin.eval import search_evals as se

    captured: dict[str, list] = {"hybrid": [], "bm25": [], "vector": []}

    def fake_hybrid(*args, **kwargs):
        captured["hybrid"].append(kwargs.get("recency_half_life_days"))
        return []

    def fake_bm25(*args, **kwargs):
        # bm25_search has no recency kwarg; failure would be a TypeError above.
        assert "recency_half_life_days" not in kwargs
        captured["bm25"].append(True)
        return []

    class SpyStore:
        backend_id = "spy"

        def search(self, vec, *, filters, k=5):
            captured["vector"].append(True)
            return []

    monkeypatch.setattr(se, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(se.q, "bm25_search", fake_bm25)

    sq = SearchQuery(
        query="A query",
        tool="find_style_examples",
        tier=2,
        expect_any=[Expectation(text_substr="x")],
    )
    se.evaluate_search(
        conn,
        FakeEmbedder(),
        SpyStore(),
        [sq],
        k=3,
        recency_half_life_days=365.0,
    )

    # All three legs ran exactly once.
    assert captured["bm25"] == [True]
    assert captured["vector"] == [True]
    # Hybrid leg saw the recency kwarg with the value we passed.
    assert captured["hybrid"] == [365.0]


def test_pass_rate_by_tier_and_mode(conn):
    """Stage two Tier-1 queries — one passes only under vector, one only under
    BM25. Hybrid covers both, so hybrid pass rate is 2/2 while each single
    backend only hits 1/2. This is the exact failure mode the per-backend
    split is designed to surface.
    """
    vector_target = _seed_code(
        conn,
        path="src/v.py",
        text="A vector_target",
        vec=[1, 0, 0, 0],
    )
    bm25_target = _seed_code(
        conn,
        path="src/b.py",
        text="C bm25_target_token text",
        vec=[0, 0, 1, 0],
    )
    # Decoys at the default (no-pattern) embedding to push bm25_target out of
    # vector top-k for query 2.
    for i in range(3):
        _seed_code(conn, path=f"src/decoy{i}.py", text=f"decoy_{i}", vec=[0, 0, 0, 1])

    qs = [
        # q1: vector wins (chunk has pattern A); BM25 misses (AND of "A" and
        # the non-existent token in any chunk).
        SearchQuery(
            query="A nonexistent_xyz_token",
            tool="find_style_examples",
            tier=1,
            expect_any=[Expectation(chunk_id=vector_target)],
        ),
        # q2: BM25 wins (chunk text has the unique token); vector misses
        # because the 3 default-embedding decoys are closer to the
        # default-embedding query.
        SearchQuery(
            query="bm25_target_token",
            tool="find_style_examples",
            tier=1,
            expect_any=[Expectation(chunk_id=bm25_target)],
        ),
    ]
    rep = evaluate_search(conn, FakeEmbedder(), SqliteVecStore(conn), qs, k=2)
    p, t = rep.pass_rate(tier=1, mode="bm25")
    assert (p, t) == (1, 2)  # only q2 hits BM25
    p, t = rep.pass_rate(tier=1, mode="vector")
    assert (p, t) == (1, 2)  # only q1 hits vector
    p, t = rep.pass_rate(tier=1, mode="hybrid")
    assert (p, t) == (2, 2)  # union recovers both


def test_renderer_returns_non_zero_on_tier1_miss(conn):
    _seed_code(conn, path="src/x.py", text="A code", vec=[1, 0, 0, 0])
    sq = SearchQuery(
        query="A code",
        tool="find_style_examples",
        tier=1,
        expect_any=[Expectation(path_substr="this_does_not_exist")],
    )
    rep = evaluate_search(conn, FakeEmbedder(), SqliteVecStore(conn), [sq], modes=("hybrid",))
    console = Console(file=StringIO(), force_terminal=False, width=120)
    exit_code = render_search_result(rep, console, show_failures=False)
    assert exit_code == 1


def test_renderer_returns_zero_when_tier1_clean(conn):
    cid = _seed_code(conn, path="src/x.py", text="A code", vec=[1, 0, 0, 0])
    sq = SearchQuery(
        query="A code",
        tool="find_style_examples",
        tier=1,
        expect_any=[Expectation(chunk_id=cid)],
    )
    rep = evaluate_search(conn, FakeEmbedder(), SqliteVecStore(conn), [sq], modes=("hybrid",))
    console = Console(file=StringIO(), force_terminal=False, width=120)
    exit_code = render_search_result(rep, console, show_failures=False)
    assert exit_code == 0


def test_renderer_gates_on_hybrid_only_not_bm25_or_vector(conn):
    """BM25-only or vector-only Tier-1 misses must NOT fail the gate when
    hybrid still passes — NL queries can't realistically satisfy BM25's
    literal-token retrieval, but the production path is hybrid where the
    vector leg covers them."""
    target = _seed_code(conn, path="src/v.py", text="A vector_target", vec=[1, 0, 0, 0])
    # Decoys at the default no-pattern embedding so they crowd out target
    # in the bare-vector leg for a no-pattern query.
    for i in range(3):
        _seed_code(conn, path=f"src/d{i}.py", text=f"decoy_{i}", vec=[0, 0, 0, 1])

    # Query has the vector pattern (so vector hits target via pattern A)
    # but no shared token (so BM25 returns nothing). Hybrid passes; BM25 fails.
    sq = SearchQuery(
        query="A nonexistent_xyz_token",
        tool="find_style_examples",
        tier=1,
        expect_any=[Expectation(chunk_id=target)],
    )
    rep = evaluate_search(conn, FakeEmbedder(), SqliteVecStore(conn), [sq], k=2)

    bm25_passed, bm25_total = rep.pass_rate(tier=1, mode="bm25")
    hybrid_passed, hybrid_total = rep.pass_rate(tier=1, mode="hybrid")
    assert (bm25_passed, bm25_total) == (0, 1)  # BM25 fails its leg
    assert (hybrid_passed, hybrid_total) == (1, 1)  # hybrid still 100%

    console = Console(file=StringIO(), force_terminal=False, width=120)
    exit_code = render_search_result(rep, console, show_failures=False)
    assert exit_code == 0, (
        "Gate must not fail when BM25 misses but hybrid passes — "
        "BM25-only is a diagnostic column, not a CI target."
    )


def test_report_pass_rate_filters_correctly():
    """Direct exercise of SearchEvalReport.pass_rate without spinning up a DB."""
    from github_twin.eval.search_evals import QueryOutcome

    q1 = SearchQuery(
        query="x", tool="find_style_examples", tier=1, expect_any=[Expectation(path_substr="a")]
    )
    q2 = SearchQuery(
        query="y", tool="find_style_examples", tier=2, expect_any=[Expectation(path_substr="b")]
    )
    outcomes = [
        QueryOutcome(query=q1, mode="bm25", passed=True, top_hits=[]),
        QueryOutcome(query=q1, mode="vector", passed=False, top_hits=[]),
        QueryOutcome(query=q2, mode="bm25", passed=True, top_hits=[]),
        QueryOutcome(query=q2, mode="vector", passed=True, top_hits=[]),
    ]
    rep = SearchEvalReport(outcomes=outcomes, k=5)
    assert rep.pass_rate(tier=1, mode="bm25") == (1, 1)
    assert rep.pass_rate(tier=1, mode="vector") == (0, 1)
    assert rep.pass_rate(tier=2) == (2, 2)
    assert rep.pass_rate() == (3, 4)
