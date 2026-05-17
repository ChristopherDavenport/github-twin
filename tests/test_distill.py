"""Tests for the distill pipeline: clustering, synthesis protocol, and rule storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github_twin.config import DistillCfg
from github_twin.distill.cluster import cluster_review_comments
from github_twin.distill.rules import distill_rules
from github_twin.distill.synth import (
    ClaudeSynthesizer,
    GeminiSynthesizer,
    OllamaSynthesizer,
    RuleResult,
    _parse_json_response,
    make_synthesizer,
)
from github_twin.embed.base import Embedder
from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target

# ---------- helpers ----------


class FakeEmbedder:
    """Tiny embedder for deterministic tests. Returns one of three fixed vectors
    based on a single character in the input — lets tests fabricate clusters."""

    dim = 4
    model_id = "fake-embedder"

    PATTERNS = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0, 0.0],
        "C": [0.0, 0.0, 1.0, 0.0],
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            for k, v in self.PATTERNS.items():
                if k in t:
                    out.append(list(v))
                    break
            else:
                out.append([0.0, 0.0, 0.0, 1.0])
        return out


class FakeSynthesizer:
    backend_id = "fake"

    def __init__(self, mapping: dict[str, RuleResult] | None = None) -> None:
        self._mapping = mapping or {}

    def synthesize(self, cluster: list[dict[str, Any]]) -> RuleResult:
        # Use the first non-empty cluster text as a key
        key = next((m["text"][:3] for m in cluster if m.get("text")), "default")
        if key in self._mapping:
            return self._mapping[key]
        return RuleResult(
            rule=f"Prefer pattern around {key!r}",
            language=None,
            example_quotes=[m["text"] for m in cluster[:2]],
            incoherent=False,
        )


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=FakeEmbedder.dim)
    seed_target(db)
    yield db
    db.close()


def _seed_review_chunks(
    conn,
    texts: list[str],
    embedder: Embedder,
    *,
    author_login: str | None = None,
    id_prefix: str = "rc",
    repo: str = "me/x",
) -> list[int]:
    vecs = embedder.embed(texts)
    ids: list[int] = []
    for i, (text, vec) in enumerate(zip(texts, vecs, strict=True)):
        aid = q.upsert_artifact(
            conn,
            target_id=1,
            kind="review_comment",
            external_id=f"{id_prefix}-{i}",
            source_url=f"https://example/{i}",
            repo=repo,
            language="scala",
            author_email=None,
            author_login=author_login,
            created_at=None,
            decision=None,
            meta={"pr_title": f"PR {i}"},
        )
        cid = q.insert_chunk(
            conn,
            artifact_id=aid,
            kind="review_comment",
            text=text,
            context={"pr_title": f"PR {i}", "repo": repo, "url": f"https://example/{i}"},
            language="scala",
        )
        q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="fake")
        ids.append(cid)
    return ids


# ---------- clustering ----------


def test_cluster_groups_similar_chunks(conn):
    embedder = FakeEmbedder()
    # 4 chunks centered on pattern A, 3 on pattern B, 1 noise on C
    texts = [
        "A: avoid Future",
        "A: avoid blocking",
        "A: use IO instead",
        "A: prefer cats-effect",
        "B: name it descriptively",
        "B: rename for clarity",
        "B: better name please",
        "C: lone outlier",
    ]
    _seed_review_chunks(conn, texts, embedder)
    clusters = cluster_review_comments(conn, min_cluster_size=3)
    assert len(clusters) == 2
    sizes = sorted(c.size for c in clusters)
    assert sizes == [3, 4]


def test_cluster_drops_too_small(conn):
    embedder = FakeEmbedder()
    _seed_review_chunks(conn, ["A one", "A two"], embedder)  # 2 < min=3
    assert cluster_review_comments(conn, min_cluster_size=3) == []


def test_cluster_dropped_when_under_min_total(conn):
    """Below min_cluster_size total, return empty without invoking HDBSCAN."""
    embedder = FakeEmbedder()
    _seed_review_chunks(conn, ["A x"], embedder)
    assert cluster_review_comments(conn, min_cluster_size=3) == []


# ---------- synth parsing ----------


def test_parse_json_response_strips_prose():
    text = """Sure! Here's the rule:
{
  "rule": "Prefer composition over inheritance",
  "language": "scala",
  "example_quotes": ["just compose it", "no inheritance please"],
  "incoherent": false
}
Hope that helps!"""
    r = _parse_json_response(text)
    assert r.rule == "Prefer composition over inheritance"
    assert r.language == "scala"
    assert r.example_quotes == ["just compose it", "no inheritance please"]
    assert r.incoherent is False


def test_parse_json_response_marks_incoherent():
    r = _parse_json_response('{"rule": "", "incoherent": true, "example_quotes": []}')
    assert r.incoherent is True


def test_parse_json_response_caps_examples():
    r = _parse_json_response('{"rule": "x", "example_quotes": ["a", "b", "c", "d", "e"]}')
    assert len(r.example_quotes) == 3


# ---------- end-to-end distill ----------


def test_distill_writes_rules_and_makes_them_retrievable(conn):
    embedder = FakeEmbedder()
    texts = [
        "A: avoid Future",
        "A: avoid blocking",
        "A: use IO instead",
        "B: name it descriptively",
        "B: rename for clarity",
        "B: better name please",
    ]
    _seed_review_chunks(conn, texts, embedder)
    synth = FakeSynthesizer()
    cfg = DistillCfg(min_cluster_size=3)
    stats = distill_rules(conn=conn, synth=synth, embedder=embedder, cfg=cfg, target_id=1)
    assert stats.clusters == 2
    assert stats.rules_written == 2
    rules = q.list_rules(conn)
    assert len(rules) == 2
    # Each rule must have associated examples drawn from its cluster
    assert all(len(r["examples"]) >= 1 for r in rules)
    # Rule chunks were embedded and are retrievable through the normal path
    n_rule_vectors = conn.execute(
        "SELECT COUNT(*) FROM chunk c JOIN vec_chunk v ON v.chunk_id=c.id WHERE c.kind='rule'"
    ).fetchone()[0]
    assert n_rule_vectors == 2


def test_distill_skips_incoherent_clusters(conn):
    """HDBSCAN requires 2+ density peaks to form any cluster at all."""
    embedder = FakeEmbedder()
    texts = ["A one", "A two", "A three", "A four", "B one", "B two", "B three", "B four"]
    _seed_review_chunks(conn, texts, embedder)
    # Synthesizer marks every cluster incoherent.
    synth = FakeSynthesizer(
        mapping={
            "A o": RuleResult(rule="", language=None, example_quotes=[], incoherent=True),
            "B o": RuleResult(rule="", language=None, example_quotes=[], incoherent=True),
        }
    )
    cfg = DistillCfg(min_cluster_size=3)
    stats = distill_rules(conn=conn, synth=synth, embedder=embedder, cfg=cfg, target_id=1)
    assert stats.clusters == 2
    assert stats.rules_written == 0
    assert stats.incoherent == 2


def test_make_synthesizer_picks_claude_when_only_anthropic_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = make_synthesizer(claude_model="claude-test", gemini_model="g", ollama_model="o")
    assert isinstance(s, ClaudeSynthesizer)


def test_make_synthesizer_picks_gemini_when_only_google_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    s = make_synthesizer(claude_model="c", gemini_model="gemini-test", ollama_model="o")
    assert isinstance(s, GeminiSynthesizer)
    assert s.backend_id == "gemini:gemini-test"


def test_make_synthesizer_falls_back_to_ollama_when_no_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = make_synthesizer(claude_model="c", gemini_model="g", ollama_model="llama-test")
    assert isinstance(s, OllamaSynthesizer)
    assert s.backend_id == "ollama:llama-test"


def test_make_synthesizer_forced_gemini_errors_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        make_synthesizer(claude_model="c", gemini_model="g", ollama_model="o", prefer="gemini")


def test_make_synthesizer_claude_beats_gemini_in_auto(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    s = make_synthesizer(claude_model="c", gemini_model="g", ollama_model="o")
    assert isinstance(s, ClaudeSynthesizer)


def test_cluster_scopes_to_author_login(conn):
    """In org-mode you cluster per-reviewer so one author's chunks form their
    own rule set instead of getting averaged into a 'house style'."""
    embedder = FakeEmbedder()
    # Alice has a clear pattern-A cluster (4 chunks).
    _seed_review_chunks(
        conn,
        ["A: avoid Future", "A: avoid blocking", "A: use IO", "A: prefer cats-effect"],
        embedder,
        author_login="alice",
        id_prefix="alice",
    )
    # Bob has a pattern-B cluster (3 chunks).
    _seed_review_chunks(
        conn,
        ["B: rename it", "B: rename for clarity", "B: better name please"],
        embedder,
        author_login="bob",
        id_prefix="bob",
    )

    # Unfiltered → both clusters (two density peaks: A and B).
    unfiltered = cluster_review_comments(conn, min_cluster_size=3)
    assert len(unfiltered) == 2
    assert sorted(c.size for c in unfiltered) == [3, 4]

    # Filtered to alice → only her cluster reaches HDBSCAN. With one density
    # peak, HDBSCAN returns no clusters (it needs ≥2 to form any).
    alice_only = cluster_review_comments(conn, min_cluster_size=3, author_login="alice")
    assert len(alice_only) == 0

    # Filtered to a stranger → empty corpus.
    nobody = cluster_review_comments(conn, min_cluster_size=3, author_login="not-a-real-user")
    assert nobody == []


def test_distill_stamps_author_login_on_rule(conn):
    embedder = FakeEmbedder()
    # Two density peaks under alice so HDBSCAN actually clusters.
    _seed_review_chunks(
        conn,
        [
            "A: avoid Future",
            "A: avoid blocking",
            "A: use IO",
            "B: rename it",
            "B: rename for clarity",
            "B: better name",
        ],
        embedder,
        author_login="alice",
        id_prefix="alice",
    )
    _seed_review_chunks(
        conn,
        ["C: log level", "C: tune logging", "C: less noise"],
        embedder,
        author_login="bob",
        id_prefix="bob",
    )

    cfg = DistillCfg(min_cluster_size=3)
    stats = distill_rules(
        conn=conn,
        synth=FakeSynthesizer(),
        embedder=embedder,
        cfg=cfg,
        target_id=1,
        author_login="alice",
    )
    assert stats.rules_written >= 1
    rows = conn.execute("SELECT author_login FROM artifact WHERE kind='rule'").fetchall()
    assert rows  # at least one rule
    assert all(r["author_login"] == "alice" for r in rows)


def test_distill_idempotent_external_ids(conn):
    """Re-running on the same clusters shouldn't proliferate artifacts."""
    embedder = FakeEmbedder()
    _seed_review_chunks(
        conn,
        ["A 1", "A 2", "A 3", "A 4", "B 1", "B 2", "B 3", "B 4"],
        embedder,
    )
    cfg = DistillCfg(min_cluster_size=3)
    distill_rules(conn=conn, synth=FakeSynthesizer(), embedder=embedder, cfg=cfg, target_id=1)
    n1 = conn.execute("SELECT COUNT(*) FROM artifact WHERE kind='rule'").fetchone()[0]
    distill_rules(conn=conn, synth=FakeSynthesizer(), embedder=embedder, cfg=cfg, target_id=1)
    n2 = conn.execute("SELECT COUNT(*) FROM artifact WHERE kind='rule'").fetchone()[0]
    assert n1 == n2 == 2


def test_cluster_review_comments_scopes_to_repo(conn):
    """Mirror of the code-path repo test, on the review pipeline. Confirms
    the loader generalization didn't break the review path."""
    embedder = FakeEmbedder()
    _seed_review_chunks(
        conn,
        [
            "A: avoid Future",
            "A: avoid blocking",
            "A: use IO",
            "B: rename it",
            "B: rename for clarity",
            "B: better name",
        ],
        embedder,
        repo="org/alpha",
        id_prefix="alpha",
    )
    _seed_review_chunks(
        conn,
        [
            "A: avoid Future",
            "A: avoid blocking",
            "A: use IO",
            "B: rename it",
            "B: rename for clarity",
            "B: better name",
        ],
        embedder,
        repo="org/beta",
        id_prefix="beta",
    )

    unfiltered = cluster_review_comments(conn, min_cluster_size=3)
    assert sorted(c.size for c in unfiltered) == [6, 6]

    alpha = cluster_review_comments(conn, min_cluster_size=3, repo="org/alpha")
    assert sorted(c.size for c in alpha) == [3, 3]
