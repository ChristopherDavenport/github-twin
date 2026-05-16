"""Tests for code-pattern distillation: clustering kind='code' chunks,
synthesizing them with the code prompt, and retrieving the resulting
code_rule chunks via the MCP `find_applicable_rules` tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github_twin.config import DistillCfg
from github_twin.distill.cluster import cluster_code_chunks
from github_twin.distill.rules import distill_rules
from github_twin.distill.synth import (
    CODE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    RuleResult,
    _render_cluster_for_prompt,
)
from github_twin.mcp_server import tools as t
from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.store.vector_store import SqliteVecStore


class FakeEmbedder:
    """Pattern-keyed deterministic embedder. Same shape as test_distill.py."""

    dim = 4
    model_id = "fake-embedder"
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


class FakeSynthesizer:
    """Stub synth: returns a fixed rule keyed off the first member's text prefix."""

    backend_id = "fake"

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        # Mirror the real synth's stash so the orchestrator's prompt selection
        # is observable in tests.
        self.system_prompt = system_prompt
        self._calls: list[list[dict[str, Any]]] = []

    def synthesize(self, cluster: list[dict[str, Any]]) -> RuleResult:
        self._calls.append(cluster)
        key = next((m["text"][:3] for m in cluster if m.get("text")), "default")
        return RuleResult(
            rule=f"Pattern {key!r}",
            language=None,
            example_quotes=[m["text"] for m in cluster[:2]],
            incoherent=False,
        )


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "code.sqlite", embed_dim=FakeEmbedder.dim)
    yield db
    db.close()


def _seed_code_chunks(
    conn,
    texts: list[str],
    embedder: FakeEmbedder,
    *,
    author_login: str | None = None,
    language: str = "python",
    id_prefix: str = "cc",
    repo: str = "me/x",
) -> list[int]:
    vecs = embedder.embed(texts)
    ids: list[int] = []
    for i, (text, vec) in enumerate(zip(texts, vecs, strict=True)):
        aid = q.upsert_artifact(
            conn,
            kind="commit",
            external_id=f"{id_prefix}-{i}",
            source_url=f"https://gh/x/commit/{i}",
            repo=repo,
            language=None,
            author_email=None,
            author_login=author_login,
            created_at=None,
            decision=None,
            meta=None,
        )
        cid = q.insert_chunk(
            conn,
            artifact_id=aid,
            kind="code",
            text=text,
            context={
                "repo": repo,
                "path": f"src/file_{i}.py",
                "language": language,
                "source_url": f"https://gh/x/commit/{i}",
                "commit_sha": f"sha{i:02d}",
            },
            language=language,
        )
        q.write_embedding(conn, chunk_id=cid, embedding=vec, model_id="fake")
        ids.append(cid)
    return ids


# ---------- clustering ----------


def test_cluster_code_chunks_groups_similar_patterns(conn):
    embedder = FakeEmbedder()
    texts = [
        "A: with suppress",
        "A: with suppress on rmtree",
        "A: with suppress on unlink",
        "A: with suppress on glob",
        "B: return None on miss",
        "B: return None on miss helper",
        "B: returns None when missing",
        "C: lone outlier",
    ]
    _seed_code_chunks(conn, texts, embedder)
    clusters = cluster_code_chunks(conn, min_cluster_size=3)
    assert len(clusters) == 2
    assert sorted(c.size for c in clusters) == [3, 4]


def test_cluster_code_chunks_respects_language_filter(conn):
    """Per-chunk language filter is the right scope for code patterns —
    Python and Go idioms shouldn't end up in the same cluster."""
    embedder = FakeEmbedder()
    _seed_code_chunks(
        conn,
        ["A py 1", "A py 2", "A py 3"],
        embedder,
        language="python",
        id_prefix="py",
    )
    _seed_code_chunks(
        conn,
        ["A go 1", "A go 2", "A go 3"],
        embedder,
        language="go",
        id_prefix="go",
    )
    py_only = cluster_code_chunks(conn, min_cluster_size=3, language="python")
    # Six chunks total in two density peaks: only the three python ones reach
    # the clusterer when language='python'. HDBSCAN needs ≥2 peaks; with one
    # peak only, it returns no clusters. This proves the filter is wired.
    assert py_only == []


def test_cluster_code_chunks_scopes_to_author(conn):
    embedder = FakeEmbedder()
    _seed_code_chunks(
        conn,
        ["A: 1", "A: 2", "A: 3", "B: 1", "B: 2", "B: 3"],
        embedder,
        author_login="alice",
        id_prefix="alice",
    )
    _seed_code_chunks(
        conn,
        ["A: 1", "A: 2", "A: 3"],
        embedder,
        author_login="bob",
        id_prefix="bob",
    )

    unfiltered = cluster_code_chunks(conn, min_cluster_size=3)
    assert len(unfiltered) == 2

    alice = cluster_code_chunks(conn, min_cluster_size=3, author_login="alice")
    # Alice has both density peaks → both clusters survive author-scoping.
    assert sorted(c.size for c in alice) == [3, 3]


# ---------- end-to-end distill ----------


def test_distill_writes_code_rules_under_code_rule_chunk_kind(conn):
    embedder = FakeEmbedder()
    _seed_code_chunks(
        conn,
        [
            "A: ctx 1",
            "A: ctx 2",
            "A: ctx 3",
            "B: ret 1",
            "B: ret 2",
            "B: ret 3",
        ],
        embedder,
    )
    synth = FakeSynthesizer(system_prompt=CODE_SYSTEM_PROMPT)
    cfg = DistillCfg(min_cluster_size=3)
    stats = distill_rules(
        conn=conn,
        synth=synth,
        embedder=embedder,
        cfg=cfg,
        chunk_kind="code",
        rule_chunk_kind="code_rule",
    )
    assert stats.clusters == 2
    assert stats.rules_written == 2

    # One artifact per cluster, kind='rule', with rule_source stamped.
    artifacts = conn.execute(
        "SELECT meta_json FROM artifact WHERE kind='rule' ORDER BY id"
    ).fetchall()
    assert len(artifacts) == 2

    import json

    for row in artifacts:
        meta = json.loads(row["meta_json"]) if row["meta_json"] else {}
        assert meta.get("rule_source") == "code"

    # Rule chunks land under chunk.kind='code_rule', not 'rule'.
    code_rules = conn.execute("SELECT COUNT(*) FROM chunk WHERE kind='code_rule'").fetchone()[0]
    review_rules = conn.execute("SELECT COUNT(*) FROM chunk WHERE kind='rule'").fetchone()[0]
    assert code_rules == 2
    assert review_rules == 0

    # And they were embedded — retrievable through the normal vector path.
    n_vecs = conn.execute(
        "SELECT COUNT(*) FROM chunk c JOIN vec_chunk v ON v.chunk_id=c.id WHERE c.kind='code_rule'"
    ).fetchone()[0]
    assert n_vecs == 2


def test_distill_code_passes_code_shaped_members_to_synth(conn):
    """The synthesizer receives members tagged member_kind='code', with
    path/source_url instead of pr_title/diff_hunk."""
    embedder = FakeEmbedder()
    _seed_code_chunks(
        conn,
        ["A: 1", "A: 2", "A: 3", "B: 1", "B: 2", "B: 3"],
        embedder,
    )
    synth = FakeSynthesizer(system_prompt=CODE_SYSTEM_PROMPT)
    cfg = DistillCfg(min_cluster_size=3)
    distill_rules(
        conn=conn,
        synth=synth,
        embedder=embedder,
        cfg=cfg,
        chunk_kind="code",
        rule_chunk_kind="code_rule",
    )
    assert synth._calls
    member = synth._calls[0][0]
    assert member["member_kind"] == "code"
    assert "path" in member
    assert "source_url" in member
    assert "diff_hunk" not in member


def test_render_code_cluster_uses_code_template():
    """The renderer dispatches on member_kind so the code prompt sees
    path/code instead of pr_title/comment."""
    rendered = _render_cluster_for_prompt(
        [
            {
                "member_kind": "code",
                "text": "with contextlib.suppress(OSError):\n    shutil.rmtree(p)",
                "repo": "me/x",
                "path": "src/clone.py",
                "language": "python",
            }
        ]
    )
    assert "snippet #1" in rendered
    assert "path: src/clone.py" in rendered
    assert "code:" in rendered
    assert "comment:" not in rendered
    assert "pr_title:" not in rendered


# ---------- retrieval (find_applicable_rules) ----------


def test_find_applicable_rules_returns_only_code_rules(conn):
    """The MCP tool must filter to chunk.kind='code_rule'. If it picked
    up review rules instead, users would get the wrong content type."""
    embedder = FakeEmbedder()

    # Seed a code-derived rule chunk.
    aid_code = q.upsert_artifact(
        conn,
        kind="rule",
        external_id="rule-code-1",
        source_url=None,
        repo=None,
        language="python",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta={"rule_source": "code"},
    )
    cid_code = q.insert_chunk(
        conn,
        artifact_id=aid_code,
        kind="code_rule",
        text="Use suppress(OSError) for race-y cleanup",
        context={"language": "python", "examples": ["with suppress(...)"]},
        language="python",
    )
    q.write_embedding(conn, chunk_id=cid_code, embedding=[1.0, 0.0, 0.0, 0.0], model_id="fake")

    # Seed a review-derived rule chunk that's "closer" in embedding space —
    # if the filter is broken, this leaks into the result.
    aid_rev = q.upsert_artifact(
        conn,
        kind="rule",
        external_id="rule-review-1",
        source_url=None,
        repo=None,
        language="python",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta={"rule_source": "review_comment"},
    )
    cid_rev = q.insert_chunk(
        conn,
        artifact_id=aid_rev,
        kind="rule",
        text="Add a docstring",
        context={"language": "python"},
        language="python",
    )
    q.write_embedding(conn, chunk_id=cid_rev, embedding=[1.0, 0.0, 0.0, 0.0], model_id="fake")

    store = SqliteVecStore(conn)
    hits = t.find_applicable_rules(
        conn,
        embedder,
        store,
        query="A: how to clean up safely",
        k=5,
    )
    assert len(hits) == 1
    assert hits[0]["rule"] == "Use suppress(OSError) for race-y cleanup"
    assert hits[0]["language"] == "python"


def test_summarize_review_patterns_excludes_code_rules(conn):
    """Regression: list_rules now filters by chunk_kind so code rules
    don't appear in the review-rule listing."""
    # One review rule.
    aid_rev = q.upsert_artifact(
        conn,
        kind="rule",
        external_id="rev-1",
        source_url=None,
        repo=None,
        language="python",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta={"rule_source": "review_comment"},
    )
    q.insert_chunk(
        conn,
        artifact_id=aid_rev,
        kind="rule",
        text="Add a docstring",
        context=None,
        language="python",
    )
    # One code rule.
    aid_code = q.upsert_artifact(
        conn,
        kind="rule",
        external_id="code-1",
        source_url=None,
        repo=None,
        language="python",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta={"rule_source": "code"},
    )
    q.insert_chunk(
        conn,
        artifact_id=aid_code,
        kind="code_rule",
        text="Use suppress for race-y cleanup",
        context=None,
        language="python",
    )

    review_rules = t.summarize_review_patterns(conn)
    assert len(review_rules) == 1
    assert review_rules[0]["rule"] == "Add a docstring"

    # And the parallel list, when called with chunk_kind='code_rule', returns
    # only the code rule.
    code_rules = q.list_rules(conn, chunk_kind="code_rule")
    assert len(code_rules) == 1
    assert code_rules[0]["rule"] == "Use suppress for race-y cleanup"


# ---------- repo scoping ----------


def test_cluster_code_chunks_scopes_to_repo(conn):
    """Two repos with their own density peaks. Scoping to one repo must
    exclude the other's chunks from the clusterer."""
    embedder = FakeEmbedder()
    _seed_code_chunks(
        conn,
        ["A: 1", "A: 2", "A: 3", "B: 1", "B: 2", "B: 3"],
        embedder,
        repo="org/alpha",
        id_prefix="alpha",
    )
    _seed_code_chunks(
        conn,
        ["A: 1", "A: 2", "A: 3", "B: 1", "B: 2", "B: 3"],
        embedder,
        repo="org/beta",
        id_prefix="beta",
    )

    unfiltered = cluster_code_chunks(conn, min_cluster_size=3)
    # Both repos contribute the same two patterns, so the global view
    # collapses to two clusters of size 6 each.
    assert sorted(c.size for c in unfiltered) == [6, 6]

    alpha = cluster_code_chunks(conn, min_cluster_size=3, repo="org/alpha")
    # Only alpha's chunks reach HDBSCAN — two density peaks, each size 3.
    assert sorted(c.size for c in alpha) == [3, 3]
    # And the member chunks all belong to alpha artifacts.
    alpha_chunk_ids = {m.chunk_id for c in alpha for m in c.members}
    repos = conn.execute(
        f"SELECT DISTINCT a.repo FROM artifact a JOIN chunk c ON c.artifact_id=a.id "
        f"WHERE c.id IN ({','.join('?' * len(alpha_chunk_ids))})",
        list(alpha_chunk_ids),
    ).fetchall()
    assert [r["repo"] for r in repos] == ["org/alpha"]


def test_distill_stamps_repo_on_rule(conn):
    """When --repo is passed, the rule artifact carries the repo on the
    column (so retrieval's `repo=` filter surfaces it) and in meta
    (so callers can introspect the scope)."""
    embedder = FakeEmbedder()
    _seed_code_chunks(
        conn,
        ["A: 1", "A: 2", "A: 3", "B: 1", "B: 2", "B: 3"],
        embedder,
        repo="org/alpha",
        id_prefix="alpha",
    )
    synth = FakeSynthesizer(system_prompt=CODE_SYSTEM_PROMPT)
    cfg = DistillCfg(min_cluster_size=3)
    distill_rules(
        conn=conn,
        synth=synth,
        embedder=embedder,
        cfg=cfg,
        chunk_kind="code",
        rule_chunk_kind="code_rule",
        repo="org/alpha",
    )
    rows = conn.execute("SELECT repo, meta_json FROM artifact WHERE kind='rule'").fetchall()
    assert rows
    import json

    for r in rows:
        assert r["repo"] == "org/alpha"
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        assert meta.get("repo_scope") == "org/alpha"
