"""Distill orchestrator.

For each cluster of review-comment chunks:
  1. Synthesize a rule via the configured backend (Claude or Ollama).
  2. Store the rule as `artifact(kind='rule', language=<dominant>)` with one
     chunk of `kind='rule'` carrying the rule text. The rule is embedded the
     same way every other chunk is, so `summarize_review_patterns` and other
     retrieval paths see it through one code path.

Re-running is idempotent: each cluster's rule artifact is keyed by a stable
identifier (`rule-<hash-of-member-ids>`), so re-clustering the same chunks
won't proliferate artifacts.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from github_twin.config import DistillCfg
from github_twin.distill.cluster import (
    Cluster,
    ClusterMember,
    cluster_code_chunks,
    cluster_review_comments,
)
from github_twin.distill.synth import RuleSynthesizer
from github_twin.embed import Embedder
from github_twin.store import queries as q
from github_twin.store.db import transaction

ChunkKind = Literal["review_comment", "code"]
RuleChunkKind = Literal["rule", "code_rule"]

log = logging.getLogger(__name__)


@dataclass
class DistillStats:
    clusters: int = 0
    rules_written: int = 0
    incoherent: int = 0
    failed: int = 0


def _cluster_external_id(cluster: Cluster) -> str:
    member_ids = sorted(m.chunk_id for m in cluster.members)
    payload = ",".join(str(i) for i in member_ids).encode()
    return "rule-" + hashlib.sha1(payload).hexdigest()[:16]


def _dominant_language(cluster: Cluster) -> str | None:
    langs = [m.language for m in cluster.members if m.language]
    if not langs:
        return None
    most, count = Counter(langs).most_common(1)[0]
    # Only commit a language tag if it dominates (≥ 60% of language-tagged members).
    if count / len(langs) >= 0.6:
        return most
    return None


def _member_for_prompt(member: ClusterMember, *, chunk_kind: ChunkKind) -> dict[str, Any]:
    ctx = member.context or {}
    if chunk_kind == "code":
        return {
            "member_kind": "code",
            "text": member.text,
            "path": ctx.get("path"),
            "repo": ctx.get("repo"),
            "source_url": ctx.get("source_url"),
            "language": member.language or ctx.get("language"),
        }
    return {
        "member_kind": "review",
        "text": member.text,
        "diff_hunk": ctx.get("diff_hunk"),
        "pr_title": ctx.get("pr_title"),
        "repo": ctx.get("repo"),
        "language": member.language,
    }


def distill_rules(
    *,
    conn: sqlite3.Connection,
    synth: RuleSynthesizer,
    embedder: Embedder,
    cfg: DistillCfg,
    target_id: int,
    author_login: str | None = None,
    chunk_kind: ChunkKind = "review_comment",
    rule_chunk_kind: RuleChunkKind = "rule",
    language: str | None = None,
    repo: str | None = None,
    report: Callable[[str], None] = lambda _: None,
) -> DistillStats:
    """Cluster chunks of `chunk_kind` and synthesize each into a rule.

    `rule_chunk_kind` is the kind stamped on the resulting rule chunk
    (`'rule'` for review-derived rules, `'code_rule'` for code-derived
    rules). The artifact kind is always `'rule'`; differentiation lives
    at the chunk level so retrieval can filter cheaply.

    `language` is honored for code clusters only (per-chunk language is
    set at chunk_diff time). For review clusters the parameter is ignored —
    review-comment language is heterogeneous within a reviewer's history.

    `repo` scopes clustering to one `'owner/name'` and is stamped on the
    resulting rule artifact so `find_applicable_rules(repo=...)` and the
    listing query both surface it correctly.
    """
    if chunk_kind == "code":
        clusters = cluster_code_chunks(
            conn,
            min_cluster_size=cfg.min_cluster_size,
            max_cluster_size=cfg.max_cluster_size,
            author_login=author_login,
            language=language,
            repo=repo,
        )
    else:
        clusters = cluster_review_comments(
            conn,
            min_cluster_size=cfg.min_cluster_size,
            max_cluster_size=cfg.max_cluster_size,
            author_login=author_login,
            repo=repo,
        )
    stats = DistillStats(clusters=len(clusters))
    scope_bits = []
    if author_login:
        scope_bits.append(f"author={author_login}")
    if repo:
        scope_bits.append(f"repo={repo}")
    if language and chunk_kind == "code":
        scope_bits.append(f"language={language}")
    scope = f" ({', '.join(scope_bits)})" if scope_bits else ""
    report(f"clustered into {len(clusters)} rule candidates{scope}")

    for cluster in clusters:
        prompt_cluster = [_member_for_prompt(m, chunk_kind=chunk_kind) for m in cluster.members]
        try:
            result = synth.synthesize(prompt_cluster)
        except Exception as exc:  # noqa: BLE001
            log.warning("synth failed for cluster %d: %s", cluster.cluster_id, exc)
            stats.failed += 1
            continue

        if result.incoherent or not result.rule.strip():
            log.info("cluster %d marked incoherent, skipping", cluster.cluster_id)
            stats.incoherent += 1
            continue

        rule_language = result.language or _dominant_language(cluster)
        external_id = _cluster_external_id(cluster)
        # Review comments key URLs under 'url'; code chunks under 'source_url'.
        member_urls = [
            (m.context or {}).get("url") or (m.context or {}).get("source_url")
            for m in cluster.members
        ]
        # `if (m.context or {}).get("repo")` guarantees non-None values, but
        # mypy can't follow that through the set comprehension — extract the
        # values explicitly so `sorted` has a `set[str]` to chew on.
        member_repos = sorted(
            {repo for m in cluster.members if (repo := (m.context or {}).get("repo")) is not None}
        )

        with transaction(conn):
            artifact_id = q.upsert_artifact(
                conn,
                target_id=target_id,
                kind="rule",
                external_id=external_id,
                source_url=None,
                repo=repo,
                language=rule_language,
                author_email=None,
                author_login=author_login,
                created_at=None,
                decision=None,
                meta={
                    "backend": synth.backend_id,
                    "cluster_size": cluster.size,
                    "example_quotes": result.example_quotes,
                    "member_chunk_ids": [m.chunk_id for m in cluster.members],
                    "member_urls": [u for u in member_urls if u],
                    "member_repos": member_repos,
                    "author_login": author_login,
                    "repo_scope": repo,
                    "rule_source": chunk_kind,
                },
            )
            q.delete_chunks_for_artifact(conn, artifact_id)
            chunk_id = q.insert_chunk(
                conn,
                artifact_id=artifact_id,
                kind=rule_chunk_kind,
                text=result.rule,
                context={
                    "language": rule_language,
                    "examples": result.example_quotes,
                    "member_chunk_ids": [m.chunk_id for m in cluster.members],
                },
                language=rule_language,
            )

        # Embed the rule so it's retrievable via the normal path.
        vec = embedder.embed([result.rule])[0]
        with transaction(conn):
            q.write_embedding(conn, chunk_id=chunk_id, embedding=vec, model_id=embedder.model_id)

        stats.rules_written += 1
        report(f"  rule [{rule_language or '*'}]: {result.rule}")
    return stats
