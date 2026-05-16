"""RAG-vs-baseline eval runner.

Two evaluations, sharing the same harness:

1. **reviews** — for held-out review comments, prompt the LLM with and
   without retrieved examples, score outputs by cosine distance to the
   ground-truth comment. Lower distance is better; we compare means via
   a paired-t test.

2. **predictions** — for held-out PRs with known decisions, ask the LLM
   to predict and call `predict_review_outcome` for the RAG path. Score
   by accuracy + per-class F1.

The judge embedder defaults to a *different* model than the retrieval
embedder, so the score isn't measuring how well retrieval clusters its
own outputs. Pass `--judge-backend` to override.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from github_twin.embed.base import Embedder
from github_twin.eval.holdout import (
    PRExample,
    iter_held_out_prs,
    iter_held_out_review_comments,
)
from github_twin.eval.llm import TextLLM
from github_twin.mcp_server.tools import (
    DECISION_KINDS,
    find_review_comments,
    predict_review_outcome,
)
from github_twin.store.vector_store import VectorStore

log = logging.getLogger(__name__)


# ---------- review-comment eval ----------


REVIEW_SYSTEM = (
    "You are leaving a single code review comment on the new code in the "
    "diff hunk below. Match the tone, length, and concern set of the past "
    "reviewer whose history you're modeling. Output the comment text only — "
    "no preamble, no markdown headers."
)


def _baseline_review_prompt(hunk: str) -> str:
    return f"Diff hunk under review:\n```\n{hunk}\n```\n\nYour review comment:"


def _rag_review_prompt(hunk: str, examples: list[dict[str, Any]]) -> str:
    block = "\n\n".join(
        f"Past hunk:\n{(e.get('diff_hunk_context') or '').strip()[:600]}\n"
        f"Past comment:\n{(e.get('comment') or '').strip()}"
        for e in examples
        if e.get("comment")
    )
    return (
        f"Examples of past reviews you've written on similar diffs:\n{block}\n\n"
        f"---\nNew diff hunk under review:\n```\n{hunk}\n```\n\n"
        f"Your review comment (consistent with the style above):"
    )


@dataclass
class ReviewEvalRow:
    artifact_id: int
    baseline_dist: float
    rag_dist: float


@dataclass
class ReviewEvalResult:
    n: int
    baseline_mean: float
    rag_mean: float
    delta: float  # baseline_mean - rag_mean; positive == RAG wins
    paired_t: float | None
    p_value: float | None  # one-sided "RAG distance is lower"
    rows: list[ReviewEvalRow] = field(default_factory=list)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    return max(0.0, 1.0 - dot / (na * nb))


def _paired_t_one_sided(deltas: list[float]) -> tuple[float | None, float | None]:
    """Paired-t on `baseline - rag` against H1: mean > 0 (RAG distances lower)."""
    n = len(deltas)
    if n < 2:
        return None, None
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return (math.inf if mean > 0 else 0.0), (0.0 if mean > 0 else 1.0)
    t = mean / (sd / math.sqrt(n))
    # One-sided p via the standard normal approximation. For n>30 the
    # difference from the true t-distribution is small enough to be honest;
    # for smaller n we just call it "approximate" in the report.
    p = 1.0 - _phi(t)
    return t, p


def _phi(x: float) -> float:
    """Standard normal CDF via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def evaluate_reviews(
    conn: sqlite3.Connection,
    *,
    retriever_embedder: Embedder,
    judge_embedder: Embedder,
    store: VectorStore,
    llm: TextLLM,
    since: str,
    author_login: str | None = None,
    repo: str | None = None,
    limit: int | None = None,
    k: int = 5,
    progress: Callable[[str], None] = lambda _: None,
) -> ReviewEvalResult:
    """Run the review-comment eval and aggregate paired distances.

    `author_login` (org-mode): scopes both the holdout AND the RAG retrieval
    to one reviewer. Without it, org-mode runs will blur many reviewers'
    voices together. `repo` does the same for a single 'owner/name'.
    """
    rows: list[ReviewEvalRow] = []
    baseline_dists: list[float] = []
    rag_dists: list[float] = []

    examples = list(
        iter_held_out_review_comments(
            conn,
            since=since,
            author_login=author_login,
            repo=repo,
            limit=limit,
        )
    )
    if not examples:
        return ReviewEvalResult(0, 0.0, 0.0, 0.0, None, None, [])

    for i, ex in enumerate(examples, 1):
        try:
            baseline_text = llm.complete(
                system=REVIEW_SYSTEM,
                user=_baseline_review_prompt(ex.diff_hunk),
            )
            rag_hits = find_review_comments(
                conn,
                retriever_embedder,
                store,
                diff_hunk=ex.diff_hunk,
                language=ex.language,
                author_login=author_login,
                repo=repo,
                k=k,
            )
            # Drop hits at or after the cutoff so we don't leak the holdout.
            rag_hits = _filter_post_cutoff(conn, rag_hits, since)
            rag_text = llm.complete(
                system=REVIEW_SYSTEM,
                user=_rag_review_prompt(ex.diff_hunk, rag_hits),
            )
            truth_vec = judge_embedder.embed([ex.truth_comment])[0]
            base_vec, rag_vec = judge_embedder.embed([baseline_text, rag_text])
            d_base = _cosine_distance(base_vec, truth_vec)
            d_rag = _cosine_distance(rag_vec, truth_vec)
        except Exception as exc:  # noqa: BLE001
            log.warning("eval skip artifact %d: %s", ex.artifact_id, exc)
            continue
        rows.append(ReviewEvalRow(ex.artifact_id, d_base, d_rag))
        baseline_dists.append(d_base)
        rag_dists.append(d_rag)
        progress(f"  [{i}/{len(examples)}] base={d_base:.3f} rag={d_rag:.3f}")

    n = len(rows)
    if n == 0:
        return ReviewEvalResult(0, 0.0, 0.0, 0.0, None, None, [])
    base_mean = sum(baseline_dists) / n
    rag_mean = sum(rag_dists) / n
    deltas = [b - r for b, r in zip(baseline_dists, rag_dists, strict=True)]
    t, p = _paired_t_one_sided(deltas)
    return ReviewEvalResult(n, base_mean, rag_mean, base_mean - rag_mean, t, p, rows)


def _filter_post_cutoff(
    conn: sqlite3.Connection, hits: list[dict[str, Any]], cutoff: str
) -> list[dict[str, Any]]:
    """Drop hits whose underlying artifact was created at or after `cutoff`.

    Retrieval doesn't know about the holdout; we enforce it here by joining
    on the artifact URL (the only field carried through find_review_comments).
    """
    if not hits:
        return hits
    urls = [h.get("url") for h in hits if h.get("url")]
    if not urls:
        return hits
    placeholders = ",".join("?" * len(urls))
    rows = conn.execute(
        f"SELECT source_url, created_at FROM artifact WHERE source_url IN ({placeholders})",
        urls,
    ).fetchall()
    bad = {r["source_url"] for r in rows if r["created_at"] and r["created_at"] >= cutoff}
    return [h for h in hits if h.get("url") not in bad]


# ---------- prediction eval ----------


PREDICT_SYSTEM = (
    "You are predicting how a code reviewer will respond to a pull request. "
    "Reply with exactly one token: approved, changes_requested, or commented. "
    "No prose."
)


def _baseline_predict_prompt(ex: PRExample) -> str:
    body = (ex.body or "").strip()[:1500]
    return f"PR title: {ex.title}\nPR body:\n{body}\n\nPredicted decision:"


def _normalize_decision(s: str) -> str | None:
    s = (s or "").strip().lower().splitlines()[0] if s else ""
    s = s.split()[0] if s else ""
    s = s.strip(".,:;\"'`*")
    if s in DECISION_KINDS:
        return s
    if s in ("changes", "change", "request_changes", "request"):
        return "changes_requested"
    if s in ("approve",):
        return "approved"
    if s in ("comment",):
        return "commented"
    return None


@dataclass
class PredictEvalRow:
    artifact_id: int
    truth: str
    baseline: str | None
    rag: str | None


@dataclass
class PredictEvalResult:
    n: int
    baseline_accuracy: float
    rag_accuracy: float
    baseline_f1: dict[str, float]  # per-class F1
    rag_f1: dict[str, float]
    rows: list[PredictEvalRow] = field(default_factory=list)


def _per_class_f1(truth: list[str], pred: list[str | None]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in DECISION_KINDS:
        tp = sum(1 for t, p in zip(truth, pred, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(truth, pred, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(truth, pred, strict=True) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[cls] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return out


def evaluate_predictions(
    conn: sqlite3.Connection,
    *,
    retriever_embedder: Embedder,
    store: VectorStore,
    llm: TextLLM,
    since: str,
    author_login: str | None = None,
    repo: str | None = None,
    limit: int | None = None,
    k: int = 20,
    progress: Callable[[str], None] = lambda _: None,
) -> PredictEvalResult:
    """When `author_login` is set, truth comes from `meta.reviewer_decisions`
    and the RAG path scopes `predict_review_outcome` to the same reviewer.
    `repo` scopes both holdout and RAG to a single 'owner/name'."""
    examples = list(
        iter_held_out_prs(
            conn,
            since=since,
            author_login=author_login,
            repo=repo,
            limit=limit,
        )
    )
    if not examples:
        return PredictEvalResult(0, 0.0, 0.0, {}, {}, [])

    rows: list[PredictEvalRow] = []
    truths: list[str] = []
    base_preds: list[str | None] = []
    rag_preds: list[str | None] = []

    for i, ex in enumerate(examples, 1):
        try:
            raw_base = llm.complete(
                system=PREDICT_SYSTEM,
                user=_baseline_predict_prompt(ex),
                max_tokens=8,
            )
            base = _normalize_decision(raw_base)
            text = f"{ex.title}\n\n{(ex.body or '')[:1500]}"
            rag_out = predict_review_outcome(
                conn,
                retriever_embedder,
                store,
                diff_or_summary=text,
                author_login=author_login,
                repo=repo,
                k=k,
            )
            rag = rag_out["prediction"] if rag_out["prediction"] in DECISION_KINDS else None
        except Exception as exc:  # noqa: BLE001
            log.warning("predict eval skip artifact %d: %s", ex.artifact_id, exc)
            continue
        truths.append(ex.truth_decision)
        base_preds.append(base)
        rag_preds.append(rag)
        rows.append(PredictEvalRow(ex.artifact_id, ex.truth_decision, base, rag))
        progress(f"  [{i}/{len(examples)}] truth={ex.truth_decision} base={base} rag={rag}")

    n = len(rows)
    if n == 0:
        return PredictEvalResult(0, 0.0, 0.0, {}, {}, [])
    base_acc = sum(1 for t, p in zip(truths, base_preds, strict=True) if t == p) / n
    rag_acc = sum(1 for t, p in zip(truths, rag_preds, strict=True) if t == p) / n
    return PredictEvalResult(
        n=n,
        baseline_accuracy=base_acc,
        rag_accuracy=rag_acc,
        baseline_f1=_per_class_f1(truths, base_preds),
        rag_f1=_per_class_f1(truths, rag_preds),
        rows=rows,
    )


# ---------- exposed for the CLI report ----------


def class_distribution(rows: Iterable[PredictEvalRow]) -> dict[str, int]:
    return dict(Counter(r.truth for r in rows))
