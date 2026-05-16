"""Console rendering for eval results."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from github_twin.eval.runner import (
    PredictEvalResult,
    ReviewEvalResult,
    class_distribution,
)
from github_twin.eval.search_evals import (
    ALL_MODES,
    Mode,
    SearchEvalReport,
    per_tool_pass_rate,
)

# Per-tier minimum-pass thresholds. Tier-1 failures should be loud.
_TIER_TARGET: dict[int, float] = {1: 1.0, 2: 0.85, 3: 0.70}


def render_review_result(result: ReviewEvalResult, console: Console) -> None:
    if result.n == 0:
        console.print("[yellow]No held-out review comments to evaluate.[/yellow]")
        return
    t = Table(title=f"Review-comment eval (n={result.n})")
    t.add_column("metric")
    t.add_column("baseline", justify="right")
    t.add_column("rag", justify="right")
    t.add_column("delta", justify="right")
    t.add_row(
        "mean cosine distance",
        f"{result.baseline_mean:.4f}",
        f"{result.rag_mean:.4f}",
        f"{result.delta:+.4f}",
    )
    console.print(t)
    if result.paired_t is not None and result.p_value is not None:
        verdict = (
            "[bold green]RAG significantly closer[/bold green]"
            if result.p_value < 0.05 and result.delta > 0
            else "[bold yellow]not significant[/bold yellow]"
        )
        console.print(
            f"Paired-t (one-sided, RAG < baseline): "
            f"t={result.paired_t:.3f}, p≈{result.p_value:.4f}  {verdict}"
        )
        if result.n < 30:
            console.print(
                "[dim](n<30; p-value uses a normal-approximation and should be "
                "treated as indicative rather than exact.)[/dim]"
            )


def render_predict_result(result: PredictEvalResult, console: Console) -> None:
    if result.n == 0:
        console.print("[yellow]No held-out PRs with decisions to evaluate.[/yellow]")
        return
    t = Table(title=f"PR outcome prediction (n={result.n})")
    t.add_column("metric")
    t.add_column("baseline", justify="right")
    t.add_column("rag", justify="right")
    t.add_row(
        "accuracy",
        f"{result.baseline_accuracy:.3f}",
        f"{result.rag_accuracy:.3f}",
    )
    for cls in result.baseline_f1:
        t.add_row(
            f"F1 / {cls}",
            f"{result.baseline_f1[cls]:.3f}",
            f"{result.rag_f1[cls]:.3f}",
        )
    console.print(t)
    dist = class_distribution(result.rows)
    if dist:
        biggest = max(dist.values()) / result.n
        console.print(
            f"Truth class distribution: {dist}  (majority-class baseline accuracy ≈ {biggest:.3f})"
        )


def render_search_result(
    report: SearchEvalReport,
    console: Console,
    *,
    show_failures: bool = True,
    failure_modes: tuple[Mode, ...] = ("hybrid",),
) -> int:
    """Print per-tier x per-backend pass rates, then Tier-1 failure breakdown.

    Returns a process exit code: 1 if Tier-1 **hybrid** (the production
    retrieval path) missed its 100% target, else 0. Tier-1 misses on the
    bm25/vector diagnostic columns log yellow (no CI gate); Tier-2/3
    misses on any column log yellow too.

    Rationale: BM25-only can't realistically satisfy NL queries that
    don't share tokens with the chunk text, but those same queries
    succeed under hybrid because the vector leg covers them. Gating
    on every mode would force authors to either rewrite the query bank
    around BM25's limitations or skip NL queries entirely.
    """
    if not report.outcomes:
        console.print("[yellow]No queries to evaluate.[/yellow]")
        return 0
    modes_present: list[Mode] = [m for m in ALL_MODES if any(o.mode == m for o in report.outcomes)]

    table = Table(title=f"Retrieval eval (k={report.k})")
    table.add_column("tier", justify="right")
    for m in modes_present:
        table.add_column(m, justify="right")
    table.add_column("target", justify="right")

    # Tier-1 gate runs against the `hybrid` column (the production path).
    # BM25-only and vector-only stay as diagnostic columns — coloring them
    # red on miss would force NL queries to satisfy the literal-token leg,
    # which BM25 can't realistically do. Coloring yellow flags regressions
    # without blocking CI.
    tier1_missed = False
    for tier in report.tiers():
        row = [str(tier)]
        target = _TIER_TARGET.get(tier, 0.0)
        for m in modes_present:
            passed, total = report.pass_rate(tier=tier, mode=m)
            rate = passed / total if total else 0.0
            cell = f"{passed}/{total} ({rate:.0%})"
            is_hybrid_t1_miss = tier == 1 and m == "hybrid" and rate < target
            if is_hybrid_t1_miss:
                cell = f"[red]{cell}[/red]"
                tier1_missed = True
            elif rate < target:
                cell = f"[yellow]{cell}[/yellow]"
            row.append(cell)
        row.append(f"≥{target:.0%}")
        table.add_row(*row)
    console.print(table)

    # Per-tool breakdown for the hybrid path — surfaces which tool's queries
    # are dragging the average down. Other modes are usually obvious from the
    # main table; one breakdown is enough for now.
    tool_table = Table(title="Per-tool (hybrid)")
    tool_table.add_column("tool")
    tool_table.add_column("pass", justify="right")
    for tool, (passed, total) in sorted(per_tool_pass_rate(report, "hybrid").items()):
        rate = passed / total if total else 0.0
        tool_table.add_row(tool, f"{passed}/{total} ({rate:.0%})")
    console.print(tool_table)

    if show_failures:
        for mode in failure_modes:
            failures = report.failures(tier=1, mode=mode)
            if not failures:
                continue
            console.print(f"\n[bold red]Tier-1 failures ({mode}):[/bold red]")
            for o in failures:
                console.print(f"  • [{o.query.tool}] {o.query.query!r}")
                for h in o.top_hits[: min(3, len(o.top_hits))]:
                    ctx = h.context or {}
                    label = (
                        ctx.get("path") or ctx.get("symbol_name") or h.artifact_source_url or "?"
                    )
                    console.print(f"      hit: {label}  (d={h.distance:.4f})")
                if not o.top_hits:
                    console.print("      (no hits returned)")

    return 1 if tier1_missed else 0
