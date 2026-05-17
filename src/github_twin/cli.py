"""github-twin command line."""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import tomllib
from pathlib import Path
from typing import Any, Literal

import typer
from rich.console import Console
from rich.table import Table

from github_twin._logging import cap_noisy_loggers, install_secret_redaction
from github_twin.config import Config, EmbedCfg, load_config
from github_twin.distill.rules import distill_rules
from github_twin.distill.synth import CODE_SYSTEM_PROMPT, SYSTEM_PROMPT, make_synthesizer
from github_twin.embed import Embedder, make_embedder
from github_twin.eval.holdout import count_eligible
from github_twin.eval.llm import make_text_llm
from github_twin.eval.report import (
    render_predict_result,
    render_review_result,
    render_search_result,
)
from github_twin.eval.runner import evaluate_predictions, evaluate_reviews
from github_twin.eval.search_evals import ALL_MODES, evaluate_search, load_queries
from github_twin.ingest.clone import prune_cache
from github_twin.ingest.github_client import GitHubClient
from github_twin.ingest.repos import enumerate_org_repos
from github_twin.observability import init_otel
from github_twin.pipeline import IdentityMissingError, run_embed, run_ingest, run_summarize
from github_twin.store import queries as q
from github_twin.store.db import open_db, transaction
from github_twin.store.query_expansion import expander_from_config, make_expander
from github_twin.store.vector_store import make_vector_store
from github_twin.target import (
    AmbiguousTargetError,
    Target,
    discover_org,
    discover_repo,
    discover_user,
    load_target,
    load_targets,
    maybe_discover_repo,
    save_target,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
clones_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Manage the persistent clone cache (only used when cache_clones=true).",
)
eval_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Held-out evaluation: RAG vs. base-LLM accuracy on review / prediction tasks.",
)
auth_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Acquire and manage a GitHub OAuth token via device flow.",
)
targets_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Manage the targets (user / org / repo) tracked in this DB.",
)
app.add_typer(clones_app, name="clones")
app.add_typer(eval_app, name="eval")
app.add_typer(auth_app, name="auth")
app.add_typer(targets_app, name="targets")
console = Console()
log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")
    cap_noisy_loggers()
    install_secret_redaction()


def _ctx(config_path: Path | None) -> tuple[Config, sqlite3.Connection]:
    cfg = load_config(config_path)
    conn = open_db(cfg.paths.db_path, cfg.embed.dim)
    return cfg, conn


def _resolve_target_arg(conn: sqlite3.Connection, target_arg: str | None) -> Target | None:
    """Translate `--target NAME` into a Target row. None means
    "no narrowing" and the caller decides what that implies (often:
    iterate all)."""
    if target_arg is None:
        return None
    try:
        t = load_target(conn, name=target_arg)
    except AmbiguousTargetError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    if t is None:
        console.print(
            f"[red]No target named {target_arg!r}. "
            "Run `gt targets list` to see what's in this DB.[/red]"
        )
        raise typer.Exit(2)
    return t


def _resolve_embed_defaults(
    backend: str | None,
    model: str | None,
    dim: int | None,
) -> tuple[str, str, int]:
    """Fill in backend-aware defaults for the three embed flags.

    Returns `(backend, model, dim)` with all three populated. Raises
    `typer.BadParameter` for unknown backends or for
    sentence_transformers without an explicit model+dim (no safe
    default — the model and its output dim are coupled).
    """
    backend = (backend or "ollama").strip()
    if backend == "ollama":
        return "ollama", model or "nomic-embed-text", dim or 768
    if backend == "gemini":
        return "gemini", model or "gemini-embedding-001", dim or 3072
    if backend == "sentence_transformers":
        if not model or not dim:
            raise typer.BadParameter(
                "--embed-model and --embed-dim are required when "
                "--embed-backend=sentence_transformers (e.g. "
                "BAAI/bge-small-en-v1.5 / 384)."
            )
        return "sentence_transformers", model, dim
    raise typer.BadParameter(
        f"Unknown --embed-backend: {backend!r}. Pick from: ollama, gemini, sentence_transformers."
    )


def _persist_embed_config(
    config_path: Path,
    backend: str,
    model: str,
    dim: int,
) -> None:
    """Write a `[embed]` block into the given config.toml. See module docs."""
    new_block = f'[embed]\nbackend = "{backend}"\nmodel = "{model}"\ndim = {int(dim)}\n'
    if not config_path.exists():
        config_path.write_text(new_block, encoding="utf-8")
        return
    with config_path.open("rb") as f:
        existing_data = tomllib.load(f)
    existing = existing_data.get("embed", {})
    if (
        existing.get("backend") == backend
        and existing.get("model") == model
        and int(existing.get("dim", -1)) == int(dim)
    ):
        return
    if not existing:
        current = config_path.read_text(encoding="utf-8")
        sep = "" if current.endswith("\n") else "\n"
        config_path.write_text(current + sep + "\n" + new_block, encoding="utf-8")
        return
    raise typer.BadParameter(
        f"Existing {config_path} has [embed] with backend="
        f"{existing.get('backend')!r}, model={existing.get('model')!r}, "
        f"dim={existing.get('dim')!r}; flags ask for backend={backend!r}, "
        f"model={model!r}, dim={dim}. Edit the file directly to change it."
    )


def _report(msg: str) -> None:
    console.print(msg)


def _run_ingest_safely(
    cfg: Config,
    conn: sqlite3.Connection,
    **kw: Any,
) -> None:
    try:
        run_ingest(cfg, conn, report=_report, **kw)
    except IdentityMissingError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


# ---------- Typer commands ----------


def _version_callback(value: bool) -> None:
    if value:
        from github_twin import __version__

        console.print(__version__)
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    _setup_logging(verbose)
    init_otel()


@app.command()
def init(
    kind: str = typer.Option(
        "auto",
        "--kind",
        help=(
            "'auto' (default): pick repo-mode if pwd is a github.com working tree, "
            "else user-mode. 'user' | 'org' | 'repo' to force a kind. Additive: "
            "each `gt init` adds another target to the DB."
        ),
    ),
    org: str | None = typer.Option(None, "--org", help="Org login. Required when --kind=org."),
    repo: str | None = typer.Option(
        None,
        "--repo",
        help=(
            "'owner/name' for --kind=repo. Falls back to pwd .git/config origin URL when omitted."
        ),
    ),
    embed_backend: str | None = typer.Option(
        None,
        "--embed-backend",
        help=(
            "Stamp embedder backend into config.toml: 'ollama' (default) | "
            "'gemini' (remote, needs GEMINI_API_KEY/GOOGLE_API_KEY) | "
            "'sentence_transformers' (requires --embed-model + --embed-dim)."
        ),
    ),
    embed_model: str | None = typer.Option(
        None,
        "--embed-model",
        help="Embedder model. Defaults per backend (nomic-embed-text / gemini-embedding-001).",
    ),
    embed_dim: int | None = typer.Option(
        None,
        "--embed-dim",
        help="Embedding dimension. Defaults per backend (768 ollama / 3072 gemini).",
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config.toml"),
) -> None:
    """Add a target (user / org / repo) to this DB.

    Additive — call multiple times to layer targets in one DB (e.g. your
    user-mode + multiple orgs). Re-running with the same (kind, name)
    refreshes that target in place.
    """
    if embed_backend is not None or embed_model is not None or embed_dim is not None:
        resolved_backend, resolved_model, resolved_dim = _resolve_embed_defaults(
            embed_backend, embed_model, embed_dim
        )
        target_config = config if config is not None else Path("config.toml")
        _persist_embed_config(target_config, resolved_backend, resolved_model, resolved_dim)
        console.print(
            f"[dim]Embed config written to {target_config}: "
            f"backend={resolved_backend} model={resolved_model} dim={resolved_dim}[/dim]"
        )
    cfg, conn = _ctx(config)
    kind = kind.lower()

    if kind == "auto":
        with GitHubClient() as gh:
            auto = maybe_discover_repo(gh)
            if auto is not None:
                target, metadata = auto
                with transaction(conn):
                    target = save_target(conn, target)
                    assert target.id is not None
                    q.upsert_repo(conn, target_id=target.id, **metadata)
                console.print(
                    f"[bold]Added:[/bold] repo {target.name} "
                    f"(id {target.external_id}) [dim](auto-detected)[/dim]"
                )
                console.print(f"Default branch: {metadata['default_branch']}")
                console.print(f"\nDB: {cfg.paths.db_path}")
                return
        kind = "user"
        console.print("[dim]No github.com .git found; falling back to user mode.[/dim]")

    if kind == "user":
        with GitHubClient() as gh:
            target = discover_user(gh, cfg.identity)
        with transaction(conn):
            target = save_target(conn, target)
        console.print(f"[bold]Added:[/bold] user {target.name} (id {target.external_id})")
        console.print("[bold]Emails discovered:[/bold]")
        for e in target.emails:
            console.print(f"  • {e}")
    elif kind == "org":
        if not org:
            console.print("[red]--org is required when --kind=org[/red]")
            raise typer.Exit(2)
        with GitHubClient() as gh:
            target = discover_org(gh, org)
            with transaction(conn):
                target = save_target(conn, target)
            assert target.id is not None
            console.print(f"[bold]Added:[/bold] org {target.name} (id {target.external_id})")
            console.print(
                f"Discovering repos in {target.name} "
                f"(include={cfg.ingest.include_repos or '*'}, "
                f"exclude={cfg.ingest.exclude_repos or 'none'})…"
            )
            n_kept = 0
            with transaction(conn):
                for r in enumerate_org_repos(
                    gh,
                    target.name,
                    include=cfg.ingest.include_repos,
                    exclude=cfg.ingest.exclude_repos,
                ):
                    q.upsert_repo(conn, target_id=target.id, **r)
                    n_kept += 1
            console.print(f"Repos saved: [bold]{n_kept}[/bold] (filters applied).")
    elif kind == "repo":
        with GitHubClient() as gh:
            try:
                target, metadata = discover_repo(gh, repo=repo)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(2) from None
        with transaction(conn):
            target = save_target(conn, target)
            assert target.id is not None
            q.upsert_repo(conn, target_id=target.id, **metadata)
        console.print(f"[bold]Added:[/bold] repo {target.name} (id {target.external_id})")
        console.print(f"Default branch: {metadata['default_branch']}")
    else:
        console.print(f"[red]Unknown --kind: {kind!r}. Expected 'user', 'org', or 'repo'.[/red]")
        raise typer.Exit(2)
    console.print(f"\nDB: {cfg.paths.db_path}")


@app.command("init-claude-md")
def init_claude_md(
    output: Path = typer.Option(
        Path("CLAUDE.md"),
        "--output",
        "-o",
        help="Where to write the file. Default: ./CLAUDE.md",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite an existing file at --output.",
    ),
    server_name: str = typer.Option(
        "github-twin",
        "--server-name",
        help="MCP server name as registered in ~/.claude.json.",
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Write a `CLAUDE.md` template wired to the github-twin MCP tools.

    Reads every target currently in the DB so the template can list them
    and document how to scope tool calls.
    """
    from datetime import date as _date

    from github_twin.templates.claude_md import render

    cfg, conn = _ctx(config)
    targets = load_targets(conn)
    if output.exists() and not overwrite:
        console.print(
            f"[red]{output} already exists.[/red] "
            "Re-run with [bold]--overwrite[/bold] to replace it."
        )
        raise typer.Exit(1)

    content = render(
        targets=targets,
        server_name=server_name,
        date=_date.today().isoformat(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    console.print(f"[bold]Wrote[/bold] {output} ({len(content)} chars).")
    if not targets:
        console.print(
            "[yellow]No targets found — the file uses placeholders. "
            "Run `gt init` then re-run with --overwrite.[/yellow]"
        )


# ---------- gt targets ----------


@targets_app.command("list")
def targets_list(config: Path | None = typer.Option(None, "--config")) -> None:
    """List every target in this DB."""
    _cfg, conn = _ctx(config)
    targets = load_targets(conn)
    if not targets:
        console.print("[yellow]No targets. Run `gt init` first.[/yellow]")
        return
    t = Table(title=f"Targets ({len(targets)})")
    t.add_column("id", justify="right")
    t.add_column("kind")
    t.add_column("name")
    t.add_column("external_id", justify="right")
    t.add_column("emails")
    for target in targets:
        t.add_row(
            str(target.id),
            target.kind,
            target.name,
            str(target.external_id),
            str(len(target.emails)) if target.is_user else "—",
        )
    console.print(t)


@targets_app.command("remove")
def targets_remove(
    name: str = typer.Argument(..., help="Target name to remove."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Delete a target and every artifact / chunk / vector it owns."""
    _cfg, conn = _ctx(config)
    try:
        target = load_target(conn, name=name)
    except AmbiguousTargetError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    if target is None or target.id is None:
        console.print(f"[red]No target named {name!r}.[/red]")
        raise typer.Exit(1)
    if not yes:
        confirm = typer.confirm(f"Delete {target.kind} target {target.name} and all its data?")
        if not confirm:
            console.print("[dim]aborted[/dim]")
            raise typer.Exit(0)
    with transaction(conn):
        q.delete_target(conn, target.id)
    console.print(f"[green]✓ removed {target.kind} {target.name}[/green]")


# ---------- gt auth ----------


@auth_app.command("login")
def auth_login(
    scopes: str | None = typer.Option(
        None,
        "--scopes",
        help=(
            "Space-separated OAuth scopes. Default: cfg.auth.default_scopes "
            "(repo read:org user:email)."
        ),
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Don't try to open the verification URL automatically.",
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Acquire a GitHub access token via OAuth device flow and persist it."""
    import webbrowser

    from github_twin.ingest import auth_storage, oauth

    cfg = load_config(config)
    scope = scopes or cfg.auth.default_scopes
    client_id = cfg.auth.client_id

    code = oauth.request_device_code(client_id, scope)

    console.print()
    console.print(f"[bold]Visit:[/bold] {code.verification_uri}")
    console.print(f"[bold]Code:[/bold] [cyan]{code.user_code}[/cyan]")
    console.print(
        f"[dim](or open the pre-filled URL: "
        f"{code.verification_uri_complete or code.verification_uri})[/dim]"
    )
    console.print()

    if not no_browser and code.verification_uri_complete:
        try:
            webbrowser.open(code.verification_uri_complete)
        except Exception as exc:  # noqa: BLE001 — browser open is best-effort
            log.debug("webbrowser.open failed: %s", exc)

    console.print("Waiting for authorization…")
    try:
        token = oauth.poll_for_token(
            client_id,
            code.device_code,
            interval=code.interval,
            expires_in=code.expires_in,
        )
    except oauth.OAuthError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    login: str | None = None
    try:
        with GitHubClient(token=token) as gh:
            resp = gh.request("GET", "/user")
            data = resp.json()
            if isinstance(data, dict) and isinstance(data.get("login"), str):
                login = data["login"]
    except Exception as exc:  # noqa: BLE001 — identify is best-effort
        log.debug("GET /user after device flow failed: %s", exc)

    kind = auth_storage.store_token(token, login=login, scopes=scope)
    where = (
        "system keyring"
        if kind == "keyring"
        else f"{cfg.paths.data_dir / 'auth' / 'token.json'} (0600)"
    )
    who = f" for [bold]{login}[/bold]" if login else ""
    console.print(f"[green]✓ stored{who} in {where}[/green]")


@auth_app.command("status")
def auth_status(config: Path | None = typer.Option(None, "--config")) -> None:
    """Show which auth source `gt` will use and what's available."""
    import shutil
    import subprocess

    from github_twin.ingest import auth_storage

    _ = load_config(config)

    persisted = auth_storage.describe_source()
    gh_available = shutil.which("gh") is not None
    gh_token_ok = False
    if gh_available:
        try:
            r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
            gh_token_ok = r.returncode == 0 and bool(r.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            gh_token_ok = False
    env_set = bool(os.environ.get("GITHUB_TOKEN"))

    if persisted is not None:
        active = f"persisted ({persisted.kind})"
    elif gh_token_ok:
        active = "gh auth token"
    elif env_set:
        active = "GITHUB_TOKEN env var"
    else:
        active = "[red]none — run `gt auth login`[/red]"

    t = Table(title="GitHub auth sources")
    t.add_column("source")
    t.add_column("present")
    t.add_column("details")
    t.add_row(
        "persisted (gt auth login)",
        "✓" if persisted else "—",
        (
            f"{persisted.kind} · {persisted.location}"
            + (f" · login={persisted.login}" if persisted and persisted.login else "")
            + (f" · scopes={persisted.scopes}" if persisted and persisted.scopes else "")
        )
        if persisted
        else "",
    )
    t.add_row(
        "gh auth token",
        "✓" if gh_token_ok else "—",
        "gh CLI not installed" if not gh_available else ("ok" if gh_token_ok else "not authed"),
    )
    t.add_row(
        "GITHUB_TOKEN env var",
        "✓" if env_set else "—",
        "set" if env_set else "",
    )
    console.print(t)
    console.print(f"[bold]Active:[/bold] {active}")


@auth_app.command("logout")
def auth_logout(config: Path | None = typer.Option(None, "--config")) -> None:
    """Remove the persisted device-flow token (keyring + file)."""
    from github_twin.ingest import auth_storage

    _ = load_config(config)
    before = auth_storage.describe_source()
    auth_storage.delete_token()
    if before is None:
        console.print("[dim]No persisted token to remove.[/dim]")
    else:
        console.print(f"[green]✓ removed persisted token ({before.kind}).[/green]")
    console.print("[dim]gh auth and GITHUB_TOKEN (if set) are untouched.[/dim]")


# ---------- end gt auth ----------


@app.command()
def ingest(
    since: str | None = typer.Option(None, help="ISO date floor (overrides cursor)"),
    commits_only: bool = typer.Option(False, "--commits-only"),
    reviews_only: bool = typer.Option(False, "--reviews-only"),
    limit: int | None = typer.Option(None, help="Cap items per source (debug)"),
    target: str | None = typer.Option(
        None, "--target", help="Restrict to one target (by name). Default: all."
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Fetch new commits + review comments. Iterates all targets unless --target."""
    cfg, conn = _ctx(config)
    target_filter = _resolve_target_arg(conn, target)
    _run_ingest_safely(
        cfg,
        conn,
        since=since,
        commits_only=commits_only,
        reviews_only=reviews_only,
        limit=limit,
        target=target_filter.id if target_filter else None,
    )


@app.command()
def embed(
    rebuild: bool = typer.Option(False, "--rebuild", help="Drop all vectors first"),
    batch_size: int | None = typer.Option(
        None,
        help="Embedder batch size (defaults to cfg.embed.batch_size).",
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Embed any chunks that don't yet have a vector."""
    cfg, conn = _ctx(config)
    run_embed(
        cfg,
        conn,
        rebuild=rebuild,
        batch_size=batch_size or cfg.embed.batch_size,
        report=_report,
    )


@app.command()
def summarize(
    kind: list[str] = typer.Option(
        None,
        "--kind",
        help="Repeatable. Restrict to specific chunk kinds. Default: cfg.summarize.kinds.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Cap the number of chunks summarized (useful for trying a model on a sample).",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="claude | gemini | ollama | auto (default: cfg.summarize.backend).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override the Ollama model (cfg.summarize.ollama_model). Ignored for cloud backends.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Clear existing summaries for the given kinds before regenerating.",
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Generate LLM summaries for code chunks (used by embed-time prefix)."""
    cfg, conn = _ctx(config)
    if backend is not None:
        cfg = cfg.model_copy(
            update={
                "summarize": cfg.summarize.model_copy(update={"backend": backend}),
            }
        )
    if model is not None:
        cfg = cfg.model_copy(
            update={
                "summarize": cfg.summarize.model_copy(update={"ollama_model": model}),
            }
        )
    kinds = tuple(kind) if kind else None
    run_summarize(cfg, conn, kinds=kinds, limit=limit, rebuild=rebuild, report=_report)


@app.command()
def sync(
    since: str | None = typer.Option(None, help="ISO date floor"),
    skip_summarize: bool = typer.Option(
        False,
        "--skip-summarize",
        help="Don't auto-run summarize before embed.",
    ),
    target: str | None = typer.Option(
        None, "--target", help="Restrict to one target (by name). Default: all."
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Incremental: ingest deltas, summarize new code chunks, then embed.

    Without `--target`, iterates every target in the DB and runs ingest
    for each. Summarize + embed are corpus-wide and run once at the end.
    """
    cfg, conn = _ctx(config)
    target_filter = _resolve_target_arg(conn, target)
    _run_ingest_safely(
        cfg,
        conn,
        since=since,
        commits_only=False,
        reviews_only=False,
        limit=None,
        target=target_filter.id if target_filter else None,
    )
    if not skip_summarize:
        run_summarize(cfg, conn, report=_report)
    run_embed(cfg, conn, rebuild=False, batch_size=cfg.embed.batch_size, report=_report)


@app.command()
def stats(
    target: str | None = typer.Option(
        None, "--target", help="Restrict to one target. Default: per-target breakdown."
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Show artifact / chunk / vector counts. Per-target breakdown by default."""
    cfg, conn = _ctx(config)
    targets = load_targets(conn)

    if not targets:
        console.print("[yellow]No targets. Run `gt init` first.[/yellow]")
        return

    if target is not None:
        t = _resolve_target_arg(conn, target)
        assert t is not None and t.id is not None
        targets = [t]

    console.print(f"[bold]DB:[/bold] {cfg.paths.db_path}")
    for tg in targets:
        assert tg.id is not None
        s = q.stats(conn, target_id=tg.id)
        console.print(
            f"\n[bold]Target:[/bold] {tg.kind} {tg.name} "
            f"(id {tg.id}" + (f", {len(tg.emails)} emails" if tg.is_user else "") + ")"
        )
        t1 = Table(title="Artifacts by kind")
        t1.add_column("kind")
        t1.add_column("count", justify="right")
        for k, n in sorted(s["artifacts"].items()):
            t1.add_row(k, str(n))
        console.print(t1)

        t2 = Table(title="Chunks by kind")
        t2.add_column("kind")
        t2.add_column("count", justify="right")
        for k, n in sorted(s["chunks"].items()):
            t2.add_row(k, str(n))
        console.print(t2)

        t3 = Table(title="Languages by chunk (top 15)")
        t3.add_column("language")
        t3.add_column("chunks", justify="right")
        for lang, n in list(s["languages"].items())[:15]:
            t3.add_row(lang or "<none>", str(n))
        console.print(t3)

        console.print(
            f"[bold]vectors:[/bold] {s['vectors']}    "
            f"[bold]pending embed:[/bold] {s['pending_embed']}"
        )


@app.command()
def repos(
    target: str | None = typer.Option(
        None, "--target", help="Restrict to one target's repos. Default: all."
    ),
    include_archived: bool = typer.Option(False, "--include-archived"),
    include_forks: bool = typer.Option(False, "--include-forks"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """List repos in the DB. Without --target, lists across all targets."""
    _cfg, conn = _ctx(config)
    target_filter = _resolve_target_arg(conn, target)
    rows = q.list_repos(
        conn,
        target_id=target_filter.id if target_filter else None,
        include_archived=include_archived,
        include_forks=include_forks,
    )
    if not rows:
        console.print(
            "[yellow]No repos. Run `gt init --kind org --org <name>` or "
            "`gt init --kind repo --repo owner/name` first.[/yellow]"
        )
        return
    # Build a target_id → name lookup so each row gets a human-readable owner.
    targets_by_id = {t.id: t.name for t in load_targets(conn)}
    title = (
        f"Repos for {target_filter.name} ({len(rows)} shown)"
        if target_filter
        else f"Repos across all targets ({len(rows)} shown)"
    )
    t = Table(title=title)
    t.add_column("target")
    t.add_column("full_name")
    t.add_column("default_branch")
    t.add_column("pushed_at")
    t.add_column("size_kb", justify="right")
    t.add_column("archived")
    t.add_column("fork")
    for r in rows:
        t.add_row(
            targets_by_id.get(r["target_id"], f"#{r['target_id']}"),
            r["full_name"],
            r["default_branch"] or "",
            r["pushed_at"] or "",
            str(r["size_kb"] or ""),
            "✓" if r["archived"] else "",
            "✓" if r["fork"] else "",
        )
    console.print(t)


@app.command()
def distill(
    backend: str = typer.Option(
        None,
        "--backend",
        help="'claude' | 'gemini' | 'ollama' | unset for auto.",
    ),
    target: str | None = typer.Option(
        None, "--target", help="Target name. Required when the DB has >1 target."
    ),
    author: str | None = typer.Option(
        None,
        "--author",
        help="GitHub login to scope clustering to a single reviewer.",
    ),
    repo: str | None = typer.Option(
        None,
        "--repo",
        help="'owner/name' to scope clustering to one repo.",
    ),
    kind: str = typer.Option(
        "review",
        "--kind",
        help="'review' or 'code'.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Per-chunk language filter. Only honored for --kind code.",
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Cluster commits or review comments and synthesize reusable rules.

    Rule artifacts are stamped with the chosen target's id, so retrieval
    can scope rules to a specific target (or coalesce across all of them).
    """
    cfg, conn = _ctx(config)
    if target is not None:
        chosen = _resolve_target_arg(conn, target)
        assert chosen is not None and chosen.id is not None
    else:
        try:
            chosen = load_target(conn)
        except AmbiguousTargetError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from None
        if chosen is None or chosen.id is None:
            console.print("[red]No targets in DB. Run `gt init` first.[/red]")
            raise typer.Exit(1)
    kind = kind.lower()
    if kind not in ("review", "code"):
        raise typer.BadParameter(f"--kind must be 'review' or 'code', got {kind!r}")
    scoped_author = author
    if scoped_author is None and chosen.is_user:
        scoped_author = None  # user-mode artifacts don't carry author_login today
    system_prompt = CODE_SYSTEM_PROMPT if kind == "code" else SYSTEM_PROMPT
    synth = make_synthesizer(
        claude_model=cfg.distill.claude_model,
        gemini_model=cfg.distill.gemini_model,
        ollama_model=cfg.distill.ollama_model,
        prefer=backend or cfg.distill.backend,
        system_prompt=system_prompt,
    )
    embedder = make_embedder(cfg.embed)
    chunk_kind: Literal["code", "review_comment"]
    rule_chunk_kind: Literal["rule", "code_rule"]
    if kind == "code":
        chunk_kind, rule_chunk_kind = "code", "code_rule"
    else:
        chunk_kind, rule_chunk_kind = "review_comment", "rule"
    stats = distill_rules(
        conn=conn,
        synth=synth,
        embedder=embedder,
        cfg=cfg.distill,
        target_id=chosen.id,
        author_login=scoped_author,
        chunk_kind=chunk_kind,
        rule_chunk_kind=rule_chunk_kind,
        language=language,
        repo=repo,
        report=_report,
    )
    scope_bits = [f"target={chosen.name}"]
    if scoped_author:
        scope_bits.append(f"author={scoped_author}")
    if repo:
        scope_bits.append(f"repo={repo}")
    if language and kind == "code":
        scope_bits.append(f"language={language}")
    scope_msg = f" ({', '.join(scope_bits)})"
    console.print(
        f"[green]distill --kind {kind}{scope_msg}[/green]: clusters={stats.clusters} "
        f"rules={stats.rules_written} incoherent={stats.incoherent} "
        f"failed={stats.failed}"
    )


@clones_app.command("prune")
def clones_prune(
    older_than_days: int | None = typer.Option(
        None,
        "--older-than-days",
        help="Also drop clones whose dir mtime is older than this many days.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print decisions without deleting anything.",
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Remove cached clones for repos no longer referenced by any target.

    Keep-set is the union of `repo.full_name` across every target — a
    clone stays as long as at least one target still references it.
    """
    cfg, conn = _ctx(config)
    keep = q.all_cached_repos(conn)
    decisions = prune_cache(
        cfg.ingest.clones_dir,
        keep=keep,
        older_than_days=older_than_days,
        dry_run=dry_run,
    )
    if not decisions:
        console.print("[green]Nothing to prune.[/green]")
        return
    verb = "Would remove" if dry_run else "Removed"
    for d in decisions:
        console.print(f"  {verb} {d.full_name} ({d.reason})  {d.path}")
    console.print(f"\n[bold]{verb} {len(decisions)} clone(s).[/bold]")


def _preflight_eligibility(
    conn: sqlite3.Connection,
    *,
    since: str,
    author: str | None,
    repo: str | None,
    surface: str,
) -> bool:
    """Print holdout counts before launching paid LLM calls."""
    counts = count_eligible(conn, since=since, author_login=author, repo=repo)
    scope_bits = [f"since={since}"]
    if author:
        scope_bits.append(f"author={author}")
    if repo:
        scope_bits.append(f"repo={repo}")
    console.print(
        f"eligible ({', '.join(scope_bits)}): "
        f"review_comments={counts['review_comments']}  "
        f"decisioned_prs={counts['decisioned_prs']}"
    )
    relevant = counts["review_comments"] if surface == "reviews" else counts["decisioned_prs"]
    if relevant == 0:
        msg = "[red]No eligible items for this scope. "
        if author and surface == "reviews":
            msg += "Check the spelling of --author, or omit it for user-mode DBs (author_login is NULL there).[/red]"
        elif author and surface == "predictions":
            msg += (
                "Check the spelling of --author; meta.reviewer_decisions must contain "
                "an entry with that login.[/red]"
            )
        elif not author and surface == "predictions":
            msg += (
                "Org-mode DBs need --author here (artifact.decision is NULL; truth is "
                "in meta.reviewer_decisions).[/red]"
            )
        else:
            msg += "Pick an earlier --since.[/red]"
        console.print(msg)
        return False
    return True


def _make_judge_embedder(cfg: Config, judge_backend: str | None) -> Embedder:
    """Pick a *different* embedder for scoring than the one used for retrieval."""
    if judge_backend == "same":
        console.print(
            "[yellow]warning: --judge-backend=same uses the retrieval embedder; "
            "eval scores will be biased toward RAG. Recommended: install [st] and "
            "use a different model.[/yellow]"
        )
        return make_embedder(cfg.embed)
    chosen = judge_backend or "sentence_transformers"
    try:
        judge_cfg = EmbedCfg(
            backend=chosen,
            model="BAAI/bge-small-en-v1.5",
            dim=384,
            device=cfg.embed.device,
            batch_size=cfg.embed.batch_size,
        )
        return make_embedder(judge_cfg)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]judge embedder unavailable ({exc}); falling back to "
            f"retrieval embedder. Install [bold]uv sync --extra st[/bold] for a "
            f"clean comparison.[/yellow]"
        )
        return make_embedder(cfg.embed)


@eval_app.command("reviews")
def eval_reviews(
    since: str = typer.Option(
        ..., "--since", help="ISO date; artifacts at/after this are held out."
    ),
    author: str | None = typer.Option(
        None,
        "--author",
        help="GitHub login.",
    ),
    repo: str | None = typer.Option(
        None,
        "--repo",
        help="'owner/name'.",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Cap held-out items."),
    k: int = 5,
    llm_backend: str | None = typer.Option(None, "--llm-backend"),
    judge_backend: str | None = typer.Option(None, "--judge-backend"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Score retrieved-context review comments against held-out ground truth."""
    cfg, conn = _ctx(config)
    if not _preflight_eligibility(
        conn,
        since=since,
        author=author,
        repo=repo,
        surface="reviews",
    ):
        raise typer.Exit(1)
    retriever_emb = make_embedder(cfg.embed)
    judge_emb = _make_judge_embedder(cfg, judge_backend)
    store = make_vector_store(conn, backend=cfg.vector_store.backend, dim=cfg.embed.dim)
    llm = make_text_llm(
        claude_model=cfg.distill.claude_model,
        gemini_model=cfg.distill.gemini_model,
        ollama_model=cfg.distill.ollama_model,
        prefer=llm_backend or "auto",
    )
    scope_msg = ""
    if author:
        scope_msg += f"  author={author}"
    if repo:
        scope_msg += f"  repo={repo}"
    console.print(
        f"eval reviews: judge={getattr(judge_emb, 'model_id', '?')}  "
        f"llm={llm.backend_id}{scope_msg}"
    )
    result = evaluate_reviews(
        conn,
        retriever_embedder=retriever_emb,
        judge_embedder=judge_emb,
        store=store,
        llm=llm,
        since=since,
        author_login=author,
        repo=repo,
        limit=limit,
        k=k,
        progress=_report,
    )
    render_review_result(result, console)


@eval_app.command("predictions")
def eval_predictions(
    since: str = typer.Option(..., "--since"),
    author: str | None = typer.Option(None, "--author"),
    repo: str | None = typer.Option(None, "--repo"),
    limit: int | None = typer.Option(None, "--limit"),
    k: int = 20,
    llm_backend: str | None = typer.Option(None, "--llm-backend"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Compare LLM-from-cold prediction against `predict_review_outcome`."""
    cfg, conn = _ctx(config)
    if not _preflight_eligibility(
        conn,
        since=since,
        author=author,
        repo=repo,
        surface="predictions",
    ):
        raise typer.Exit(1)
    retriever_emb = make_embedder(cfg.embed)
    store = make_vector_store(conn, backend=cfg.vector_store.backend, dim=cfg.embed.dim)
    llm = make_text_llm(
        claude_model=cfg.distill.claude_model,
        gemini_model=cfg.distill.gemini_model,
        ollama_model=cfg.distill.ollama_model,
        prefer=llm_backend or "auto",
    )
    scope_msg = ""
    if author:
        scope_msg += f"  author={author}"
    if repo:
        scope_msg += f"  repo={repo}"
    console.print(f"eval predictions: llm={llm.backend_id}{scope_msg}")
    result = evaluate_predictions(
        conn,
        retriever_embedder=retriever_emb,
        store=store,
        llm=llm,
        since=since,
        author_login=author,
        repo=repo,
        limit=limit,
        k=k,
        progress=_report,
    )
    render_predict_result(result, console)


@eval_app.command("search")
def eval_search(
    yaml_file: Path = typer.Argument(..., help="YAML query suite (see evals/queries/)."),
    k: int = typer.Option(5, "--k", help="Top-K considered when matching expectations."),
    mode: str = typer.Option(
        "all",
        "--mode",
        help="bm25 | vector | hybrid | all (default).",
    ),
    expansion: str | None = typer.Option(
        None,
        "--expansion",
        help="Override cfg.retrieval.query_expansion: off | rule | ollama.",
    ),
    recency_half_life_days: float | None = typer.Option(
        None,
        "--recency-half-life-days",
        help=(
            "Override cfg.retrieval.recency_half_life_days for this run. "
            "Hybrid-mode only; bm25/vector legs stay unweighted. "
            "Pass 0 to force-disable when the cfg has a non-None default."
        ),
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Retrieval-quality eval."""
    cfg, conn = _ctx(config)
    if mode == "all":
        modes = ALL_MODES
    else:
        wanted = tuple(m.strip() for m in mode.split(",") if m.strip())
        unknown = [m for m in wanted if m not in ALL_MODES]
        if unknown:
            console.print(
                f"[red]unknown mode(s): {unknown}; expected one of {list(ALL_MODES)}[/red]"
            )
            raise typer.Exit(2)
        modes = wanted  # type: ignore[assignment]
    queries = load_queries(yaml_file)
    if not queries:
        console.print(f"[yellow]{yaml_file}: no queries found.[/yellow]")
        raise typer.Exit(0)
    embedder = make_embedder(cfg.embed)
    store = make_vector_store(conn, backend=cfg.vector_store.backend, dim=cfg.embed.dim)
    if expansion is not None:
        cache_path = (
            (
                cfg.retrieval.expansion_cache_path
                or (cfg.paths.data_dir / "query_expansion_cache.sqlite")
            )
            if expansion == "ollama"
            else None
        )
        expander = make_expander(
            expansion,
            ollama_model=cfg.retrieval.ollama_model,
            ollama_host=cfg.retrieval.ollama_host,
            cache_path=cache_path,
        )
    else:
        expander = expander_from_config(cfg)
    effective_recency = (
        recency_half_life_days
        if recency_half_life_days is not None
        else cfg.retrieval.recency_half_life_days
    )
    console.print(
        f"eval search: {len(queries)} queries  k={k}  modes={list(modes)}  "
        f"embedder={getattr(embedder, 'model_id', '?')}  "
        f"expander={getattr(expander, 'backend_id', 'off')}  "
        f"recency_half_life_days={effective_recency}"
    )
    report = evaluate_search(
        conn,
        embedder,
        store,
        queries,
        k=k,
        modes=modes,
        expander=expander,
        recency_half_life_days=effective_recency,
    )
    exit_code = render_search_result(report, console)
    if exit_code:
        raise typer.Exit(exit_code)


@app.command()
def serve(config: Path | None = typer.Option(None, "--config")) -> None:
    """Run the MCP server over stdio."""
    from github_twin.mcp_server.server import run

    run(config_path=config)


if __name__ == "__main__":  # pragma: no cover
    app()
    sys.exit(0)
