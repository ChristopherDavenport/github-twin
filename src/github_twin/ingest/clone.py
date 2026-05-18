"""`git clone` helpers for github-twin ingest.

`cloned_repo(full_name)` is a context manager. Default behaviour is
**process-and-purge**: clone into a tempdir, yield the path + HEAD sha,
`shutil.rmtree` on exit (even on exception). Persistent caching is
opt-in via `cache_dir=<path>` — clones live under `<cache_dir>/<owner>/<name>`
and `git fetch` updates on reuse.

Clone depth: pass `depth=None` for a full-history clone (used by commits
ingest, which walks `git log` + `git show` locally instead of hitting the
API). Default is `depth=1` (shallow, HEAD only) which is what `files.py`
needs for its HEAD walk. If a persistent cache was previously shallow and
the caller now asks for full history, the existing checkout is unshallowed
once via `git fetch --unshallow` and subsequent fetches stay deep.

Auth: the token is injected into the clone URL as
`https://oauth2:<token>@github.com/<full_name>.git`. Immediately after
clone/fetch we rewrite `origin` to the public URL so the token never
lingers in `.git/config`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

from github_twin.ingest.github_client import _resolve_token

log = logging.getLogger(__name__)


class CloneError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClonedRepo:
    full_name: str  # 'owner/name'
    path: Path  # working tree root
    head_sha: str  # HEAD sha captured immediately after clone/fetch
    from_cache: bool  # True if reused an existing on-disk clone


def _git(args: list[str], *, cwd: Path | None = None) -> str:
    """Run git, return stdout (stripped). Raise CloneError on non-zero exit.

    `errors="replace"` because `git show` of a diff can include bytes that
    aren't valid UTF-8 (source files in legacy encodings, binary patch
    fragments inside hunks). The diff text is downstream-chunked and
    embedded; substituting U+FFFD for undecodable bytes is preferable to
    aborting an entire repo's commit walk.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except FileNotFoundError as e:  # pragma: no cover - git is a hard dep
        raise CloneError("`git` binary not found on PATH") from e
    if out.returncode != 0:
        raise CloneError(
            f"git {' '.join(args)} failed ({out.returncode}): {out.stderr.strip()[:400]}"
        )
    return out.stdout.strip()


def _auth_url(full_name: str, token: str) -> str:
    return f"https://oauth2:{token}@github.com/{full_name}.git"


def _public_url(full_name: str) -> str:
    return f"https://github.com/{full_name}.git"


def _scrub_remote(path: Path, full_name: str) -> None:
    """Rewrite origin to the tokenless URL so the token never lives on disk."""
    _git(["remote", "set-url", "origin", _public_url(full_name)], cwd=path)


def _fresh_clone(path: Path, full_name: str, token: str, *, depth: int | None = 1) -> str:
    args = ["clone", "--no-tags"]
    if depth is not None:
        args += ["--depth", str(depth), "--single-branch"]
    args += [_auth_url(full_name, token), str(path)]
    _git(args)
    _scrub_remote(path, full_name)
    return _git(["rev-parse", "HEAD"], cwd=path)


def _is_shallow(path: Path) -> bool:
    return _git(["rev-parse", "--is-shallow-repository"], cwd=path) == "true"


def _fetch_update(path: Path, full_name: str, token: str, *, depth: int | None = 1) -> str:
    """Re-point origin to the auth URL just for the fetch, then scrub again.

    `depth=1` keeps shallow clones shallow. `depth=None` requests a full
    history fetch — if the existing clone is shallow we run `--unshallow`
    first to backfill the history, then subsequent fetches are plain.
    """
    _git(["remote", "set-url", "origin", _auth_url(full_name, token)], cwd=path)
    try:
        if depth is None:
            if _is_shallow(path):
                _git(["fetch", "--unshallow", "--no-tags", "origin"], cwd=path)
            else:
                _git(["fetch", "--no-tags", "origin"], cwd=path)
        else:
            _git(["fetch", "--depth", str(depth), "--no-tags", "origin"], cwd=path)
        # default branch may differ from where the local checkout points; we
        # reset hard to whatever origin/HEAD is now.
        head_ref = _git(["symbolic-ref", "refs/remotes/origin/HEAD"], cwd=path)
        # 'refs/remotes/origin/main' → 'origin/main'
        short = head_ref.replace("refs/remotes/", "", 1)
        _git(["reset", "--hard", short], cwd=path)
    finally:
        _scrub_remote(path, full_name)
    return _git(["rev-parse", "HEAD"], cwd=path)


@dataclass(frozen=True)
class PruneDecision:
    full_name: str  # 'owner/name'
    path: Path
    reason: str  # 'not-in-keep' | 'stale'


def _iter_cache(cache_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield (full_name, path) for each owner/name directory under cache_dir."""
    if not cache_dir.is_dir():
        return
    for owner_dir in cache_dir.iterdir():
        if not owner_dir.is_dir():
            continue
        for repo_dir in owner_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            yield f"{owner_dir.name}/{repo_dir.name}", repo_dir


def _is_stale(path: Path, older_than_days: int, *, now: float | None = None) -> bool:
    cutoff = (now or time.time()) - older_than_days * 86400
    try:
        return path.stat().st_mtime < cutoff
    except OSError:
        return False


def plan_prune(
    cache_dir: Path,
    *,
    keep: set[str],
    older_than_days: int | None = None,
    now: float | None = None,
) -> list[PruneDecision]:
    """Decide which cached clones should be removed. Pure: no FS side effects.

    A clone is dropped if either:
      - its `owner/name` isn't in `keep`, or
      - `older_than_days` is set and the dir mtime is older than the cutoff.
    """
    decisions: list[PruneDecision] = []
    for full_name, path in _iter_cache(cache_dir):
        if full_name not in keep:
            decisions.append(PruneDecision(full_name, path, "not-in-keep"))
            continue
        if older_than_days is not None and _is_stale(path, older_than_days, now=now):
            decisions.append(PruneDecision(full_name, path, "stale"))
    return decisions


def prune_cache(
    cache_dir: Path,
    *,
    keep: set[str],
    older_than_days: int | None = None,
    dry_run: bool = False,
) -> list[PruneDecision]:
    """Plan and (unless dry-run) execute the prune. Returns the decisions
    so the caller can report them. `rmtree` errors are logged and skipped —
    we never want a stuck `.git/index.lock` to abort a whole GC pass."""
    decisions = plan_prune(cache_dir, keep=keep, older_than_days=older_than_days)
    if dry_run:
        return decisions
    for d in decisions:
        try:
            shutil.rmtree(d.path, ignore_errors=False)
        except OSError as e:
            log.warning("prune: rmtree %s failed: %s", d.path, e)
    return decisions


@contextmanager
def cloned_repo(
    full_name: str,
    *,
    cache_dir: Path | None = None,
    token: str | None = None,
    depth: int | None = 1,
) -> Iterator[ClonedRepo]:
    """Yield a working-tree path for `full_name`.

    cache_dir=None  → tempdir, deleted on exit.
    cache_dir=<p>   → persistent under `<p>/<owner>/<name>`, retained on exit.
    depth=1         → shallow clone (default; HEAD only).
    depth=None      → full-history clone (commits ingest path).
    """
    tok = token or _resolve_token()
    if cache_dir is None:
        tmp = Path(tempfile.mkdtemp(prefix="gt-clone-"))
        try:
            head = _fresh_clone(tmp, full_name, tok, depth=depth)
            yield ClonedRepo(full_name=full_name, path=tmp, head_sha=head, from_cache=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        owner, name = full_name.split("/", 1)
        target = cache_dir / owner / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if (target / ".git").is_dir():
            head = _fetch_update(target, full_name, tok, depth=depth)
            yield ClonedRepo(full_name=full_name, path=target, head_sha=head, from_cache=True)
        else:
            head = _fresh_clone(target, full_name, tok, depth=depth)
            yield ClonedRepo(full_name=full_name, path=target, head_sha=head, from_cache=False)


def commits_clone(
    full_name: str,
    *,
    cache_dir: Path | None,
    token: str | None = None,
) -> AbstractContextManager[ClonedRepo]:
    """Convenience wrapper for the commits ingest path: persistent + deep.

    `cache_dir=None` is accepted for tests that monkeypatch the clone
    provider; production callers always go through `run_ingest`, which
    resolves `clones_dir` against `paths.data_dir` before dispatch.
    """
    return cloned_repo(full_name, cache_dir=cache_dir, token=token, depth=None)
