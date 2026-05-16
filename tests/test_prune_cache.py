"""Tests for the clone-cache GC (`prune_cache` / `plan_prune`)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from github_twin.ingest.clone import plan_prune, prune_cache


def _seed_cache(root: Path, repos: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for full in repos:
        owner, name = full.split("/", 1)
        p = root / owner / name
        p.mkdir(parents=True)
        (p / ".git").mkdir()
        (p / "README.md").write_text("hi")
        out[full] = p
    return out


def test_plan_prune_keeps_known_and_drops_unknown(tmp_path: Path):
    paths = _seed_cache(tmp_path, ["org/a", "org/b", "org/c"])
    decisions = plan_prune(tmp_path, keep={"org/a"})
    drops = {d.full_name for d in decisions}
    assert drops == {"org/b", "org/c"}
    assert all(d.reason == "not-in-keep" for d in decisions)
    # Pure: nothing deleted on disk.
    assert all(p.exists() for p in paths.values())


def test_plan_prune_age_filter_keeps_fresh(tmp_path: Path):
    paths = _seed_cache(tmp_path, ["org/fresh", "org/stale"])
    # Set stale repo's mtime to 60 days ago.
    sixty_days = 60 * 86400
    old = time.time() - sixty_days
    os.utime(paths["org/stale"], (old, old))

    keep = {"org/fresh", "org/stale"}
    decisions = plan_prune(tmp_path, keep=keep, older_than_days=30)
    drops = [(d.full_name, d.reason) for d in decisions]
    assert drops == [("org/stale", "stale")]


def test_plan_prune_age_filter_and_unknown_together(tmp_path: Path):
    paths = _seed_cache(tmp_path, ["org/fresh", "org/orphan"])
    # orphan is not in keep -> dropped as not-in-keep, age never consulted.
    os.utime(paths["org/orphan"], (0, 0))
    decisions = plan_prune(tmp_path, keep={"org/fresh"}, older_than_days=30)
    assert {(d.full_name, d.reason) for d in decisions} == {
        ("org/orphan", "not-in-keep"),
    }


def test_prune_cache_deletes_unless_dry_run(tmp_path: Path):
    paths = _seed_cache(tmp_path, ["org/keep", "org/drop"])
    # Dry-run: returns the decision but leaves the dir on disk.
    dry = prune_cache(tmp_path, keep={"org/keep"}, dry_run=True)
    assert [d.full_name for d in dry] == ["org/drop"]
    assert paths["org/drop"].exists()

    # Real run: actually removes.
    real = prune_cache(tmp_path, keep={"org/keep"})
    assert [d.full_name for d in real] == ["org/drop"]
    assert not paths["org/drop"].exists()
    assert paths["org/keep"].exists()


def test_plan_prune_handles_missing_cache_dir(tmp_path: Path):
    """A pristine install where the cache dir doesn't exist yet — plan returns []."""
    assert plan_prune(tmp_path / "never-created", keep=set()) == []
