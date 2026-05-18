"""`IngestCfg.clones_dir` resolves under `paths.data_dir` by default.

The legacy default (`Path("./data/clones")`) leaked the cwd: the first
org-mode `gt sync` would create `./data/clones/` in whatever directory
you happened to be in, which then tripped the (now-removed) `./data`
cwd auto-detect on the next run. `clones_dir = None` + `resolved_clones_dir`
keeps the path strictly under the resolved data_dir.
"""

from __future__ import annotations

from pathlib import Path

from github_twin.config import Config, IngestCfg, PathsCfg, resolved_clones_dir


def test_unset_clones_dir_resolves_under_data_dir(tmp_path: Path):
    cfg = Config(paths=PathsCfg(data_dir=tmp_path / "twin"))
    assert cfg.ingest.clones_dir is None
    assert resolved_clones_dir(cfg) == tmp_path / "twin" / "clones"


def test_explicit_clones_dir_is_honored(tmp_path: Path):
    cfg = Config(
        paths=PathsCfg(data_dir=tmp_path / "twin"),
        ingest=IngestCfg(clones_dir=tmp_path / "elsewhere" / "clones"),
    )
    assert resolved_clones_dir(cfg) == tmp_path / "elsewhere" / "clones"


def test_default_ingest_cfg_has_no_cwd_leak():
    """The default `IngestCfg.clones_dir` is None — no cwd-relative path."""
    assert IngestCfg().clones_dir is None
