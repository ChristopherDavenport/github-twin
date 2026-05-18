"""`gt init` embedder flags — stamp the embed backend / model / dim
into `config.toml` at init time so the choice survives every subsequent
command.

We test the two helpers (`_resolve_embed_defaults`, `_persist_embed_config`)
directly for the matrix coverage, then run one end-to-end `gt init`
through Typer's CliRunner to confirm the flags reach the config file
and the DB is opened with the right dim.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from github_twin.cli import (
    _persist_embed_config,
    _resolve_embed_defaults,
    app,
)

# ---------- _resolve_embed_defaults ----------


def test_resolve_defaults_ollama_uses_nomic_768():
    assert _resolve_embed_defaults(None, None, None) == ("ollama", "nomic-embed-text", 768)


def test_resolve_defaults_gemini_uses_gemini_3072():
    assert _resolve_embed_defaults("gemini", None, None) == (
        "gemini",
        "gemini-embedding-001",
        3072,
    )


def test_resolve_defaults_respect_explicit_model_and_dim():
    assert _resolve_embed_defaults("gemini", "gemini-embedding-001", 1536) == (
        "gemini",
        "gemini-embedding-001",
        1536,
    )


def test_resolve_defaults_st_requires_model_and_dim():
    with pytest.raises(typer.BadParameter, match="sentence_transformers"):
        _resolve_embed_defaults("sentence_transformers", None, None)


def test_resolve_defaults_st_accepts_model_and_dim():
    assert _resolve_embed_defaults("sentence_transformers", "BAAI/bge-small-en-v1.5", 384) == (
        "sentence_transformers",
        "BAAI/bge-small-en-v1.5",
        384,
    )


def test_resolve_defaults_rejects_unknown_backend():
    with pytest.raises(typer.BadParameter, match="Unknown --embed-backend"):
        _resolve_embed_defaults("nonsense", None, None)


# ---------- _persist_embed_config ----------


def test_persist_writes_fresh_file(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _persist_embed_config(cfg_path, "gemini", "gemini-embedding-001", 3072)
    data = tomllib.loads(cfg_path.read_text())
    assert data == {"embed": {"backend": "gemini", "model": "gemini-embedding-001", "dim": 3072}}


def test_persist_appends_when_other_sections_exist(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[paths]\ndata_dir = "/tmp/x"\n')
    _persist_embed_config(cfg_path, "gemini", "gemini-embedding-001", 3072)
    data = tomllib.loads(cfg_path.read_text())
    assert data["paths"] == {"data_dir": "/tmp/x"}
    assert data["embed"] == {
        "backend": "gemini",
        "model": "gemini-embedding-001",
        "dim": 3072,
    }


def test_persist_is_idempotent_when_values_match(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    initial = '[embed]\nbackend = "gemini"\nmodel = "gemini-embedding-001"\ndim = 3072\n'
    cfg_path.write_text(initial)
    _persist_embed_config(cfg_path, "gemini", "gemini-embedding-001", 3072)
    # Re-read: byte-identical → didn't rewrite.
    assert cfg_path.read_text() == initial


def test_persist_refuses_mismatched_existing_embed(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[embed]\nbackend = "ollama"\nmodel = "nomic-embed-text"\ndim = 768\n')
    with pytest.raises(typer.BadParameter, match="Existing"):
        _persist_embed_config(cfg_path, "gemini", "gemini-embedding-001", 3072)
    # Original untouched.
    data = tomllib.loads(cfg_path.read_text())
    assert data["embed"]["backend"] == "ollama"


# ---------- end-to-end `gt init` ----------


def _fake_user_target(name: str = "alice", ext_id: int = 42):
    from github_twin.target import Target

    return Target(kind="user", name=name, external_id=ext_id, emails=[])


def test_gt_init_writes_embed_config_and_opens_db_with_chosen_dim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """End-to-end: `gt init --embed-backend gemini` writes config.toml
    and creates the DB's vec_chunk with dim=3072."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(tmp_path / "data"))
    # Belt-and-braces: scrub embed env vars so they can't fight the flag.
    for k in ("GT_EMBED__BACKEND", "GT_EMBED__MODEL", "GT_EMBED__DIM"):
        monkeypatch.delenv(k, raising=False)

    # Stub out the GitHub roundtrip — init's user-mode path constructs a
    # GitHubClient and calls discover_user. We patch both.
    class _FakeGH:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("github_twin.cli.GitHubClient", _FakeGH)
    monkeypatch.setattr("github_twin.cli.discover_user", lambda gh, identity: _fake_user_target())

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["init", "--kind", "user", "--embed-backend", "gemini"],
    )
    assert result.exit_code == 0, result.output

    # config.toml lives next to the DB, inside the resolved data_dir.
    cfg_path = tmp_path / "data" / "config.toml"
    assert cfg_path.exists()
    # A stray ./config.toml MUST NOT have been written in cwd.
    assert not (tmp_path / "config.toml").exists()
    data = tomllib.loads(cfg_path.read_text())
    assert data["embed"] == {
        "backend": "gemini",
        "model": "gemini-embedding-001",
        "dim": 3072,
    }

    # And the DB was created with the 3072-dim vec table — open_db
    # would have raised on dim mismatch otherwise.
    db_path = tmp_path / "data" / "db.sqlite"
    assert db_path.exists()
    import sqlite3

    import sqlite_vec

    conn = sqlite3.connect(db_path)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunk'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None and "FLOAT[3072]" in row[0]


def test_gt_init_user_then_org_layer_into_one_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two `gt init` calls in the same cwd (user, then org) should layer
    BOTH targets into one DB sitting at the resolved data_dir. This is
    the bug the consistency fix was designed for — pre-fix, the two
    inits ended up at different paths (config in cwd, data in XDG)."""
    from github_twin.target import Target

    cwd = tmp_path / "project"
    cwd.mkdir()
    data_dir = tmp_path / "twin-data"
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("GT_PATHS__DATA_DIR", str(data_dir))

    class _FakeGH:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("github_twin.cli.GitHubClient", _FakeGH)
    monkeypatch.setattr(
        "github_twin.cli.discover_user",
        lambda gh, identity: Target(kind="user", name="alice", external_id=42, emails=[]),
    )
    monkeypatch.setattr(
        "github_twin.cli.discover_org",
        lambda gh, org: Target(kind="org", name=org, external_id=99, emails=[]),
    )
    monkeypatch.setattr("github_twin.cli.enumerate_org_repos", lambda *a, **kw: iter([]))

    runner = CliRunner()
    r1 = runner.invoke(app, ["init", "--kind", "user"])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["init", "--kind", "org", "--org", "http4s"])
    assert r2.exit_code == 0, r2.output

    # One DB in the resolved data_dir, holding both target rows.
    db_path = data_dir / "db.sqlite"
    assert db_path.exists()
    # No DB or config leaked into cwd.
    assert not (cwd / "config.toml").exists()
    assert not (cwd / "data").exists()

    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT kind, name FROM target ORDER BY id").fetchall()
    finally:
        conn.close()
    assert rows == [("user", "alice"), ("org", "http4s")]
