"""`gt wiki export` — materialize the SQLite corpus as a markdown vault.

These tests pin the on-disk shape (frontmatter + cross-links), the
idempotency contract (no-op second run), the prune-on-removal contract
(generated files for deleted rules vanish), and the hand-edit guard
(stripping `generated: true` preserves the file forever).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.config import Config
from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.wiki.export import export_wiki, resolve_vault_root
from github_twin.wiki.scan import list_generated_files, parse_frontmatter
from github_twin.wiki.slug import repo_slug, rule_slug
from tests.conftest import seed_target


class _FakeLLM:
    """Inline TextLLM that returns a canned profile. We don't need
    realistic shape — just stable text so the cache hash is meaningful."""

    backend_id = "fake"
    model_id = "fake-llm"

    def complete(self, *, system: str, user: str, max_tokens: int = 600) -> str:
        return f"This is a synthesized profile.\n\nSamples summarized: {len(user)} chars."


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "wiki.sqlite", embed_dim=4)
    seed_target(db, kind="user", name="alice")
    yield db
    db.close()


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.paths.data_dir = tmp_path / "data"
    return cfg


def _seed_rule(
    conn,
    *,
    text: str,
    language: str | None,
    chunk_kind: str = "rule",
    cluster_size: int = 5,
    examples: list[str] | None = None,
    urls: list[str] | None = None,
    external_id: str | None = None,
) -> int:
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="rule",
        external_id=external_id or f"r-{chunk_kind}-{text[:20]}",
        source_url=urls[0] if urls else None,
        repo=None,
        language=language,
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta={
            "cluster_size": cluster_size,
            "example_quotes": examples or ["example"],
            "member_urls": urls or [],
            "member_repos": [],
        },
    )
    return q.insert_chunk(
        conn,
        artifact_id=aid,
        kind=chunk_kind,
        text=text,
        context={"language": language},
        language=language,
    )


def _seed_repo(conn, *, full_name: str, default_branch: str = "main") -> None:
    q.upsert_repo(
        conn,
        target_id=1,
        full_name=full_name,
        default_branch=default_branch,
        pushed_at="2025-01-01T00:00:00Z",
    )


# ---------- shape ----------


def test_rule_page_has_frontmatter_and_body(conn, cfg):
    _seed_rule(
        conn,
        text="Prefer functional reactive style for stream handlers.",
        language="scala",
        cluster_size=12,
        examples=["see PR #42 for the canonical example"],
        urls=["https://github.com/acme/x/pull/42#discussion_r1"],
    )
    export_wiki(conn, cfg)
    root = resolve_vault_root(cfg)
    expected = (
        root
        / "rules"
        / "scala"
        / f"{rule_slug('Prefer functional reactive style for stream handlers.')}.md"
    )
    assert expected.exists(), list(root.rglob("*.md"))
    body = expected.read_text()
    fm = parse_frontmatter(body)
    assert fm["generated"] == "true"
    assert fm["source"] == "github-twin"
    assert fm["type"] == "rule"
    assert fm["language"] == "scala"
    assert fm["cluster_size"] == "12"
    assert "Prefer functional reactive style for stream handlers." in body
    assert "see PR #42 for the canonical example" in body
    assert "(https://github.com/acme/x/pull/42#discussion_r1)" in body


def test_empty_corpus_emits_index_and_scratch_readme(conn, cfg):
    out = export_wiki(conn, cfg)
    root = resolve_vault_root(cfg)
    assert (root / "index.md").exists()
    assert (root / "scratch" / "README.md").exists()
    assert (root / "rules" / "_index.md").exists()
    assert out["written"] >= 4  # index + 3 section indexes + scratch readme


# ---------- idempotency ----------


def test_second_export_writes_nothing(conn, cfg):
    _seed_rule(conn, text="Keep handlers small.", language="python")
    first = export_wiki(conn, cfg)
    assert first["written"] > 0
    second = export_wiki(conn, cfg)
    assert second["written"] == 0
    assert second["removed"] == 0
    assert second["unchanged"] >= first["written"]


# ---------- prune ----------


def test_removed_rule_gets_pruned(conn, cfg):
    _seed_rule(
        conn,
        text="Wrap every IO in resource scopes.",
        language="rust",
        external_id="r-rust-wrap-io",
    )
    export_wiki(conn, cfg)
    root = resolve_vault_root(cfg)
    rule_path = root / "rules" / "rust" / f"{rule_slug('Wrap every IO in resource scopes.')}.md"
    assert rule_path.exists()

    # Now delete the rule artifact + its chunks; re-export should prune.
    q.delete_artifact(
        conn,
        conn.execute("SELECT id FROM artifact WHERE external_id = 'r-rust-wrap-io'").fetchone()[
            "id"
        ],
    )
    out = export_wiki(conn, cfg)
    assert out["removed"] >= 1
    assert not rule_path.exists()


# ---------- hand-edit guard ----------


def test_handwritten_file_with_frontmatter_stripped_is_preserved(conn, cfg):
    """User can adopt a generated file by removing the `generated: true`
    frontmatter; subsequent exports must not overwrite or delete it."""
    _seed_rule(conn, text="Always validate user input.", language="python", external_id="r-px")
    export_wiki(conn, cfg)
    root = resolve_vault_root(cfg)
    rule_path = root / "rules" / "python" / f"{rule_slug('Always validate user input.')}.md"
    assert rule_path.exists()

    handwritten = "# I adopted this\n\nMy own notes go here.\n"
    rule_path.write_text(handwritten)

    # Re-export: file lacks `generated: true` so the orchestrator treats it
    # as adopted — preserves bytes exactly + counts it under `adopted`.
    out = export_wiki(conn, cfg)
    assert rule_path.read_text() == handwritten
    assert out["adopted"] >= 1
    # Pruning is scoped to generated files (list_generated_files filter),
    # so an adopted file is also never deleted, even if its rule was
    # removed from the DB.
    q.delete_artifact(
        conn,
        conn.execute("SELECT id FROM artifact WHERE external_id = 'r-px'").fetchone()["id"],
    )
    out2 = export_wiki(conn, cfg)
    assert rule_path.exists()
    assert rule_path.read_text() == handwritten
    # The new rule path *would* have been pruned if it were generated, but
    # since it's adopted it survives. removed count covers only generated
    # paths that fell out.
    assert out2["removed"] == 0


def test_list_generated_files_skips_handwritten(tmp_path):
    """Pure unit check for the prune-set membership rule."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "rules").mkdir()
    (root / "rules" / "gen.md").write_text(
        "---\ngenerated: true\nsource: github-twin\n---\n\n# x\n"
    )
    (root / "rules" / "mine.md").write_text("# my notes\n\nhello\n")
    gens = list_generated_files(root)
    assert (root / "rules" / "gen.md") in gens
    assert (root / "rules" / "mine.md") not in gens


# ---------- repo overview ----------


def test_repo_overview_renders_with_top_paths(conn, cfg):
    _seed_repo(conn, full_name="acme/x")
    # Two file chunks under repo acme/x at different paths so top_paths
    # has something to rank.
    for path in ["src/a.py", "src/b.py", "src/a.py"]:
        aid = q.upsert_artifact(
            conn,
            target_id=1,
            kind="file",
            external_id=f"f-{path}-{path.count('/')}",
            source_url=None,
            repo="acme/x",
            language="python",
            author_email=None,
            author_login="alice",
            created_at=None,
            decision=None,
            meta=None,
        )
        q.insert_chunk(
            conn,
            artifact_id=aid,
            kind="file",
            text=f"# {path}\n",
            context={"path": path, "language": "python"},
            language="python",
        )
    export_wiki(conn, cfg)
    root = resolve_vault_root(cfg)
    repo_page = root / "repos" / f"{repo_slug('acme/x')}.md"
    assert repo_page.exists()
    body = repo_page.read_text()
    assert "acme/x" in body
    assert "src/a.py" in body  # top file
    assert "alice" in body  # top contributor
    # Top-files block links to the internal per-file page, not GitHub directly.
    assert "[[files/acme__x/src/a.py|" in body


# ---------- per-file pages ----------


def test_file_page_renders_per_chunk_summaries(conn, cfg):
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="file",
        external_id="f-server",
        source_url=None,
        repo="acme/x",
        language="python",
        author_email=None,
        author_login="alice",
        created_at=None,
        decision=None,
        meta=None,
    )
    # Two chunks under the same file: one fully-AST-tagged, one bare.
    cid1 = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="file",
        text="def serve(): ...",
        context={"path": "src/server.py", "start_line": 10, "end_line": 30},
        language="python",
    )
    cid2 = q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="file",
        text="def handle(): ...",
        context={"path": "src/server.py"},
        language="python",
    )
    # Patch the per-chunk metadata the AST chunker would have set + a summary.
    conn.execute(
        "UPDATE chunk SET symbol_name='serve', node_kind='function_definition', "
        "summary='Entry point that binds the socket and dispatches to handle().' "
        "WHERE id=?",
        (cid1,),
    )
    conn.execute(
        "UPDATE chunk SET summary='Inner per-request handler.' WHERE id=?",
        (cid2,),
    )

    export_wiki(conn, cfg)
    root = resolve_vault_root(cfg)
    file_page = root / "files" / "acme__x" / "src" / "server.py.md"
    assert file_page.exists()
    body = file_page.read_text()
    fm = parse_frontmatter(body)
    assert fm["type"] == "file"
    assert fm["repo"] == "acme/x"
    assert fm["path"] == "src/server.py"
    assert fm["chunk_count"] == "2"
    # First chunk has full AST metadata + summary
    assert "`serve` (function_definition)" in body
    assert "lines 10–30" in body  # em-dash range
    assert "Entry point that binds the socket" in body
    # Second chunk has only summary; heading falls back to bare `chunk`
    assert "Inner per-request handler." in body


def test_file_page_handles_missing_summary(conn, cfg):
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="file",
        external_id="f-bare",
        source_url=None,
        repo="acme/x",
        language="python",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="file",
        text="x = 1",
        context={"path": "src/bare.py"},
        language="python",
    )
    export_wiki(conn, cfg)
    root = resolve_vault_root(cfg)
    file_page = root / "files" / "acme__x" / "src" / "bare.py.md"
    assert file_page.exists()
    body = file_page.read_text()
    assert "no summary yet" in body


# ---------- profiles ----------


def test_profile_placeholder_when_no_llm(conn, cfg):
    """User-mode target with one review_comment → placeholder profile."""
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="review_comment",
        external_id="c-1",
        source_url=None,
        repo="acme/x",
        language="python",
        author_email=None,
        author_login=None,  # user-mode: null
        created_at="2025-01-01T00:00:00Z",
        decision=None,
        meta=None,
    )
    q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="review_comment",
        text="Please add a docstring.",
        context={},
        language="python",
    )
    export_wiki(conn, cfg, profile_llm=None)
    root = resolve_vault_root(cfg)
    profile_page = root / "profiles" / "alice.md"
    assert profile_page.exists()
    body = profile_page.read_text()
    fm = parse_frontmatter(body)
    assert fm["type"] == "profile"
    assert fm["placeholder"] == "true"


def test_profile_uses_llm_when_provided(conn, cfg):
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="review_comment",
        external_id="c-llm",
        source_url=None,
        repo="acme/x",
        language="python",
        author_email=None,
        author_login=None,
        created_at="2025-01-01T00:00:00Z",
        decision=None,
        meta=None,
    )
    q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="review_comment",
        text="Please add tests for the boundary case.",
        context={},
        language="python",
    )
    export_wiki(conn, cfg, profile_llm=_FakeLLM())
    root = resolve_vault_root(cfg)
    profile_page = root / "profiles" / "alice.md"
    body = profile_page.read_text()
    assert "synthesized profile" in body
    fm = parse_frontmatter(body)
    assert fm.get("placeholder") != "true"
