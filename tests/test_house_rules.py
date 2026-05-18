"""`house_rules` MCP tool: render distilled rules as a Markdown block.

The retrieval mechanics are already covered by `test_distill.py` /
`test_queries.py`; these tests only pin the Markdown rendering and the
counts the tool reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.mcp_server.tools import house_rules
from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "house.sqlite", embed_dim=4)
    seed_target(db)
    yield db
    db.close()


def _seed_rule(
    conn,
    *,
    text: str,
    chunk_kind: str,
    language: str | None,
    cluster_size: int = 5,
    example_quote: str = "example",
    urls: list[str] | None = None,
    external_id: str | None = None,
    repo: str | None = None,
    author_login: str | None = None,
) -> int:
    """Seed one `kind='rule'` artifact + matching chunk. Mirrors what
    `distill_rules` would do but without the LLM."""
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="rule",
        external_id=external_id or f"r-{chunk_kind}-{text[:20]}",
        source_url=urls[0] if urls else None,
        repo=repo,
        language=language,
        author_email=None,
        author_login=author_login,
        created_at=None,
        decision=None,
        meta={
            "backend": "fake",
            "cluster_size": cluster_size,
            "example_quotes": [example_quote],
            "member_chunk_ids": [],
            "member_urls": urls or [],
            "member_repos": [],
            "rule_source": "review_comment" if chunk_kind == "rule" else "code",
        },
    )
    return q.insert_chunk(
        conn,
        artifact_id=aid,
        kind=chunk_kind,
        text=text,
        context={"language": language, "examples": [example_quote]},
        language=language,
    )


# ---------- shape of the rendered markdown ----------


def test_empty_corpus_returns_placeholder(conn):
    out = house_rules(conn)
    assert out["review_rules"] == 0
    assert out["code_rules"] == 0
    assert "No rules distilled yet" in out["markdown"]
    assert "gt distill" in out["markdown"]


def test_review_rules_only(conn):
    _seed_rule(
        conn,
        text="Prefer Async[F] over concrete IO in libraries.",
        chunk_kind="rule",
        language="scala",
        cluster_size=14,
        example_quote="Let's keep IO confined to Main.",
        urls=["https://github.com/me/x/pull/42#discussion_r123"],
    )
    out = house_rules(conn)
    md = out["markdown"]
    assert out["review_rules"] == 1
    assert out["code_rules"] == 0
    # Section header present
    assert "## Reviewer conventions" in md
    # Language bucket
    assert "### scala" in md
    # Rule body + cluster size + example + source link
    assert "**Prefer Async[F] over concrete IO in libraries.**" in md
    assert "from 14 examples" in md
    assert "> Let's keep IO confined to Main." in md
    assert "[source](https://github.com/me/x/pull/42#discussion_r123)" in md
    # No "Code patterns" section when there are no code rules
    assert "## Code patterns" not in md


def test_code_rules_only(conn):
    _seed_rule(
        conn,
        text="Wrap acquire/release in Resource.",
        chunk_kind="code_rule",
        language="scala",
        cluster_size=8,
    )
    out = house_rules(conn)
    md = out["markdown"]
    assert out["review_rules"] == 0
    assert out["code_rules"] == 1
    assert "## Code patterns" in md
    assert "**Wrap acquire/release in Resource.**" in md


def test_both_kinds_render_in_order(conn):
    _seed_rule(conn, text="review-rule", chunk_kind="rule", language="python")
    _seed_rule(conn, text="code-rule", chunk_kind="code_rule", language="python")
    out = house_rules(conn)
    md = out["markdown"]
    assert out["review_rules"] == 1
    assert out["code_rules"] == 1
    # Reviewer section appears before code-patterns section.
    assert md.index("## Reviewer conventions") < md.index("## Code patterns")


def test_groups_by_language_with_named_first(conn):
    _seed_rule(conn, text="r-python", chunk_kind="rule", language="python", external_id="r1")
    _seed_rule(conn, text="r-no-lang", chunk_kind="rule", language=None, external_id="r2")
    _seed_rule(conn, text="r-scala", chunk_kind="rule", language="scala", external_id="r3")
    md = house_rules(conn)["markdown"]
    # named languages sorted alphabetically, unspecified last
    py = md.index("### python")
    sc = md.index("### scala")
    unspec = md.index("### (language unspecified)")
    assert py < sc < unspec


def test_language_filter_restricts_results(conn):
    _seed_rule(conn, text="r-python", chunk_kind="rule", language="python", external_id="r1")
    _seed_rule(conn, text="r-scala", chunk_kind="rule", language="scala", external_id="r2")
    out = house_rules(conn, language="python")
    md = out["markdown"]
    assert out["review_rules"] == 1
    assert "r-python" in md
    assert "r-scala" not in md


def test_long_quote_is_truncated(conn):
    long_quote = "x " * 200
    _seed_rule(
        conn,
        text="rule with long quote",
        chunk_kind="rule",
        language="python",
        example_quote=long_quote,
    )
    md = house_rules(conn)["markdown"]
    # The quote line shouldn't be the original 400 chars verbatim;
    # truncation appends "…".
    assert "…" in md
    # Verify no line in the markdown is wildly long.
    for line in md.splitlines():
        if line.startswith("  > "):
            assert len(line) < 220


def test_returns_dict_with_markdown_and_counts(conn):
    """The tool's return shape is the contract MCP clients depend on."""
    _seed_rule(conn, text="x", chunk_kind="rule", language="python")
    out = house_rules(conn)
    assert set(out.keys()) == {"markdown", "review_rules", "code_rules"}
    assert isinstance(out["markdown"], str)
    assert isinstance(out["review_rules"], int)
    assert isinstance(out["code_rules"], int)


# ---------- repo / author_login / scope filters ----------


def test_repo_filter_restricts_to_dominant_repo(conn):
    _seed_rule(
        conn,
        text="x-rule",
        chunk_kind="rule",
        language="scala",
        repo="me/x",
        external_id="rx",
    )
    _seed_rule(
        conn,
        text="y-rule",
        chunk_kind="rule",
        language="scala",
        repo="me/y",
        external_id="ry",
    )
    out = house_rules(conn, repo="me/x")
    md = out["markdown"]
    assert out["review_rules"] == 1
    assert "x-rule" in md
    assert "y-rule" not in md


def test_author_login_filter_restricts_to_one_author(conn):
    _seed_rule(
        conn,
        text="alice-rule",
        chunk_kind="rule",
        language="scala",
        author_login="alice",
        external_id="ra",
    )
    _seed_rule(
        conn,
        text="bob-rule",
        chunk_kind="rule",
        language="scala",
        author_login="bob",
        external_id="rb",
    )
    out = house_rules(conn, author_login="alice")
    md = out["markdown"]
    assert out["review_rules"] == 1
    assert "alice-rule" in md
    assert "bob-rule" not in md


def test_scope_personal_does_not_filter_author_login_in_user_mode(conn):
    """Regression test for issue #13.

    User-mode ingest leaves `artifact.author_login = NULL` by design, so
    `scope="personal"` must NOT derive an author_login filter from the
    user-mode target's name — doing so would zero out every result against
    a real user-mode corpus. target_id alone narrows correctly.
    """
    # The conftest fixture seeded a user-mode target named 'me'.
    # Both seeded rules sit under target_id=1; in a real user-mode DB
    # author_login would be NULL, but we seed it here to prove the fix
    # doesn't accidentally re-introduce an author_login filter.
    _seed_rule(
        conn,
        text="me-rule",
        chunk_kind="rule",
        language="scala",
        author_login="me",
        external_id="ra",
    )
    _seed_rule(
        conn,
        text="bob-rule",
        chunk_kind="rule",
        language="scala",
        author_login="bob",
        external_id="rb",
    )
    out = house_rules(conn, scope="personal")
    md = out["markdown"]
    assert out["review_rules"] == 2
    assert "me-rule" in md
    assert "bob-rule" in md


def test_explicit_kwargs_win_over_scope(conn):
    """`scope="personal"` should not override an explicit author_login."""
    _seed_rule(
        conn,
        text="me-rule",
        chunk_kind="rule",
        language="scala",
        author_login="me",
        external_id="ra",
    )
    _seed_rule(
        conn,
        text="bob-rule",
        chunk_kind="rule",
        language="scala",
        author_login="bob",
        external_id="rb",
    )
    out = house_rules(conn, scope="personal", author_login="bob")
    md = out["markdown"]
    assert out["review_rules"] == 1
    assert "bob-rule" in md
    assert "me-rule" not in md
