"""LLM chunk summaries: pipeline + idempotency + clean-up behavior.

Uses a FakeLLM rather than calling Ollama, so these are fast and
deterministic. The LLM's behavior — and the per-kind prompts — are
exercised together because the prompts shape what gets sent to the
model, and we want one place to assert the contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from github_twin.process.summarize import _clean_summary, summarize_chunks
from github_twin.store import queries as q
from github_twin.store.db import open_db
from tests.conftest import seed_target


@dataclass
class FakeLLM:
    """Records every call; returns a canned response. Tests inspect
    `.calls` to assert prompt shape per kind. `delay` simulates a slow
    network backend so the concurrent-pool tests can verify parallelism."""

    backend_id: str = "fake"
    response: str = "Validates auth headers and dispatches to the right handler."
    calls: list[tuple[str, str]] = field(default_factory=list)
    raise_on_call: Exception | None = None
    delay: float = 0.0

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        if self.delay:
            time.sleep(self.delay)
        self.calls.append((system, user))
        if self.raise_on_call:
            raise self.raise_on_call
        return self.response


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "summarize.sqlite", embed_dim=4)
    seed_target(db)
    yield db
    db.close()


def _seed_code(
    conn,
    *,
    text: str,
    path: str = "src/x.py",
    symbol: str = "f",
    node_kind: str = "function_definition",
) -> int:
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="commit",
        external_id=f"a-{symbol}-{path}",
        source_url=None,
        repo="me/x",
        language="python",
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    return q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="code",
        text=text,
        context={"path": path, "symbol_name": symbol, "node_kind": node_kind, "language": "python"},
        language="python",
    )


def _seed_review(conn, *, text: str = "LGTM, ship it") -> int:
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="review_comment",
        external_id="r-1",
        source_url=None,
        repo="me/x",
        language=None,
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    return q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="review_comment",
        text=text,
        context={"repo": "me/x", "pr_number": 1},
        language=None,
    )


# ---------- happy path ----------


def test_summarize_writes_to_chunk_summary(conn):
    cid = _seed_code(conn, text="def handle(req):\n    return _dispatch(req)")
    llm = FakeLLM(response="Validates auth headers and dispatches the request.")
    n = summarize_chunks(conn, llm)
    assert n == 1
    row = conn.execute("SELECT summary FROM chunk WHERE id=?", (cid,)).fetchone()
    assert row["summary"] == "Validates auth headers and dispatches the request."


def test_summarize_is_idempotent(conn):
    """Second run finds nothing to do."""
    _seed_code(conn, text="def f(): pass")
    llm = FakeLLM()
    assert summarize_chunks(conn, llm) == 1
    assert summarize_chunks(conn, llm) == 0


def test_summarize_skips_unsupported_kinds_by_default(conn):
    """review_comment is NL already; default kinds exclude it."""
    _seed_review(conn)
    _seed_code(conn, text="def x(): pass")
    llm = FakeLLM()
    n = summarize_chunks(conn, llm)
    assert n == 1
    # The review_comment chunk stays NULL.
    rows = conn.execute("SELECT kind, summary FROM chunk ORDER BY id").fetchall()
    by_kind = {r["kind"]: r["summary"] for r in rows}
    assert by_kind["review_comment"] is None
    assert by_kind["code"] is not None


def test_summarize_respects_explicit_kinds(conn):
    """Narrow run targets only the requested kind."""
    _seed_code(conn, text="def a(): pass", symbol="a")
    _seed_code(conn, text="class B: pass", symbol="B", node_kind="class_definition")
    llm = FakeLLM()
    n = summarize_chunks(conn, llm, kinds=("code",))
    assert n == 2


def test_summarize_rejects_unsupported_kind():
    """Asking to summarize review_comment is a programmer error — fail
    loudly so the bad config doesn't silently produce bad summaries."""
    with pytest.raises(ValueError, match="unsupported"):
        summarize_chunks(None, FakeLLM(), kinds=("review_comment",))


def test_summarize_limit_caps_count(conn):
    for i in range(5):
        _seed_code(conn, text=f"def f{i}(): pass", symbol=f"f{i}", path=f"src/{i}.py")
    llm = FakeLLM()
    n = summarize_chunks(conn, llm, limit=2)
    assert n == 2
    # Three rows are still NULL.
    pending = conn.execute("SELECT COUNT(*) AS n FROM chunk WHERE summary IS NULL").fetchone()
    assert pending["n"] == 3


def test_summarize_failure_on_one_chunk_does_not_abort_run(conn):
    """If the LLM raises on a chunk, log + skip — keep the others."""
    cid1 = _seed_code(conn, text="def a(): pass", symbol="a", path="a.py")
    _seed_code(conn, text="def b(): pass", symbol="b", path="b.py")
    llm = FakeLLM()

    # Make the LLM raise only on the first call.
    original_complete = llm.complete
    state = {"n": 0}

    def flaky(*, system, user, max_tokens=512):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("simulated model failure")
        return original_complete(system=system, user=user, max_tokens=max_tokens)

    llm.complete = flaky  # type: ignore[method-assign]
    n = summarize_chunks(conn, llm)
    assert n == 1
    # First chunk left NULL, second populated.
    row1 = conn.execute("SELECT summary FROM chunk WHERE id=?", (cid1,)).fetchone()
    assert row1["summary"] is None


def test_summarize_empty_output_leaves_null(conn):
    cid = _seed_code(conn, text="def x(): pass")
    llm = FakeLLM(response="   ")
    n = summarize_chunks(conn, llm)
    assert n == 0
    row = conn.execute("SELECT summary FROM chunk WHERE id=?", (cid,)).fetchone()
    assert row["summary"] is None


# ---------- prompt shape ----------


def test_code_prompt_includes_location_header(conn):
    _seed_code(
        conn,
        text="def f(): return 1",
        symbol="f",
        node_kind="function_definition",
        path="src/router.py",
    )
    llm = FakeLLM()
    summarize_chunks(conn, llm)
    system, user = llm.calls[0]
    assert "Output exactly one sentence" in system
    assert "src/router.py" in user
    assert "function_definition" in user
    assert "def f(): return 1" in user


def test_commit_message_prompt_uses_commit_system_prompt(conn):
    aid = q.upsert_artifact(
        conn,
        target_id=1,
        kind="commit",
        external_id="c-1",
        source_url=None,
        repo="me/x",
        language=None,
        author_email=None,
        author_login=None,
        created_at=None,
        decision=None,
        meta=None,
    )
    q.insert_chunk(
        conn,
        artifact_id=aid,
        kind="commit_message",
        text="fix: handle empty input",
        context={"repo": "me/x", "commit_sha": "abc1234"},
    )
    llm = FakeLLM()
    summarize_chunks(conn, llm, kinds=("commit_message",))
    system, user = llm.calls[0]
    assert "commit messages" in system
    assert "fix: handle empty input" in user


# ---------- rebuild ----------


def test_rebuild_clears_existing_summaries(conn):
    cid = _seed_code(conn, text="def f(): pass")
    llm = FakeLLM(response="first pass")
    summarize_chunks(conn, llm)
    assert (
        conn.execute("SELECT summary FROM chunk WHERE id=?", (cid,)).fetchone()["summary"]
        == "first pass"
    )

    llm2 = FakeLLM(response="second pass after rebuild")
    n = summarize_chunks(conn, llm2, rebuild=True)
    assert n == 1
    final = conn.execute("SELECT summary FROM chunk WHERE id=?", (cid,)).fetchone()["summary"]
    assert final == "second pass after rebuild"


# ---------- _clean_summary ----------


def test_clean_summary_strips_bullets_and_quotes():
    assert _clean_summary("- Validates input.") == "Validates input."
    assert _clean_summary('"Validates input."') == "Validates input."
    assert _clean_summary("Summary: validates input.") == "validates input."


def test_clean_summary_takes_first_nonempty_line():
    raw = "\n\n  Validates the request.\n\nExtra noise here.\n"
    assert _clean_summary(raw) == "Validates the request."


def test_clean_summary_caps_length():
    long = "x " * 500
    out = _clean_summary(long)
    assert len(out) <= 321  # 320 + ellipsis
    assert out.endswith("…")


def test_clean_summary_empty_input_returns_empty():
    assert _clean_summary("") == ""
    assert _clean_summary("   ") == ""


# ---------- concurrency ----------


def test_concurrent_run_writes_all_chunks(conn):
    """With concurrency > 1, every chunk should still get a summary.
    Arrival order is non-deterministic, so we compare on a set."""
    ids = [
        _seed_code(conn, text=f"def f{i}(): return {i}", symbol=f"f{i}", path=f"src/{i}.py")
        for i in range(6)
    ]
    llm = FakeLLM(response="Returns a constant.")
    n = summarize_chunks(conn, llm, concurrency=4)
    assert n == 6
    rows = conn.execute(
        "SELECT id, summary FROM chunk WHERE id IN (?, ?, ?, ?, ?, ?)", ids
    ).fetchall()
    by_id = {r["id"]: r["summary"] for r in rows}
    assert set(by_id.keys()) == set(ids)
    assert all(s == "Returns a constant." for s in by_id.values())


def test_concurrent_run_parallelizes_llm_calls(conn):
    """Wall clock should be meaningfully under (n * delay) when concurrency
    > 1. Loose bound (60% of serial) absorbs CI noise while still proving
    that calls overlap."""
    for i in range(8):
        _seed_code(conn, text=f"def g{i}(): pass", symbol=f"g{i}", path=f"src/g{i}.py")
    delay = 0.05
    llm = FakeLLM(delay=delay)
    t0 = time.monotonic()
    n = summarize_chunks(conn, llm, concurrency=4)
    elapsed = time.monotonic() - t0
    assert n == 8
    # Serial would be >= 8 * 0.05 = 0.40s. With concurrency=4 we expect
    # roughly 2 * 0.05 = 0.10s; allow generous headroom up to 0.24s.
    assert elapsed < 8 * delay * 0.6, f"expected parallel speedup, got {elapsed:.3f}s"


def test_concurrent_run_isolates_per_chunk_failures(conn):
    """One worker raising must not abort the rest. The failed chunk stays
    NULL; others land. Uses a thread-safe counter to fail exactly one
    call regardless of which thread it lands on."""
    ids = [
        _seed_code(conn, text=f"def h{i}(): pass", symbol=f"h{i}", path=f"src/h{i}.py")
        for i in range(6)
    ]
    llm = FakeLLM(response="ok.")
    import threading

    lock = threading.Lock()
    state = {"n": 0}
    original_complete = llm.complete

    def flaky(*, system, user, max_tokens=512):
        with lock:
            state["n"] += 1
            should_raise = state["n"] == 3
        if should_raise:
            raise RuntimeError("simulated model failure")
        return original_complete(system=system, user=user, max_tokens=max_tokens)

    llm.complete = flaky  # type: ignore[method-assign]
    n = summarize_chunks(conn, llm, concurrency=4)
    assert n == 5
    rows = conn.execute(
        "SELECT id, summary FROM chunk WHERE id IN (?, ?, ?, ?, ?, ?)", ids
    ).fetchall()
    populated = [r for r in rows if r["summary"] is not None]
    null = [r for r in rows if r["summary"] is None]
    assert len(populated) == 5
    assert len(null) == 1


def test_concurrent_run_honors_limit(conn):
    """With limit=3 and 10 pending at concurrency=4, only 3 summaries
    should be written and at most 3 + (concurrency - 1) extra in-flight
    submissions should occur. In practice the streaming pool short-
    circuits at exactly `limit` calls because `_take` enforces the cap
    on the upstream generator."""
    for i in range(10):
        _seed_code(conn, text=f"def k{i}(): pass", symbol=f"k{i}", path=f"src/k{i}.py")
    llm = FakeLLM()
    n = summarize_chunks(conn, llm, limit=3, concurrency=4)
    assert n == 3
    # _take caps the iterator at 3, so the pool sees at most 3 chunks.
    assert len(llm.calls) == 3
    pending = conn.execute("SELECT COUNT(*) AS n FROM chunk WHERE summary IS NULL").fetchone()
    assert pending["n"] == 7
