"""LLM-generated chunk summaries for contextual retrieval.

amanmcp's research reports that prepending an LLM-generated 1–2 sentence
description to each chunk before embedding moves Tier-1 pass-rate from
~75% to ~92% on a code corpus. The deterministic header we already
ship covers path + symbol + node_kind + docstring; what it can't do is
bridge between a natural-language query ("Eq instance for a wrapper
case class") and a code chunk whose only NL surface is a one-word
identifier (`VaultSecretEq`). A summary written by a model fills that
gap.

This module is a separate pass — not part of ingest — because
summarization is the slowest step in the pipeline (~50–200 ms / chunk
even with a small local model). Running it on demand lets the user pick
when to pay the cost and which model to use.

Output is persisted in `chunk.summary` (TEXT, nullable). When non-NULL,
`embed.prefix.build_header` includes it in the chunk header. Bumping
`pipeline.EMBED_TEXT_VERSION` after summaries land triggers a full
re-embed so the vector index actually sees them.

`gt summarize` is the CLI entry; `run_summarize` is the pipeline-level
function the MCP `sync` tool wires into.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Iterable

from github_twin.eval.llm import TextLLM
from github_twin.store import queries as q
from github_twin.store.db import transaction

log = logging.getLogger(__name__)

Reporter = Callable[[str], None]


def _noop(_: str) -> None:
    return None


# Per-kind prompts. Each takes the chunk's `text` (and optionally a few
# bits of context) and asks for a single-sentence NL description. The
# system prompt locks output length and forbids markdown / preamble so
# we can just take the raw response.

_CODE_SYSTEM = (
    "You write one-sentence descriptions of code for a retrieval index. "
    "Output exactly one sentence (≤25 words), no markdown, no preamble. "
    "Name what the code does, the key identifiers, and the type of thing "
    "it is (function / class / method / type / interface / impl). Do not "
    "quote the code back to me."
)

_COMMIT_SYSTEM = (
    "You write one-sentence descriptions of commit messages for a "
    "retrieval index. Output exactly one sentence (≤25 words) capturing "
    "what changed and why. No markdown, no preamble. Strip CI tags."
)

# `kinds` we'll actually summarize. Other kinds are excluded because:
# - `review_comment`: already NL; summarizing would compress with loss.
# - `pr_summary`: already title + body; structurally similar to a summary.
# - `rule`: distilled NL output of `gt distill`; further compression
#    is counterproductive.
_SUPPORTED_KINDS: tuple[str, ...] = ("code", "file", "code_rule", "commit_message")


def _prompt_for(chunk: q.ChunkRow) -> tuple[str, str]:
    """Return (system, user) prompt strings for a chunk. The user prompt
    embeds a short context line + the chunk text; the system prompt is
    fixed per kind. We pull only the smallest context fields that help
    the model — overstuffing burns tokens and degrades small models."""
    ctx = chunk.context or {}
    if chunk.kind == "commit_message":
        repo = ctx.get("repo") or ""
        user = (
            f"Repo: {repo}\nCommit message:\n"
            f"---\n{chunk.text[:2000]}\n---\n\nOne-sentence description:"
        )
        return _COMMIT_SYSTEM, user

    # code / file / code_rule
    path = ctx.get("path") or ""
    language = ctx.get("language") or ""
    symbol = ctx.get("symbol_name") or ""
    node_kind = ctx.get("node_kind") or ""
    header_bits = [b for b in (path, symbol, node_kind, language) if b]
    header = " :: ".join(header_bits) if header_bits else "<no metadata>"
    user = f"Location: {header}\nCode:\n---\n{chunk.text[:2400]}\n---\n\nOne-sentence description:"
    return _CODE_SYSTEM, user


def _clean_summary(text: str) -> str:
    """Strip the noise small models love to add: leading bullets,
    Markdown bold/italic, surrounding quotes, trailing chatter on a
    second line. Keep the first non-empty line and bound the length."""
    text = (text or "").strip()
    if not text:
        return ""
    # First non-empty line only.
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    # Strip bullet/markdown leaders.
    for prefix in ("- ", "* ", "• ", "1. ", "Description: ", "Summary: "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    # Strip surrounding quotes if present.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "`"):
        text = text[1:-1]
    # Hard cap (~320 chars ≈ 50 words). Anything longer than this is the
    # model misbehaving and would dilute embeddings.
    if len(text) > 320:
        text = text[:320].rsplit(" ", 1)[0] + "…"
    return text.strip()


def summarize_chunks(
    conn: sqlite3.Connection,
    llm: TextLLM,
    *,
    kinds: tuple[str, ...] = _SUPPORTED_KINDS,
    limit: int | None = None,
    batch_size: int = 8,
    max_tokens: int = 80,
    report: Reporter = _noop,
    rebuild: bool = False,
) -> int:
    """Generate summaries for chunks missing one. Returns count written.

    `kinds` defaults to the code-shaped chunks; pass a narrower tuple to
    target one kind. `rebuild=True` clears existing summaries for the
    given kinds first — used when changing models / prompts.
    """
    bad_kinds = [k for k in kinds if k not in _SUPPORTED_KINDS]
    if bad_kinds:
        raise ValueError(
            f"unsupported summary kinds: {bad_kinds} (supported: {list(_SUPPORTED_KINDS)})"
        )
    if rebuild:
        n = q.clear_chunk_summaries(conn, kinds=kinds)
        report(f"rebuild: cleared {n} existing summaries for {list(kinds)}")

    total_pending = conn.execute(
        f"SELECT COUNT(*) AS n FROM chunk "
        f"WHERE summary IS NULL AND kind IN ({','.join('?' * len(kinds))})",
        kinds,
    ).fetchone()["n"]
    cap = total_pending if limit is None else min(total_pending, limit)
    report(f"summarizing {cap} chunks (kinds={list(kinds)}) with {llm.backend_id}")

    done = 0
    t_start = time.monotonic()
    for chunk in _take(q.pending_summary_chunks(conn, kinds=kinds, batch_size=batch_size), cap):
        try:
            system, user = _prompt_for(chunk)
            raw = llm.complete(system=system, user=user, max_tokens=max_tokens)
            summary = _clean_summary(raw)
        except Exception as exc:  # noqa: BLE001
            # Don't kill the whole run on one chunk; log and move on.
            log.warning("summarize skip chunk %d: %s", chunk.id, exc)
            continue
        if not summary:
            # Empty / unparseable output is worse than no summary —
            # leave NULL so a later run with a better model can fill it.
            continue
        with transaction(conn):
            q.write_chunk_summary(conn, chunk_id=chunk.id, summary=summary)
        done += 1
        if done % 25 == 0 or done == cap:
            elapsed = time.monotonic() - t_start
            rate = done / elapsed if elapsed > 0 else 0.0
            pct = (done / cap * 100.0) if cap else 0.0
            eta_str = (
                f", ETA ~{_fmt_duration((cap - done) / rate)}" if rate > 0 and done < cap else ""
            )
            report(
                f"  ... {done}/{cap}  ({pct:.1f}%, {rate:.2f} cps, "
                f"{_fmt_duration(elapsed)} elapsed{eta_str})"
            )
    report(f"summarized {done} chunks")
    return done


def _take(it: Iterable[q.ChunkRow], n: int) -> Iterable[q.ChunkRow]:
    if n <= 0:
        return
    for i, x in enumerate(it):
        if i >= n:
            return
        yield x


def _fmt_duration(seconds: float) -> str:
    """Render a duration in compact human form: 12s, 4m54s, 7h40m."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    h, rem = divmod(s, 3600)
    return f"{h}h{rem // 60:02d}m"
