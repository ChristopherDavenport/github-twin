"""Developer-profile synthesis.

`gt distill` produces clustered *rules*; this module produces a
different artifact — a short Markdown sketch of one developer's voice
suitable for pasting into an agent's system prompt.

Why a separate module: distill rules clusters first, then summarizes
each cluster; profiling is one LLM call over a flat list of recent
comments. Same `TextLLM` dispatch, different prompt + different
return shape. Caching is handled by the MCP-tool layer
(`mcp_server/tools.py:developer_profile`); this module is pure
synthesis.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from github_twin.eval.llm import TextLLM
from github_twin.store.queries import ChunkRow

# Cap per-comment text in the prompt so a single 50-comment user prompt
# stays well under any reasonable model's context window. ~600 chars per
# comment × 50 comments = 30 KB; comfortable for every backend we ship.
_MAX_COMMENT_CHARS = 600

_SYSTEM_PROMPT = (
    "You are summarizing a developer's review voice for use as background "
    "context in a coding agent's system prompt. Read these review comments "
    "the developer wrote and produce 2–3 short paragraphs of Markdown "
    "covering: their tone (formal / casual / blunt / encouraging), the "
    "concerns they routinely raise (testing? performance? naming? type "
    "safety?), what they value (consistency? simplicity? explicit error "
    "handling?), and any characteristic phrases they use. Be specific — "
    "name the libraries, patterns, and identifiers they mention, not "
    "generic descriptors. Output Markdown only — no preamble, no "
    "meta-commentary, no surrounding code fences."
)


def _truncate(text: str, max_chars: int = _MAX_COMMENT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def sample_hash(comments: Iterable[ChunkRow]) -> str:
    """Stable hash over the chunk_ids of the sample. The cache uses this
    to detect when a new `gt sync` has changed the set of recent
    comments — sorted so order doesn't matter."""
    ids = sorted(c.id for c in comments)
    payload = ",".join(str(i) for i in ids).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def synthesize_profile(
    llm: TextLLM,
    comments: list[ChunkRow],
    *,
    max_tokens: int = 600,
) -> str:
    """Build a 2–3 paragraph Markdown profile from a list of recent
    review comments. Returns the LLM's raw response (trimmed)."""
    if not comments:
        return ""
    bodies = "\n\n---\n\n".join(_truncate(c.text) for c in comments if c.text)
    user_prompt = f"Recent review comments ({len(comments)}):\n\n{bodies}"
    out = llm.complete(system=_SYSTEM_PROMPT, user=user_prompt, max_tokens=max_tokens)
    return out.strip()
