"""Asymmetric query expansion for the BM25 leg of hybrid retrieval.

BM25 only matches exact tokens. A user query "search function" misses
code that calls it `func Search`, `searchEngine`, or `find_query`. This
module turns each query token into a small OR-group of alternates so
BM25 can bridge those gaps. Embeddings already capture synonymy, so we
deliberately do NOT touch the vector query — amanmcp's research
measured a -15pp regression when both backends are expanded, and our
own asymmetry regression test enforces the split.

Backends:

- `RuleExpander` (default) — deterministic, dependency-free. Generates
  case variants, camelCase ↔ snake_case ↔ kebab-case splits, and a
  hand-tuned table of language-idiom synonyms (function ↔ func ↔ fn).
  Fast (~µs per call), no network, no model.
- `OllamaExpander` (opt-in) — wraps an Ollama call ("give 3-5
  alternate terms per token") with a SQLite on-disk cache keyed by
  query hash. First call ~150–300 ms with `qwen3:0.6b`; cached calls
  ~ms. The cache is what makes this tolerable on the MCP hot path.
- `CompositeExpander` — unions Rule + (optional) Ollama, deduplicating.
  This is what `make_expander("ollama")` actually returns — even when
  the LLM is configured, the rule-based core runs alongside it so we
  don't lose deterministic wins to a flaky model.

The output type is `list[list[str]]`: one OR-group per *original*
token, original token always present at index 0, alternates after.
`store.queries._fts_match_from_groups` consumes that shape directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from github_twin.config import Config

log = logging.getLogger(__name__)


@runtime_checkable
class QueryExpander(Protocol):
    backend_id: str

    def expand(self, query_text: str) -> list[list[str]]:
        """Return one OR-group per whitespace-split token in `query_text`.
        Each group has the original token at index 0 and zero-or-more
        alternates after. Empty input → []. Implementations must be
        deterministic on cache hits.
        """
        ...


# ---------- rule-based core ----------


# Hand-curated synonym table. Keys are lowercase; matching is
# case-insensitive. Values include the key (so the table is symmetric:
# any term in a row pulls in the rest). Keep small and code-shaped —
# overly aggressive synonyms hurt BM25 precision faster than they help
# recall.
_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"function", "func", "fn", "method", "def", "procedure"}),
    frozenset({"class", "struct", "type", "record", "interface", "trait"}),
    frozenset({"search", "find", "lookup", "query", "fetch", "get"}),
    frozenset({"index", "indexer", "store", "engine"}),
    frozenset({"embed", "embedder", "embedding", "vector", "encode"}),
    frozenset({"test", "spec", "suite", "expect", "assert"}),
    frozenset({"error", "err", "fail", "failure", "exception"}),
    frozenset({"config", "settings", "options", "cfg"}),
    frozenset({"connect", "connection", "client", "session"}),
    frozenset({"retry", "backoff", "redrive"}),
    frozenset({"queue", "channel", "stream", "pipe"}),
    frozenset({"lock", "mutex", "semaphore", "latch"}),
    frozenset({"create", "new", "make", "init", "construct"}),
    frozenset({"delete", "remove", "drop", "destroy", "purge"}),
    frozenset({"update", "modify", "edit", "patch", "set"}),
    frozenset({"read", "load", "open", "scan"}),
    frozenset({"write", "save", "persist", "store"}),
    frozenset({"start", "begin", "launch", "spawn"}),
    frozenset({"stop", "halt", "kill", "shutdown", "close"}),
    frozenset({"parse", "decode", "deserialize"}),
    frozenset({"serialize", "encode", "marshal"}),
)


_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Token-extractor: alnum + underscore. Strips punctuation we don't want
# to feed FTS5 directly. `_fts_match_from_groups` re-quotes per token.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _case_variants(token: str) -> set[str]:
    """Return common case variants of `token` (lowercase, original,
    capitalized) — BM25 in FTS5 is case-insensitive by default but
    surfacing the variants is harmless and keeps the OR-group readable
    in logs / tests."""
    out = {token, token.lower()}
    if token.isupper() or token.islower():
        out.add(token.capitalize())
    return out


def _split_compounds(token: str) -> set[str]:
    """Generate sub-tokens from camelCase / snake_case / kebab-case
    inputs. `getUser_id` → {get, User, id}. Single-word inputs return
    the empty set so callers know to fall back to alternates only."""
    parts: set[str] = set()
    # Underscore + dash boundaries first.
    for chunk in re.split(r"[_\-]+", token):
        if not chunk:
            continue
        # camelCase / PascalCase boundaries within each chunk.
        for piece in _CAMEL_SPLIT.split(chunk):
            piece = piece.strip()
            if piece and piece.lower() != token.lower():
                parts.add(piece.lower())
    return parts


def _synonyms_for(token: str) -> set[str]:
    lower = token.lower()
    for group in _SYNONYM_GROUPS:
        if lower in group:
            return {w for w in group if w != lower}
    return set()


def _alternates_for(token: str) -> list[str]:
    """Compose case + compound + synonym alternates, dedup, drop the
    original. Result order is stable: synonyms before compounds before
    case variants, alphabetical within each band. (Stable output keeps
    the FTS5 query text reproducible across runs — easier to reason
    about cache keys and test assertions.)"""
    syns = sorted(_synonyms_for(token))
    compounds = sorted(_split_compounds(token))
    cases = sorted(_case_variants(token) - {token})
    seen: set[str] = set()
    out: list[str] = []
    for word in (*syns, *compounds, *cases):
        low = word.lower()
        if low == token.lower():
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(word)
    return out


class RuleExpander:
    """Deterministic rule-based expander. Always safe to leave on."""

    backend_id = "rule"

    def expand(self, query_text: str) -> list[list[str]]:
        tokens = _TOKEN_RE.findall(query_text or "")
        if not tokens:
            return []
        return [[tok, *_alternates_for(tok)] for tok in tokens]


# ---------- Ollama-backed expander with SQLite cache ----------


_OLLAMA_PROMPT = (
    "You expand search query tokens for a code-search engine. For each "
    "token below, output up to 5 alternate terms (synonyms, common "
    "abbreviations, language-keyword variants). Respond with strict JSON: "
    "an object mapping each input token to an array of alternates, no prose. "
    "Tokens: {tokens}"
)


class OllamaExpander:
    """LLM-backed expander. Caches results per-token in a tiny SQLite
    file so the MCP hot path stays fast on repeat queries. The cache
    is keyed on (model, token) — switching models invalidates without
    a manual flush."""

    backend_id = "ollama"

    def __init__(
        self,
        *,
        model: str = "qwen3:0.6b",
        host: str = "http://127.0.0.1:11434",
        cache_path: Path | None = None,
        timeout: float = 30.0,
        max_alternates_per_token: int = 5,
    ) -> None:
        self._model = model
        self._host = host
        self._timeout = timeout
        self._max_alternates = max_alternates_per_token
        self._cache = _ExpansionCache(cache_path) if cache_path is not None else None

    def expand(self, query_text: str) -> list[list[str]]:
        tokens = _TOKEN_RE.findall(query_text or "")
        if not tokens:
            return []

        # Per-token cache lookup.
        cached: dict[str, list[str]] = {}
        missing: list[str] = []
        if self._cache is not None:
            for tok in tokens:
                hit = self._cache.get(self._model, tok)
                if hit is not None:
                    cached[tok] = hit
                else:
                    missing.append(tok)
        else:
            missing = list(tokens)

        # One Ollama call for everything missing. Failures degrade to
        # empty alternates so retrieval still works.
        if missing:
            try:
                fresh = self._fetch(missing)
            except Exception as e:  # noqa: BLE001
                log.warning("OllamaExpander fetch failed (%s); using []", e)
                fresh = {tok: [] for tok in missing}
            if self._cache is not None:
                for tok, alts in fresh.items():
                    self._cache.put(self._model, tok, alts)
            cached.update(fresh)

        return [[tok, *cached.get(tok, [])] for tok in tokens]

    def _fetch(self, tokens: list[str]) -> dict[str, list[str]]:
        import ollama

        client = ollama.Client(host=self._host, timeout=self._timeout)
        prompt = _OLLAMA_PROMPT.format(tokens=", ".join(tokens))
        resp = client.generate(
            model=self._model,
            prompt=prompt,
            format="json",
            options={"temperature": 0.0},
        )
        raw = resp.get("response", "") or ""
        parsed = _parse_ollama_json(raw, tokens)
        # Trim to max_alternates and drop the original token from each list.
        out: dict[str, list[str]] = {}
        for tok in tokens:
            alts = [a for a in parsed.get(tok, []) if a.lower() != tok.lower()]
            out[tok] = alts[: self._max_alternates]
        return out


def _parse_ollama_json(raw: str, tokens: Iterable[str]) -> dict[str, list[str]]:
    """Best-effort JSON parsing. Small local models occasionally wrap the
    answer in prose or omit some tokens — return what we can salvage,
    fall through to empty for missing keys."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, list[str]] = {}
    for tok in tokens:
        alts = obj.get(tok) or obj.get(tok.lower()) or obj.get(tok.upper())
        if isinstance(alts, list):
            out[tok] = [str(a) for a in alts if isinstance(a, str | int | float)]
    return out


class _ExpansionCache:
    """Tiny SQLite-backed key/value store for query expansion.

    A separate DB file so the cache survives `gt embed --rebuild` and
    so concurrent MCP servers don't fight the main connection."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS expansion (
      key         TEXT PRIMARY KEY,
      alternates  TEXT NOT NULL,           -- JSON list
      created_at  REAL NOT NULL
    )
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(self._SCHEMA)

    @staticmethod
    def _key(model: str, token: str) -> str:
        digest = hashlib.sha1(f"{model}\0{token.lower()}".encode()).hexdigest()
        return digest

    def get(self, model: str, token: str) -> list[str] | None:
        row = self._conn.execute(
            "SELECT alternates FROM expansion WHERE key = ?",
            (self._key(model, token),),
        ).fetchone()
        if row is None:
            return None
        try:
            return list(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError):
            return None

    def put(self, model: str, token: str, alternates: list[str]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO expansion (key, alternates, created_at) VALUES (?, ?, ?)",
            (self._key(model, token), json.dumps(alternates), time.time()),
        )
        self._conn.commit()


# ---------- composite + factory ----------


class CompositeExpander:
    """Union the alternates from multiple expanders, preserving order
    and removing duplicates. Used by `make_expander("ollama")` so the
    rule-based core always runs alongside the LLM call."""

    backend_id = "composite"

    def __init__(self, *expanders: QueryExpander) -> None:
        self._expanders = tuple(expanders)

    def expand(self, query_text: str) -> list[list[str]]:
        results = [exp.expand(query_text) for exp in self._expanders]
        # Sieve: tokens come from the first non-empty result; alternates
        # union across all results.
        primary = next((r for r in results if r), [])
        if not primary:
            return []
        out: list[list[str]] = []
        for i, group in enumerate(primary):
            seen: set[str] = set()
            merged: list[str] = []
            for token in group:
                low = token.lower()
                if low in seen:
                    continue
                seen.add(low)
                merged.append(token)
            for r in results[1:]:
                if i >= len(r):
                    continue
                for token in r[i][1:]:  # skip the original (already added)
                    low = token.lower()
                    if low in seen:
                        continue
                    seen.add(low)
                    merged.append(token)
            out.append(merged)
        return out


def make_expander(
    backend: str,
    *,
    ollama_model: str = "qwen3:0.6b",
    ollama_host: str = "http://127.0.0.1:11434",
    cache_path: Path | None = None,
) -> QueryExpander | None:
    """Build the configured expander. Returns None when expansion is off."""
    if backend == "off":
        return None
    if backend == "rule":
        return RuleExpander()
    if backend == "ollama":
        return CompositeExpander(
            RuleExpander(),
            OllamaExpander(
                model=ollama_model,
                host=ollama_host,
                cache_path=cache_path,
            ),
        )
    raise ValueError(f"unknown query_expansion backend: {backend!r}")


def expander_from_config(cfg: Config) -> QueryExpander | None:
    """Build the expander from a Config. Resolves expansion_cache_path
    relative to `cfg.paths.data_dir` when not absolute."""
    backend = cfg.retrieval.query_expansion
    if backend == "off":
        return None
    cache_path: Path | None = None
    if backend == "ollama":
        cache_path = cfg.retrieval.expansion_cache_path or (
            cfg.paths.data_dir / "query_expansion_cache.sqlite"
        )
    return make_expander(
        backend,
        ollama_model=cfg.retrieval.ollama_model,
        ollama_host=cfg.retrieval.ollama_host,
        cache_path=cache_path,
    )
