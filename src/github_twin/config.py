from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> Path:
    """Pick the default data directory based on environment + cwd state.

    Priority:
      1. `./data` in cwd if it already exists — backward-compat for
         pre-XDG installs (early adopters running `uv run gt sync`
         from the source tree had their corpus land here).
      2. `$XDG_DATA_HOME/github-twin` when XDG_DATA_HOME is set
         (Linux convention; macOS users frequently set this too).
      3. `~/.local/share/github-twin` (XDG fallback per spec).

    The `GT_PATHS__DATA_DIR` env var always wins — this default only
    fires when nothing else is configured.
    """
    cwd_data = Path.cwd() / "data"
    if cwd_data.exists():
        return cwd_data
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "github-twin"


class PathsCfg(BaseModel):
    data_dir: Path = Field(default_factory=_default_data_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db.sqlite"


class VectorStoreCfg(BaseModel):
    # 'sqlite-vec' (default, brute-force KNN, no extra deps)
    # | 'faiss' (opt-in: pip install github-twin[faiss], loads vectors into
    # RAM at startup, scales to ~10M vectors).
    backend: str = "sqlite-vec"


class RetrievalCfg(BaseModel):
    """Asymmetric query-expansion for the BM25 leg of hybrid search.

    BM25 only matches exact tokens; embeddings already capture synonymy.
    Setting `query_expansion = 'rule'` flips on a deterministic table of
    code-shaped synonyms (function/func/fn, search/find/lookup, ...);
    'ollama' adds a local LLM pass on top, cached on disk so the MCP hot
    path stays fast. Vector queries are NEVER expanded (research and
    our own regression test agree it makes things worse there).
    """

    # 'off' | 'rule' (default — free, deterministic) | 'ollama'.
    query_expansion: str = "rule"
    # Ollama model used when query_expansion='ollama'. Small + fast.
    ollama_model: str = "qwen3:0.6b"
    ollama_host: str = "http://127.0.0.1:11434"
    # Cache path is resolved against `paths.data_dir` when None.
    expansion_cache_path: Path | None = None
    # Exponential half-life (in days) for recency-weighted re-ranking inside
    # `hybrid_search`. None / 0 = off (current behavior). When set, the
    # fused RRF score of each candidate is multiplied by
    # `0.5 ** (age_days / half_life_days)` before the final top-k slice.
    # Decay applies only to style-bearing artifact kinds (commit, pr,
    # review_comment, issue_comment); file-at-HEAD and synthesized rule
    # artifacts are left untouched. `predict_review_outcome` bypasses
    # hybrid_search and is unaffected by design (its inverse-distance vote
    # weighting is calibrated on raw L2).
    recency_half_life_days: float | None = None


class EmbedCfg(BaseModel):
    # 'ollama' (default, local) | 'sentence_transformers' (local, opt-in via
    # [st] extra) | 'gemini' (remote — sends chunk text to Google; uses the
    # google-genai dep that's already pulled in by distill/eval).
    backend: str = "ollama"
    model: str = "nomic-embed-text"
    dim: int = 768
    ollama_host: str = "http://127.0.0.1:11434"
    # Default batch size used by `gt embed` and `gt sync`. 128 balances Ollama
    # latency vs. memory per request; raise to 256–512 for org-scale ingest if
    # the host has the RAM.
    batch_size: int = 128
    # Only used by the sentence_transformers backend: 'cuda' | 'mps' | 'cpu'
    # | None (auto-detect).
    device: str | None = None


class IngestCfg(BaseModel):
    since: str = "2018-01-01"
    include_repos: list[str] = Field(default_factory=list)
    exclude_repos: list[str] = Field(default_factory=list)
    exclude_paths: list[str] = Field(
        default_factory=lambda: [
            "**/*.lock",
            "**/package-lock.json",
            "**/yarn.lock",
            "**/Cargo.lock",
            "**/vendor/**",
            "**/node_modules/**",
            "**/dist/**",
            "**/build/**",
            "**/.min.js",
        ]
    )
    # Org-mode file ingest knobs (O-C). Default is process-and-purge so a
    # medium org doesn't fill the disk with 50–200 GB of clones.
    cache_clones: bool = False
    clones_dir: Path = Path("./data/clones")
    # Repos above this size are skipped entirely (giant monorepos, datasets
    # checked into git, etc.). 500 MB is generous for source code.
    max_repo_size_kb: int = 500_000
    # Commits ingest: walk a deep local clone (git log + git show) instead of
    # paginating /repos/{r}/commits + per-sha patch fetches. Default; flip to
    # False to fall back to the API path during rollback.
    use_local_git_for_commits: bool = True


class IdentityCfg(BaseModel):
    extra_emails: list[str] = Field(default_factory=list)
    ignore_emails: list[str] = Field(default_factory=list)


class SummarizeCfg(BaseModel):
    """`gt summarize` — generates a 1-sentence NL description per code chunk
    and stores it in `chunk.summary`. The embed-time prefix then includes
    it so vector queries can bridge NL → identifier-only code chunks.

    Default backend is `ollama` so the path stays fully local; switch to
    `claude` / `gemini` when the API keys are set and you want crisper
    output (the prompt is small so cost is bounded). `auto` mirrors
    `DistillCfg.backend` — Claude > Gemini > Ollama by API-key presence.
    """

    backend: str = "auto"
    # Ollama model. `qwen2.5-coder:7b` is best for code; `llama3.2` works
    # in a pinch (default in dev) but produces fuzzier summaries.
    ollama_model: str = "llama3.2"
    claude_model: str = "claude-haiku-4-5-20251001"
    gemini_model: str = "gemini-2.5-flash"
    # Cap the per-chunk completion length (tokens). qwen3-family "thinking"
    # models consume budget on internal reasoning before emitting visible
    # output, so 200 headroom covers both classic and thinking models;
    # `_clean_summary` trims to one sentence anyway.
    max_tokens: int = 200
    # Cap for the `developer_profile` MCP tool, which asks for a 2–3
    # paragraph response instead of a single sentence. Larger budget
    # without affecting per-chunk summarize calls.
    profile_max_tokens: int = 600
    # Kinds to summarize. Keep code-shaped; NL kinds opt out by design.
    kinds: tuple[str, ...] = ("code", "file", "code_rule", "commit_message")


class AuthCfg(BaseModel):
    """GitHub auth resolution + device-flow OAuth.

    `client_id` is the public Client ID of the github-twin OAuth App
    (https://github.com/settings/applications/3603560). Public by
    design — device flow does not use a client secret. Override via
    `GT_AUTH__CLIENT_ID` for testing or downstream forks.
    """

    client_id: str = "Ov23liAUxXgwgIJp6jqZ"
    # Default scopes for `gt auth login`. Mirrors the scopes the README
    # asks PAT users to grant: `repo` for private repo + PR access,
    # `read:org` for org membership, `user:email` for the identity sweep.
    default_scopes: str = "repo read:org user:email"


class WikiCfg(BaseModel):
    """`gt wiki export` — materialize a markdown vault on top of the SQLite
    corpus. The vault is Obsidian-compatible (frontmatter + `[[wikilinks]]`)
    and sits under `paths.data_dir / 'wiki'` by default; override per-run with
    `gt wiki export --out PATH`.

    Round-trip: any `.md` you drop into `<vault>/scratch/` is ingested as a
    `kind='note'` artifact on the next `gt sync` and feeds into hybrid
    retrieval like any other chunk. Auto-generated wiki files carry a
    `generated: true` frontmatter flag so the scratch ingester (which only
    scans `scratch/`) can never loop on its own output.
    """

    enabled: bool = True
    # None resolves to `paths.data_dir / 'wiki'` at call time.
    out: Path | None = None
    # Window size for splitting scratch notes into chunks. ~1200 chars
    # keeps each chunk's embed-time prefix dominant in vector space while
    # still capturing a paragraph or two of narrative.
    note_chunk_chars: int = 1200


class DistillCfg(BaseModel):
    # Sonnet is the cost/quality sweet spot for ~20-50 cluster runs; bump to
    # claude-opus-4-7 if rules feel shallow.
    claude_model: str = "claude-sonnet-4-6"
    # Flash is free-tier friendly and plenty for extraction; switch to
    # "gemini-2.5-pro" for paid + best quality.
    gemini_model: str = "gemini-2.5-flash"
    ollama_model: str = "llama3.2"
    # 'auto' precedence: Claude -> Gemini -> Ollama, based on which API key is set.
    backend: str = "auto"
    min_cluster_size: int = 3
    # Skip clusters above this size — usually means the corpus collapsed into
    # one giant blob and the rule won't be coherent. 100 is a sane org-scale
    # ceiling; user-mode rarely produces clusters above ~40 anyway.
    max_cluster_size: int = 100
    # HDBSCAN seed for reproducibility.
    random_state: int = 42


class Config(BaseSettings):
    """Top-level config. Loaded from config.toml in CWD, env vars override individual fields."""

    model_config = SettingsConfigDict(env_prefix="GT_", env_nested_delimiter="__", extra="ignore")

    paths: PathsCfg = Field(default_factory=PathsCfg)
    embed: EmbedCfg = Field(default_factory=EmbedCfg)
    vector_store: VectorStoreCfg = Field(default_factory=VectorStoreCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    identity: IdentityCfg = Field(default_factory=IdentityCfg)
    summarize: SummarizeCfg = Field(default_factory=SummarizeCfg)
    distill: DistillCfg = Field(default_factory=DistillCfg)
    auth: AuthCfg = Field(default_factory=AuthCfg)
    wiki: WikiCfg = Field(default_factory=WikiCfg)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        candidate = Path(path) if path else Path("config.toml")
        if candidate.exists():
            with candidate.open("rb") as f:
                data = tomllib.load(f)
            return cls(**data)
        return cls()


def load_config(path: Path | str | None = None) -> Config:
    return Config.load(path)
