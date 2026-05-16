# github-twin

[![PyPI version](https://img.shields.io/pypi/v/github-twin.svg)](https://pypi.org/project/github-twin/)
[![Python versions](https://img.shields.io/pypi/pyversions/github-twin.svg)](https://pypi.org/project/github-twin/)
[![CI](https://github.com/ChristopherDavenport/github-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/ChristopherDavenport/github-twin/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A personal RAG over your GitHub history, served to Claude Code (or any MCP
client) as a stdio server. Two scopes, one codebase:

- **User mode** — index your own commits + review comments. Surfaces your
  past code as style examples and your past comments as review hints when
  an agent is writing or reviewing new code.
- **Org mode** — index a whole GitHub org's files-at-HEAD, commits, and PR
  reviews across every member. Queries scope by repo, language, or
  reviewer login.

Retrieval is hybrid (BM25 + vector via RRF), AST-aware via tree-sitter for
python/scala/javascript/typescript/go/rust, and contextually enriched at
embed time with per-chunk headers + optional LLM-generated summaries.

## Install

The fastest path is [uvx](https://docs.astral.sh/uv/) — no virtualenv to
manage, isolated per-tool:

```sh
# One-shot
uvx github-twin --help

# Pinned version
uvx github-twin@0.1.0 --help

# With sentence-transformers for the alt embedder
uvx --with 'github-twin[st]' github-twin --help
```

If you prefer a project-local install:

```sh
uv add github-twin            # or: pip install github-twin
gt --help
```

`gt` and `github-twin` are the same Typer app — use whichever fits your
muscle memory.

## Authenticate

Pick whichever is least friction — github-twin tries them in this order:

1. **OAuth device flow (no `gh` install needed):**
   ```sh
   uvx github-twin auth login        # opens browser, persists token
   uvx github-twin auth status       # show which source is active
   ```
   Token persists in the OS keyring (macOS Keychain / Linux Secret
   Service / Windows Credential Manager) or, when unavailable, a 0600
   file under your data dir.
2. **Existing `gh` CLI**: if you've already run `gh auth login`,
   `gt` picks up the token via `gh auth token` — nothing to do.
3. **`GITHUB_TOKEN` env var**: a classic PAT works too; useful for CI /
   headless / docker. Required scopes: `repo`, `read:org`, `user:email`.

## Wire into Claude Code

The MCP server runs over stdio via `github-twin serve` (or `gt serve`).
Run `uvx github-twin auth login` once on the box that will host the
server, then add an entry to `~/.claude.json` (or your
`mcp_servers.json`):

```json
{
  "mcpServers": {
    "github-twin": {
      "command": "uvx",
      "args": ["github-twin", "serve"],
      "env": {
        "GT_PATHS__DATA_DIR": "/path/to/your/github-twin-data"
      }
    }
  }
}
```

If you'd rather not persist a token and instead supply it inline (CI,
ephemeral container), add `"GITHUB_TOKEN": "ghp_..."` to that `env`
block; it acts as the lowest-priority fallback.

Restart Claude Code; the `find_*`, `predict_review_outcome`,
`summarize_review_patterns`, and `sync` tools will be available.

## Quickstart

Pick a directory to hold the SQLite DB and ingested cache:

```sh
export GT_PATHS__DATA_DIR=~/github-twin-data
uvx github-twin auth login                 # one-time OAuth (or set GITHUB_TOKEN)

# user mode (your own GitHub history)
uvx github-twin init                       # discover identity via /user
uvx github-twin sync                       # ingest + summarize + embed
uvx github-twin serve                      # MCP server over stdio

# org mode (whole org)
GT_PATHS__DATA_DIR=~/twin-http4s \
  uvx github-twin init --kind org --org http4s
GT_PATHS__DATA_DIR=~/twin-http4s uvx github-twin sync
```

`gt sync` is incremental on subsequent runs.

## LLM provider matrix

The retrieval surface (find_*, predict_review_outcome) always runs locally
on the SQLite index — no API call. **LLM calls only happen** in three
places:

- `gt distill` — clusters review comments / commits into rules.
- `gt summarize` — generates per-chunk NL summaries used by the embed-time
  prefix.
- `gt eval reviews` / `eval predictions` — held-out RAG-vs-baseline scoring.

Each picks a backend by precedence **Claude → Gemini → Ollama** (whichever
API key is set), or you can force one explicitly.

| Provider | Env var | What it covers |
|---|---|---|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | Distill / summarize / eval LLM. Best quality. |
| Google (Gemini) | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Distill / summarize / eval LLM. Free tier is generous. |
| Ollama (local) | `OLLAMA_HOST` (default `http://127.0.0.1:11434`) | Distill / summarize / eval LLM. Fully offline. |

### Embeddings are always local

There's no Anthropic embedding API, and we deliberately keep the embedder
backend separate from the LLM backend. Choose one:

- **Default — Ollama** (`nomic-embed-text`, 768-dim, ~50ms/chunk).
  Requires a running Ollama daemon. Zero cost.
- **Alternative — sentence-transformers** (`uv add 'github-twin[st]'`,
  pulls `torch`). Useful when an Ollama daemon isn't available or you
  want a specific HuggingFace model.

So a "cloud-LLM only" setup still needs an embedder process — either
Ollama or the `[st]` extra.

## Required GitHub token scopes

When you `gt init`, the GH client needs:

- `repo` — private repos and PR comments on them
- `user:email` — verified email addresses for the user-mode identity sweep
- `read:org` — org member listing and private org repo discovery

A fine-grained PAT works; classic tokens too.

## Retrieval

Hybrid search by default: BM25 (SQLite FTS5) and vector similarity run in
parallel, then fuse via Reciprocal Rank Fusion (k=60). The vector leg
matches semantic intent; the BM25 leg catches exact identifiers
(`getUserById`, `SQLITE_OPEN_READWRITE`) that vector search routinely
misses. Design reference: [Anthropic — Contextual
Retrieval](https://www.anthropic.com/engineering/contextual-retrieval).

At embed time, each chunk gets a deterministic header prepended:
`# path :: symbol_name (node_kind)`, plus the function's leading
docstring/comment when present, plus an optional LLM-generated summary
(see `gt summarize`). The header lets vector queries land on chunks
whose bodies only contain identifiers (e.g. natural-language queries
against a `VaultSecretEq` function).

BM25 query expansion is on by default (`cfg.retrieval.query_expansion =
"rule"`), with rule-based code-shaped synonyms applied **only** to the
BM25 leg — embeddings already capture synonymy, so expansion never
touches the vector query. Switch to `"ollama"` to add LLM-generated
alternates on top, cached on disk per-token.

`predict_review_outcome` stays on pure vector retrieval because its
inverse-distance vote weighting depends on calibrated L2 distance.

## MCP tools

All retrieval tools accept optional `repo=` and `author_login=` filters.

| Tool | Returns |
|---|---|
| `find_review_comments(diff_hunk, language?, repo?, author_login?, k=5)` | Past review comments on diffs similar to the input. |
| `find_style_examples(query, language?, repo?, author_login?, k=5)` | Past code chunks matching a description. |
| `find_code(query, language?, repo?, path_glob?, node_kind?, k=5)` | Source snippets from files at HEAD (org mode). |
| `find_applicable_rules(query, language?, repo?, author_login?, k=5)` | Distilled code-pattern rules relevant to a coding task. |
| `predict_review_outcome(diff_or_summary, language?, repo?, author_login?, k=20)` | Weighted prediction over nearest past PRs: `{approved, changes_requested, commented}`. |
| `summarize_review_patterns(language?, limit=20)` | Distilled rules from clustered review comments (run `gt distill` first). |
| `sync(since?)` | Incremental ingest + summarize + embed. |

## CLI

```
gt init [--kind user|org|repo] [--org N] [--repo owner/name]
gt repos                                       # list discovered org repos
gt ingest                                      # backfill
gt summarize [--limit N] [--backend ...]       # LLM NL summaries per chunk
gt embed                                       # embed pending chunks
gt sync [--skip-summarize]                     # incremental: ingest → summarize → embed
gt stats                                       # corpus counts
gt distill [--backend ...] [--author ...]      # rule extraction
gt clones prune [--older-than-days N]          # GC the persistent clone cache
gt eval reviews     --since DATE [...]         # held-out RAG-vs-baseline eval
gt eval predictions --since DATE [...]
gt eval search evals/queries/default.yaml      # retrieval-quality dogfood
gt serve                                       # MCP stdio server
```

Use `github-twin <command>` interchangeably with `gt <command>`.

## Pluggable backends

| Surface | Env / config key | Default | Alt |
|---|---|---|---|
| LLM (`cfg.distill.backend`, `cfg.summarize.backend`) | `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / Ollama | `auto` (cloud > local) | force `claude` / `gemini` / `ollama` |
| Embedder (`cfg.embed.backend`) | — | `ollama` (`nomic-embed-text`) | `sentence_transformers` via `[st]` extra |
| Vector store (`cfg.vector_store.backend`) | — | `sqlite-vec` (brute-force KNN) | `faiss` via `[faiss]` extra |
| BM25 query expansion (`cfg.retrieval.query_expansion`) | — | `rule` (deterministic) | `ollama` (LLM, cached) or `off` |

All settings are layered: defaults → `config.toml` in CWD → env vars
prefixed `GT_` (nested via `__`, e.g. `GT_EMBED__BACKEND=sentence_transformers`).

## Held-out evaluation

`gt eval` runs the same prompt with and without retrieval and measures
RAG's accuracy lift on real held-out data:

```sh
# Review-comment voice match (cosine distance to ground truth)
uvx github-twin eval reviews --since 2025-01-01 --limit 100

# Org-mode: scope to one reviewer (and optionally one repo)
uvx github-twin eval reviews     --since 2025-01-01 --author alice --repo http4s/http4s
uvx github-twin eval predictions --since 2025-01-01 --author alice

# Retrieval-quality dogfood (per-tier, per-backend pass rates)
uvx github-twin eval search evals/queries/default.yaml --mode all
```

The harness pre-flights eligibility counts so typo'd `--author` or
`--repo` fail fast without burning LLM calls. The judge embedder
defaults to a different model than the retriever
(sentence-transformers BGE-small with the `[st]` extra installed) to
avoid measuring how well retrieval clusters its own outputs.

## Observability (OpenTelemetry)

Spans for every MCP tool call, every embedder call, and every retrieval
leg, exported via OTLP. **Auto-detected** — nothing fires unless the
environment is configured. Specifically:

1. Install the `[otel]` extra (carries the SDK + HTTP OTLP exporter):

   ```sh
   uvx --with 'github-twin[otel]' github-twin serve
   ```

2. Point at an OTLP HTTP collector via env vars:

   ```sh
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
   export OTEL_SERVICE_NAME=github-twin            # optional
   ```

`OTEL_SDK_DISABLED=true` forces it off even when an endpoint is set.

Without `[otel]` *or* without an endpoint env var, the code paths
still run but every span is a free no-op from `opentelemetry-api`'s
built-in tracer. **stdout is never used** — even with telemetry on —
because MCP speaks JSON over stdin/stdout and a stray console exporter
would corrupt the channel. The OTLP HTTP exporter posts to your
collector; SDK warnings route through Python `logging` (stderr).

Wired into Claude Code:

```json
{
  "mcpServers": {
    "github-twin": {
      "command": "uvx",
      "args": ["--with", "github-twin[otel]", "github-twin", "serve"],
      "env": {
        "GITHUB_TOKEN": "ghp_...",
        "GT_PATHS__DATA_DIR": "/path/to/twin-data",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
        "OTEL_SERVICE_NAME": "github-twin"
      }
    }
  }
}
```

Span names + key attributes you can pivot on:

| Span | Useful attributes |
|---|---|
| `mcp.tool.{find_review_comments,find_style_examples,find_code,find_applicable_rules,predict_review_outcome,summarize_review_patterns,sync}` | `gh_twin.tool.k`, `gh_twin.filter.*`, `gh_twin.result.count` (or `.prediction`/`.confidence` for predict) |
| `embedder.embed` | `gh_twin.embed.input_chars`, `gh_twin.embed.model` |
| `retrieval.hybrid_search` | `gh_twin.retrieval.{chunk_kind,k,expander,hits,top_distance}` |
| `retrieval.vector_search` (predict_review_outcome) | same shape, sans `expander` |

A broken or unreachable collector emits a single `Failed to export
span batch` log line per flush attempt and never propagates into the
tool handler — pinned by `tests/test_observability.py`.

gRPC users: install `opentelemetry-exporter-otlp-proto-grpc` alongside
the `[otel]` extra and the SDK picks it up automatically based on
`OTEL_EXPORTER_OTLP_PROTOCOL=grpc`.

## Storage

Each target gets its own SQLite DB. The default location follows the
[XDG Base Directory spec](https://specifications.freedesktop.org/basedir-spec/latest/):

- `$XDG_DATA_HOME/github-twin/` when `XDG_DATA_HOME` is set
- `~/.local/share/github-twin/` otherwise
- **Backward-compat:** if a `./data` directory exists in the current
  working directory, that wins instead. Pre-XDG installs keep working.

Layout:

```
<data_dir>/
  db.sqlite                  # artifacts + chunks + vectors + FTS5 index
  raw/                       # on-disk cache of raw GitHub responses
  clones/                    # persistent shallow clones (if cache_clones=true)
  query_expansion_cache.sqlite  # only when retrieval.query_expansion=ollama
```

Override per-target via `GT_PATHS__DATA_DIR` — switching from user mode
to org mode is just changing this env var to point at a different
directory.

## Releasing

Versions come from git tags via [hatch-vcs](https://github.com/ofek/hatch-vcs).
Cutting a release is one command:

```sh
git tag v0.2.0    # PEP 440 forms: v0.2.0, v0.2.0a1, v0.2.0rc1, v0.2.0.post1
git push --tags
```

The push to a `v*` tag triggers `.github/workflows/release.yml`, which:

1. Runs pytest + ruff + `uv build` across Python 3.12 / 3.13.
2. Publishes the wheel + sdist to PyPI via [Trusted Publishing
   (OIDC)](https://docs.pypi.org/trusted-publishers/). **No PyPI token
   is stored in repo secrets** — PyPI verifies the GitHub-signed OIDC
   token against the Trusted Publisher you register on the project
   page (workflow filename `release.yml`, environment `pypi`).
3. Creates a GitHub Release with auto-generated notes (PRs since the
   previous tag) and attaches the wheel + sdist. Pre-release tags
   (`a/b/rc`) are flagged so they don't replace "Latest".

First-time setup, once per repo:

1. Push the project to GitHub (any account / org).
2. Register the Trusted Publisher on PyPI:
   - https://pypi.org/manage/account/publishing/
   - Owner: your GitHub user/org, Repository: `github-twin`,
     Workflow: `release.yml`, Environment: `pypi`.
   - (Or use the "Pending Publisher" flow if the project doesn't exist
     on PyPI yet.)
3. On GitHub: Settings → Environments → New environment `pypi`. Add
   yourself as a Required Reviewer for an extra approval step before
   each publish (optional but recommended).

`.github/workflows/ci.yml` runs on every PR and push to `main` — the
release workflow re-runs the same checks before publishing, so a
broken main never produces a release.

## Design notes

- **Embed-time prefix** (`embed.prefix.prefix_chunk`): per-kind header
  spliced before each chunk's text at embed time, never written back to
  `chunk.text`. Bumps `EMBED_TEXT_VERSION` whenever the shape changes
  so the next `gt embed` re-derives vectors.
- **AST chunking** (`process.chunkers`): tree-sitter walks emit chunks
  per declarable unit; falls back to line-window for unsupported
  languages or parse failures.
- **Asymmetric query expansion**
  (`store.query_expansion`): BM25 leg only, vector leg always sees the
  raw embedding — pinned by `test_hybrid_search.py`.

The original design plan lives in
[`getting_started.md`](./getting_started.md) along with the full
walkthrough.

## License

MIT — see [LICENSE](./LICENSE).
