# CLAUDE.md

Project context for agentic readers (Claude Code, etc.). User-facing docs are
in `README.md` and `getting_started.md`.

## What this is

A personal RAG over GitHub history exposed as an MCP server. Two target
kinds, one DB per target:

- **User mode** — one person's commits + review comments. The original P1/P2
  scope: "write like me" + "review like me".
- **Org mode** — a whole GitHub org's files-at-HEAD + commits + reviews across
  all members. Adds an `author_login` axis for filtering and per-reviewer
  evaluation.

Built phase by phase: P1 retrieval → P2 distillation → O-A through O-F
(target abstraction, repo discovery, file-at-HEAD ingest with
process-and-purge clones, org-wide commits + reviews, distill ergonomics,
scale polish) → P3 `predict_review_outcome` → held-out eval (`gt eval`).

## Status

Stable. 283 tests + 7 skipped (the skips are optional sentence-transformers
and faiss deps). User-mode and org-mode are both functional end-to-end.
Retrieval is hybrid (BM25 + vector via RRF) by default. Code chunking is
AST-aware via tree-sitter for python / scala / javascript / typescript
(+ tsx) / go / rust, with a line-window fallback for unsupported
languages or parser failures. Embed text is prefixed with a deterministic
per-chunk header (path / symbol / node-kind / leading docstring) so vector
queries can hit chunks by NL even when the body contains only identifiers —
see `src/github_twin/embed/prefix.py` and `EMBED_TEXT_VERSION` in
`pipeline.py`. Bump the version constant whenever the prefix shape changes;
the next `gt embed` wipes vec_chunk and re-embeds.

```sh
~/.local/bin/uv run pytest -q       # expect 363 passed + 7 skipped
~/.local/bin/uv run ruff check src/ tests/
~/.local/bin/uv run ruff format --check src/ tests/
~/.local/bin/uv run mypy            # strict on src/github_twin/; tests not in scope
~/.local/bin/uv run gt stats        # live user-mode DB sanity
~/.local/bin/uv run gt eval search evals/queries/default.yaml   # retrieval dogfood
```

## Layout

```
src/github_twin/
  cli.py                 # Typer entry point, all gt subcommands
  config.py              # Pydantic-settings; env var prefix GT_, nested __
  pipeline.py            # run_ingest / run_embed; dispatches user vs. org
  target.py              # Target dataclass; discover_user, discover_org
  embed/
    base.py              # Embedder Protocol
    ollama.py            # default, with shrink-fallback for oversized inputs
    sentence_transformers.py  # opt-in via [st] extra
    prefix.py            # prefix_chunk: per-kind embed-time header (contextual retrieval)
  ingest/
    github_client.py     # httpx wrapper, rate-limit aware
    cache.py             # data/raw/ on-disk cache for raw GitHub responses
    commits.py           # ingest_commits (user) + ingest_commits_org
    reviews.py           # ingest_reviews (user) + ingest_reviews_org
    files.py             # org-mode file-at-HEAD walk (O-C)
    repos.py             # org repo enumeration (O-B)
    clone.py             # cloned_repo() context manager + prune_cache (O-F)
  process/
    chunkers.py          # chunk_diff, chunk_file, chunk_pr_summary, chunk_commit_message; AST-aware with line-window fallback
    language.py          # extension -> pygments language tag
    leading_doc.py       # extract docstring or preceding doc-comments for an AST node
    summarize.py         # gt summarize: per-chunk LLM NL summary, persisted in chunk.summary
    grammars/            # tree-sitter grammar registry (one module per language)
      __init__.py        # LanguageGrammar dataclass + grammar_for_language(tag)
      python.py          # function_definition / class_definition / decorated_definition
      scala.py           # object/class/trait_definition + function_definition/declaration
      javascript.py      # function/class_declaration + method_definition
      typescript.py      # adds interface/type_alias/abstract_class_declaration; registers tsx too
      go.py              # function/method_declaration + type_spec (NOT type_declaration)
      rust.py            # function_item / impl_item / struct_item / enum_item / trait_item
  store/
    schema.sql           # canonical schema (idempotent, IF NOT EXISTS); includes chunk_fts + triggers
    db.py                # open_db + pre-schema migrations + chunk_fts backfill
    queries.py           # all SQL access (vector_search, bm25_search, _fts_escape, _fts_match_from_groups)
    query_expansion.py   # QueryExpander Protocol + RuleExpander, OllamaExpander, CompositeExpander
    vector_store.py      # VectorStore Protocol; SqliteVecStore + FaissVectorStore; hybrid_search (RRF)
  distill/
    cluster.py           # HDBSCAN over review_comment embeddings
    synth.py             # RuleSynthesizer Protocol; Claude/Gemini/Ollama
    rules.py             # orchestrator: cluster → synthesize → embed rule
  mcp_server/
    server.py            # FastMCP entry; defines + registers tools (spans per @mcp.tool)
    tools.py             # pure-Python tool impls (testable without MCP; inner spans for embed + retrieval)
  observability.py       # OTel auto-detect: init_otel() + tracer() + set_safe_attributes()
  wiki/
    __init__.py          # re-exports export_wiki, ingest_notes, resolve_vault_root
    export.py            # gt wiki export: render rules/profiles/repos/index, write-on-diff
    render.py            # per-entity markdown renderers (frontmatter + body)
    ingest_notes.py      # scratch/*.md round-trip; SHA-256-keyed kind='note' artifacts
    scan.py              # frontmatter parser + generated-file walker (prune seam)
    slug.py              # stable filenames: rule slug + 8-char sha1, owner__name, etc.
  eval/
    holdout.py           # iter_held_out_* + count_eligible
    llm.py               # TextLLM Protocol; Claude/Gemini/Ollama
    runner.py            # evaluate_reviews + evaluate_predictions
    search_evals.py      # tiered YAML retrieval-quality runner (gt eval search)
    report.py            # Rich-table rendering (review, predict, search)
tests/                   # pytest; one file per module concern
evals/queries/           # checked-in YAML fixtures for gt eval search
data/                    # gitignored; one subdir per target if you switch
```

## Pluggable backends (all driven by config)

| Surface | Config key | Default | Opt-in |
|---|---|---|---|
| Embedder | `cfg.embed.backend` | `ollama` | `sentence_transformers` (`[st]` extra) or `gemini` (remote — uses existing `google-genai` dep; sends chunk text off-box to Google) |
| Vector store | `cfg.vector_store.backend` | `sqlite-vec` | `faiss` (`[faiss]` extra) |
| BM25 query expansion | `cfg.retrieval.query_expansion` | `rule` | `ollama` (small local model) or `off` |
| Chunk summary LLM | `cfg.summarize.backend` | `auto` (Claude > Gemini > Ollama) | force any of the three |
| Distill | `cfg.distill.backend` | `auto` (Claude > Gemini > Ollama) | force any of the three |

Switching DBs between targets is done via `GT_PATHS__DATA_DIR=...`; one DB
per `target` row by construction.

The `gemini` embedder is the only backend that sends chunk text to a
remote API. Pick it deliberately — it exists for users who have Gemini
auth but neither Ollama nor the `[st]` extra available. Default model
is `gemini-embedding-001` at 3072 dims; `cfg.embed.dim` 1536 or 768 are
also supported via the SDK's `output_dimensionality`.

Gemini auth has two paths, all four Gemini surfaces (embed, distill,
summarize, eval) share them via `gemini_client.make_gemini_client`:
1. **API key** — `GEMINI_API_KEY` or `GOOGLE_API_KEY`. Wins when both
   paths are configured.
2. **Vertex AI + ADC** — set `GT_GEMINI_PROJECT` (and optionally
   `GT_GEMINI_LOCATION`, default `us-central1`) and bootstrap once
   with `gcloud auth application-default login`. The credential lives
   on disk; no key in your shell. Requires `aiplatform.googleapis.com`
   enabled on the project, and billing applies even for "free"
   Gemini models (AI Studio free tier does NOT extend to Vertex).

`has_gemini_auth()` in the same module is the predicate that lets the
`auto` selector in `make_synthesizer` / `make_text_llm` consider Gemini
eligible when only the project env is set. Env vars are read directly
(not via pydantic-settings), so single underscores are intentional.

The embedder backend is a per-DB commitment (`sqlite-vec` bakes the
dim into the virtual table at first creation), so `gt init` accepts
`--embed-backend` / `--embed-model` / `--embed-dim` flags that stamp
the chosen values into `config.toml` *before* the DB is opened.
Helpers: `_resolve_embed_defaults` and `_persist_embed_config` in
`cli.py`. Re-running with the same values is idempotent; re-running
with different values against an existing `[embed]` block raises
`typer.BadParameter` to refuse a silent overwrite.

## Embed-time prefix (contextual retrieval)

`pipeline._flush` doesn't embed `chunk.text` raw — it routes through
`embed.prefix.prefix_chunk(c)`, which prepends a deterministic header:

| chunk.kind | header |
|---|---|
| `code`, `code_rule`, `file` (AST) | `# {path} :: {symbol_name} ({node_kind})` + optional `# {summary}` + optional `# {leading_doc}` + blank line |
| `code`, `file` (line-window fallback) | `# {path}` + optional `# {summary}` + blank line |
| `commit_message` | `# commit {repo}@{sha[:7]}` + optional `# {summary}` + blank line |
| `review_comment` | `# review on {repo} #{pr_number}` (+ ` ({path})` when path is present) + blank line |
| `pr_summary`, `rule` | unchanged — text already structured |

`{summary}` is the LLM-generated NL description written by
`gt summarize` into `chunk.summary` (TEXT column, NULL when not yet
run). When non-NULL it's spliced in as a header line. The summary
bridges NL queries to identifier-only code chunks — without it,
"Eq instance for a wrapper case class" can't reach a chunk whose
only NL surface is the symbol `VaultSecretEq`.

Headers are NOT persisted in `chunk.text`; they exist only at embed time.
That keeps BM25 (external-content FTS5 over raw `chunk.text`) unchanged
and means re-running embed re-derives them deterministically.

`EMBED_TEXT_VERSION` in `pipeline.py` is the seam: bump it whenever the
prefix shape changes. `run_embed` reads the stored version from
`sync_cursor` (resource `embed_text_version`); on mismatch it wipes
`vec_chunk`, nulls `chunk.embed_model`, and re-embeds the whole corpus.
Brand-new DBs (no vectors yet) skip the wipe and just stamp the current
version after their first embed pass. Current version: **4**.

## Summarize (LLM chunk summaries)

`gt summarize [--limit N] [--kind code] [--backend ollama] [--model M]
[--rebuild]` populates `chunk.summary`. Backends use the same
`eval.llm.TextLLM` seam as `gt eval reviews` — Claude / Gemini / Ollama.
Default kinds = `code, file, code_rule, commit_message`; NL-shaped kinds
(`review_comment`, `pr_summary`, `rule`) opt out by design — summarizing
them just compresses with loss.

`gt sync` runs summarize before embed by default; pass `--skip-summarize`
to skip. After summarize completes, the next `gt embed` will detect the
`EMBED_TEXT_VERSION` bump and re-embed the whole corpus so vectors
include the new summary lines.

Pragmatic local model choice: **`llama3.2` (3B)** is the right default
for code summaries on a single laptop — ~0.85 s/chunk and high signal.
qwen3:0.6b is faster per token but emits internal "thinking" tokens
that consume the budget and ~35% of summaries come back empty; not
worth it on this corpus.

## Observability (OpenTelemetry)

`observability.init_otel()` is called once from the CLI main callback
(`cli.py:main`). Auto-detection from standard OTel env vars:

- On when `OTEL_EXPORTER_OTLP_ENDPOINT` or
  `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set AND the `[otel]` extra is
  installed. Wires `BatchSpanProcessor` to `OTLPSpanExporter` (HTTP).
- Off otherwise. Spans go to `opentelemetry-api`'s built-in noop tracer
  — every call site still executes, just at ~zero cost.

`OTEL_SDK_DISABLED=true` is the standard kill switch and overrides
configured endpoints.

**stdio contract:** the MCP server speaks JSON over stdin/stdout. We
deliberately never register a ConsoleSpanExporter. The OTLP HTTP
exporter is the only sink. SDK warnings (e.g. unreachable collector)
go through Python `logging`, which defaults to stderr. Pinned by
`tests/test_observability.py:test_mcp_tool_call_does_not_write_to_stdout`.

Span layout:

- `mcp.tool.{name}` — one per MCP tool invocation, attributes
  `gh_twin.tool.k`, `gh_twin.filter.*`, `gh_twin.result.count`.
- Inside `tools.py`: `embedder.embed` and `retrieval.hybrid_search`
  (or `retrieval.vector_search` for `predict_review_outcome`'s
  bypass path). The latter sets `gh_twin.retrieval.hits`,
  `gh_twin.retrieval.top_distance`, `gh_twin.retrieval.chunk_kind`,
  `gh_twin.retrieval.expander`.
- `set_safe_attributes(span, **attrs)` is the canonical writer:
  drops `None`, coerces non-primitive types, accepts lists/tuples.

Failure isolation: a misconfigured endpoint, missing SDK, or broken
TracerProvider initialization all degrade to `is_active() == False`
and never raise into a tool handler.

## Storage layout + secret hygiene

`resolve_data_dir()` in `config.py` is the single source of truth for
the data directory: `GT_PATHS__DATA_DIR` env var > `$XDG_DATA_HOME/github-twin`
> `~/.local/share/github-twin`. Pure function — it does NOT read cwd, so
two `gt init` calls in the same shell layer cleanly into one DB at the
resolved path. There is no `./data`-in-cwd auto-detect.

Everything per-data-dir lives under that root:

- `<data_dir>/db.sqlite` (the SQLite corpus)
- `<data_dir>/config.toml` (loaded by `Config.load()` when no `--config`
  is given; written by `gt init --embed-backend ...` via
  `_persist_embed_config`)
- `<data_dir>/raw/` (raw GitHub response cache via `RawCache`)
- `<data_dir>/clones/` (org-mode persistent clones, when
  `IngestCfg.clones_dir` is unset — `resolved_clones_dir(cfg)` is the
  helper)
- `<data_dir>/wiki/` (`gt wiki export` default output)
- `<data_dir>/auth/token.json` (OAuth file fallback)

`cli.py:_warn_legacy_cwd_paths` fires a one-shot WARN on every CLI
invocation if a stray `./config.toml` or `./data/db.sqlite` is sitting
in the cwd but the resolved data_dir is elsewhere — informational only,
no auto-move.

Long-lived connections use `db_session()` (in `store/db.py`) — a
contextmanager that pairs `open_db()` with a guaranteed `close()` and
swallows close-time errors so teardown failures can't mask successful
runs. The MCP server's main loop is wrapped in it. Short-lived CLI
commands still call `open_db()` directly; process exit handles cleanup.

Secret hygiene is enforced in two layers (`_logging.py`):

1. **`cap_noisy_loggers()`** caps third-party loggers known to dump
   request/response data at DEBUG (`httpx`, `httpcore`, `anthropic`,
   `google*`, `openai`, `urllib3`) at WARNING — so `--verbose` doesn't
   surface auth headers from inside SDK internals.
2. **`SecretRedactingFilter`** on the root logger scrubs
   `Bearer <tok>` and `x-api-key: <val>` patterns from outbound log
   records, AND replaces the literal value of any
   `GITHUB_TOKEN` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` /
   `GOOGLE_API_KEY` env var that's ≥12 chars. Mutates `record.msg`
   and clears `record.args` so handler `%` formatting can't
   re-introduce the secret. Idempotent install. Persisted OAuth
   tokens loaded via `auth_storage.load_token()` are registered
   into the same filter at first-use time via
   `register_secret_value()` so they get the same scrubbing.

Both are wired in `cli.py:_setup_logging`, so every CLI invocation
(and every MCP server run) gets them. `GT_GEMINI_PROJECT` is
deliberately NOT in `_SECRET_ENV_VARS` — project IDs are visible
diagnostic context, not credentials; ADC tokens live on disk at
`~/.config/gcloud/application_default_credentials.json` and never
appear in env. Pinned by `test_logging_redaction.py:test_gemini_project_env_not_treated_as_secret`.

## GitHub auth

Token resolution (`ingest/github_client.py:_resolve_token`) tries three
sources in order:

1. **Persisted OAuth device-flow token** written by `gt auth login`.
   Stored via `ingest/auth_storage.py` with keyring-first / 0600-file
   fallback. Keyring service is `github-twin`, user `oauth`; file path
   is `<data_dir>/auth/token.json`.
2. **`gh auth token`** subprocess output, when `gh` is on PATH.
3. **`GITHUB_TOKEN`** env var.

`gt auth login` runs the GitHub Device Authorization Grant against the
`github-twin` OAuth App (Client ID `Ov23liAUxXgwgIJp6jqZ`, public).
Override via `GT_AUTH__CLIENT_ID` for a downstream fork or test
fixture. The device-flow client lives in `ingest/oauth.py` and accepts
an injectable `sleep` callable so tests can run without wall-clock
waits.

`keyring>=24` is a hard dep, not an extra — the storage layer
imports it unconditionally and silently falls back to file mode on
boxes without a usable backend (typical on headless WSL/SSH/docker
where no D-Bus / no Secret Service is available).

## Packaging & release

Wheel + sdist build with `uv build`. Two console scripts in
`pyproject.toml`: `gt` and `github-twin` — both bind to
`github_twin.cli:app`. The second name is what `uvx github-twin ...`
picks up automatically (uvx looks for a script matching the package
name); `gt` stays for daily shell use.

Distributable via:
- `uvx github-twin <cmd>` — zero-install, isolated env, **the recommended
  path for Claude Code MCP integration** (`~/.claude.json` points at
  `uvx github-twin serve`).
- `uv add github-twin` / `pip install github-twin` — project-local.
- `uv build && uv publish` to release to PyPI (or any index via
  `--publish-url`).

Embeddings never go to a remote API. The LLM seam (`distill`,
`summarize`, `eval`) can run cloud (Claude / Gemini) or local (Ollama).
"Fully cloud" still needs a local embedder — Ollama daemon or the
`[st]` extra (sentence-transformers + torch).

## Schema invariants

- `chunk_fts` is an external-content FTS5 index over `chunk.text`
  (`tokenize="porter unicode61 remove_diacritics 2 tokenchars '_'"`).
  Triggers `chunk_ai/au/ad` in `schema.sql` keep it synced with `chunk`. Do
  NOT `DELETE FROM chunk_fts` directly — external-content tables reject it;
  delete from `chunk` and the trigger handles cleanup.
  `delete_chunks_for_artifact` depends on this trigger.
- `chunk_fts_docsize` is the load-bearing probe for "is the index
  populated?" — a plain `SELECT 1 FROM chunk_fts` always returns rows when
  `chunk` has rows, regardless of index state.
- `artifact.kind` ∈ `{commit, pr, review_comment, issue_comment, file, rule, note}`
- `chunk.kind` ∈ `{code, review_comment, commit_message, file, pr_summary, rule, code_rule, note}`
  - `note` rides under `artifact.kind='note'` and originates from
    `<vault>/scratch/*.md` (the wiki round-trip). `external_id` is the
    SHA-256 of file contents so editing a note swaps the artifact
    cleanly; `source_url` is the local `file://` path. Notes flow
    through hybrid_search like any other chunk — no new MCP tool is
    needed; filter on chunk.kind='note' if you want to scope to them.
  - `rule` and `code_rule` both ride under `artifact.kind='rule'`; the
    chunk-level split lets retrieval filter cheaply. `rule` is
    review-comment-derived (from `gt distill`, default `--kind review`);
    `code_rule` is commit-diff-derived (`gt distill --kind code`).
  - `artifact.meta.rule_source ∈ {'review_comment', 'code'}` mirrors this
    at the artifact level for diagnostics.
- `target` is a singleton (`CHECK (id = 1)`), with `kind ∈ {user, org, repo}`
  - `repo` kind is a single-repo scope (`name='owner/name'`); the pipeline
    treats it as org-mode-with-one-repo so `ingest_files` /
    `ingest_commits_org` / `ingest_reviews_org` are reused unchanged.
- `repo` table populated in org mode and repo mode (one row in repo mode).
  `repo.archived` and `repo.visibility ∈ {public, private, internal, NULL}`
  are stamped from the GitHub `/orgs/{org}/repos` response. `gt sync`
  re-runs `enumerate_org_repos` (always with `include_archived=True`)
  before ingest so a repo that flipped to archived after `gt init` gets
  its row refreshed — downstream `q.list_repos(include_archived=False)`
  (the default at every ingest read site) then naturally excludes it.
  Opt back in with `--include-archived` on `gt init` / `gt sync` or
  `ingest.include_archived = true` in `config.toml`. "Internal-archived"
  repos (`visibility=internal` AND `archived=true`) are caught by the
  archived filter — no separate visibility flag.
- `artifact.decision` is set only in user mode; org-mode equivalent is
  `meta.reviewer_decisions = [{login, state, submitted_at}]`
- `artifact.author_login` is populated by org-mode ingest; user-mode leaves
  it NULL (the corpus is one person by construction)
- `chunk.language` is per-chunk; queries filter on this column, never on
  `artifact.language`
- `chunk.node_kind` and `chunk.symbol_name` are populated by the AST
  chunker for languages with a registered grammar; NULL on line-window
  fallback chunks and on non-code chunk kinds (`commit_message`,
  `review_comment`, `pr_summary`, `rule`, `code_rule`). `node_kind` holds
  the raw tree-sitter node type (`function_definition`,
  `class_declaration`, `impl_item`, ...); `symbol_name` is the
  human-readable identifier extracted via the grammar's `symbol_name`
  callback. `vector_search` / `bm25_search` / `VectorSearchFilters` all
  accept an optional `node_kind` filter; `find_code` surfaces it as an
  MCP parameter.

## How to run things

```sh
# tests + lint
~/.local/bin/uv run pytest -q
~/.local/bin/uv run ruff check src/ tests/

# live user-mode DB
~/.local/bin/uv run gt stats
~/.local/bin/uv run gt sync                    # incremental ingest + summarize + embed + wiki export
~/.local/bin/uv run gt sync --skip-wiki        # skip the scratch-note ingest + vault export
~/.local/bin/uv run gt sync --include-archived # let archived repos through this run
~/.local/bin/uv run gt summarize               # standalone (cfg.summarize backend)
~/.local/bin/uv run gt wiki export             # one-shot: re-render the markdown vault from the DB

# org-mode (fresh data dir)
GT_PATHS__DATA_DIR=./data-org uv run gt init --kind org --org <name>
# --include-archived: keep archived repos (also catches internal-archived).
GT_PATHS__DATA_DIR=./data-org uv run gt init --kind org --org <name> --include-archived
GT_PATHS__DATA_DIR=./data-org uv run gt ingest
GT_PATHS__DATA_DIR=./data-org uv run gt embed

# eval
uv run gt eval reviews     --since 2025-01-01 [--author X] [--repo Y] [--limit N]
uv run gt eval predictions --since 2025-01-01 --author X  # org-mode needs --author
uv run gt eval search      evals/queries/default.yaml      # retrieval-quality dogfood
```

## Working on this codebase

- Use the existing chunker / embedder / store seams. New chunk kinds are
  fine; new artifact kinds need a schema decision (write a migration in
  `db._run_pre_schema_migrations` and update `schema.sql`).
- `hybrid_search` (in `store/vector_store.py`) is the canonical retrieval
  call: it RRF-fuses `store.search` (vector) with `q.bm25_search` (keyword)
  and returns `SearchHit`s with `distance = 1 - rrf_score`. All MCP tools
  except `predict_review_outcome` go through it. `predict_review_outcome`
  intentionally uses `store.search` directly because its inverse-distance
  vote weighting needs calibrated L2 distance — don't "fix" the
  inconsistency.
- `q.bm25_search` mirrors `q.vector_search`'s candidate-set pattern for
  filter parity. Without an expander, user text passes through
  `q._fts_escape` (defang FTS5 metacharacters, implicit-AND of bare
  tokens). With an expander, each query token becomes an OR-group via
  `q._fts_match_from_groups` and groups are joined with explicit `AND`
  (implicit-AND breaks when a parenthesized OR-group is in the mix).
- Asymmetric query expansion: `cfg.retrieval.query_expansion` controls
  the BM25-leg expander; the vector leg is never expanded (amanmcp
  research measured -15pp dense regression). `expander_from_config(cfg)`
  builds it; the MCP server constructs once at startup and passes
  through every tool. Backends: `rule` (deterministic synonyms + case +
  camelCase splits, zero deps), `ollama` (Rule + small local LLM with
  per-token SQLite cache at `data/query_expansion_cache.sqlite`), `off`.
  The asymmetry contract is pinned by
  `test_hybrid_search.py:test_hybrid_passes_expander_only_to_bm25_leg`.
- `delete_chunks_for_artifact` is the idempotency seam — every ingest writer
  calls it before re-inserting chunks under an upserted artifact.
- **Prompt-management surface** (`tools.py:house_rules`,
  `tools.py:developer_profile`, the `scope` parameter): these are
  the "static memory block" tools, distinct from the per-query
  retrieval tools. Both accept `language` / `repo` / `author_login` /
  `scope` filters — `language` is the load-bearing one for
  `house_rules` since an unscoped block mixes idioms from every
  language in the corpus inside a single-language session.
  `house_rules` is a pure rendering of `q.list_rules` output as
  Markdown; the SQL-side filters live on `q.list_rules`.
  `developer_profile` synthesizes a Markdown profile via
  `distill.profile.synthesize_profile` and caches in
  `developer_profile_cache`. Cache key is composite via
  `_profile_cache_key(author, language, repo)` so different scopings
  don't clobber each other; bare-author calls still key under the
  unscoped login (preserves pre-scope cache entries). Invalidation
  is `sample_hash` over recent chunk_ids inside a given key.
  `recent_review_comments` filters on `c.language` (the chunk's
  language, populated from the diff-hunk path). `scope` is sugar —
  see `_resolve_scope` in `tools.py`; resolves against
  `load_target(conn)` so user-mode `scope="personal"` fills
  `author_login`, repo-mode `scope="project"` fills `repo`. Explicit
  kwargs always win.
- **`gt init-claude-md`** scaffolds a `CLAUDE.md` from
  `templates/claude_md.py:CLAUDE_MD_TEMPLATE` with the current target
  name + MCP server name substituted in. Template lives as a Python
  string constant (not a `.md` file) so hatchling needs no
  package-data plumbing.
- **Wiki vault** (`gt wiki export`, `wiki/` module): materializes the
  corpus as an Obsidian-compatible markdown vault under
  `cfg.paths.data_dir/wiki/` (override with `--out` or `cfg.wiki.out`).
  Three entity sections — `rules/{lang}/{slug}.md`,
  `profiles/{login}.md`, `repos/{owner}__{name}.md` — plus an
  `index.md` and per-section `_index.md`. Cross-linked with Obsidian
  `[[wikilinks]]` and GitHub permalinks. Reuses `q.list_rules`,
  `q.repo_overview` (new), and `tools.developer_profile` (which caches
  through `developer_profile_cache`).
- **Wiki round-trip** (`<vault>/scratch/`, `wiki/ingest_notes.py`):
  any `.md` dropped into the scratch folder is ingested on the next
  `gt sync` as a `kind='note'` artifact keyed by SHA-256 of file
  contents. Auto-generated files everywhere in the vault carry
  `generated: true` in YAML frontmatter; the scratch ingester walks
  only `scratch/` and double-checks the marker so re-exports can never
  loop on themselves. Notes flow through hybrid_search like any other
  chunk. Bumping the `note` prefix shape in `embed/prefix.py` requires
  bumping `EMBED_TEXT_VERSION` (currently **4**, was 3 pre-notes).
- `gt distill` produces `kind='rule'` artifacts keyed by member-chunk-id hash,
  so re-running is idempotent and safe to switch backends mid-stream.
- Tests use a `FakeEmbedder` (4-dim, pattern-keyed) for deterministic vector
  search; never call out to real Ollama in unit tests.
- `gt eval search <yaml>` runs the retrieval-quality dogfood suite (tiered
  YAML query → per-backend pass rates). Tier 1 missing 100% exits non-zero
  so CI can gate. The runner lives at `src/github_twin/eval/search_evals.py`;
  default fixture at `evals/queries/default.yaml`. Per-backend split is the
  load-bearing column — a regressing leg can hide behind a healthy
  hybrid number; the table makes the split visible.
- Embed-time prefix: `src/github_twin/embed/prefix.py:prefix_chunk(c)` runs
  inside `pipeline._flush` and prepends a deterministic per-kind header
  (path / symbol_name / node_kind / leading docstring for code chunks,
  `# commit repo@sha` / `# review on repo#PR (path)` for the others). Headers
  are derived at embed time only — they are NOT stored in `chunk.text`, so
  BM25 (which sees raw text via the external-content FTS5 index) is
  unchanged. Any edit to the prefix shape must bump `EMBED_TEXT_VERSION` in
  `pipeline.py`; on next `gt embed` the version-cursor mismatch triggers a
  vec_chunk wipe + full re-embed (see `_embed_text_version_needs_bump`).
  Stored in `sync_cursor` under resource `embed_text_version`.
- Adding a new language grammar = one file under `process/grammars/<lang>.py`
  declaring `chunk_node_kinds`, a `symbol_name` callback, and (optionally)
  a `leading_doc` callback, plus an import line in `grammars/__init__.py`.
  Most languages just point `leading_doc=extract_preceding_comments` from
  `process/leading_doc.py`; Python is special-cased to also pick up the
  inside-body docstring. `language` must match what
  `language_for_path` returns; `parser_name` is the `tree-sitter-language-pack`
  key. The chunker's two paths read from the same registry:
  - `chunk_file` (file-at-HEAD, org mode): emits every matching AST node in
    the tree, so a class and its methods both surface as separate chunks.
    Multi-granularity by design.
  - `chunk_diff` (commits): emits the *deepest* chunkable ancestor of each
    significantly-added line (non-blank `+` lines). A method change inside a
    class emits the method, not the class; a class-header change emits the
    class. Implemented as a post-order claim-propagation walk in
    `_chunk_hunk_ast`.
  Both paths fall back to `_chunk_file_line_windows` / `_added_blocks` when
  no grammar applies, parser fails, or AST walk yields no chunks for the
  region. That guarantees no language regresses to zero coverage.

## Pointers

- Design plan: `/home/chris/.claude/plans/using-github-history-build-robust-lantern.md`
- Auto-memory: `/home/chris/.claude/projects/-home-chris-coding-github-twin/memory/`
- End-user docs: `README.md`, `getting_started.md`
