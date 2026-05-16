# Putting it together

## What it is

A RAG is a pipeline first and a model second. The seven previous docs
each describe one stage in isolation; this one connects them.

The data flows in one direction during indexing:

```
  GitHub        local git walk   (chunk_diff, chunk_file,
  REST API  ─┐  + REST API        chunk_pr_summary,
             │  ──────────────┐   chunk_commit_message;
             ▼                ▼   AST-aware via tree-sitter)
          ┌─────────┐   ┌──────────┐
          │ ingest  ├──▶│ process  ├──▶ chunk rows
          └─────────┘   └──────────┘    (text + node_kind +
                                         symbol_name + ...)
                                               │
                                               ▼
                                        ┌───────────────┐
                                        │  summarize    │  optional: one-sentence
                                        │  (LLM)        │  LLM summary per chunk
                                        └───────┬───────┘
                                                ▼
                                        ┌───────────────┐
                                        │ prefix_chunk  │  splice header +
                                        │ (deterministic│  doc + summary onto
                                        │  + summary)   │  every chunk
                                        └───────┬───────┘
                                                ▼
                                        ┌───────────────┐
                                        │ embed         │  Ollama or
                                        │ (Embedder)    │  sentence-transformers
                                        └───────┬───────┘
                                                ▼
                                ┌─────────────────────────┐
                                │ store: SQLite           │
                                │  - chunk (text + meta)  │
                                │  - chunk_fts (BM25)     │
                                │  - vec_chunk (vectors)  │
                                └─────────────────────────┘
```

…and in the other direction at query time:

```
  agent (Claude Code, etc.)
       │   tool call: find_review_comments(diff_hunk, language="scala")
       ▼
  ┌──────────────────────────────────────────┐
  │ MCP server (gt serve, stdio)             │
  │                                          │
  │  embed(diff_hunk) ─┐                     │
  │                    ▼                     │
  │  hybrid_search ─┬─▶ BM25 leg (FTS5)      │   ◀── language /
  │                 │   (+ asymmetric        │       repo /
  │                 │     query expansion)   │       author filter
  │                 │                        │
  │                 └─▶ vector leg (sqlite-  │
  │                     vec, k-NN under      │
  │                     same SQL filter)     │
  │                                          │
  │  RRF fusion (k=60) → top-K hits          │
  │  emit OTel spans if configured           │
  └──────────────────────────────────────────┘
       │
       ▼  five real past review comments
  agent uses them as context for its actual job
```

The pipeline is the point: a "RAG" without ingest is just a chat model; a
RAG without retrieval is just a database; a RAG without evaluation is just
hope.

## How github-twin composes the stages

- **Pipeline dispatcher**: `src/github_twin/pipeline.py:run_ingest` reads
  the singleton `target` row and dispatches to either the user-mode path
  (`ingest_commits` + `ingest_reviews`) or the org-mode path
  (`ingest_files` → `ingest_commits_org` → `ingest_reviews_org`).
- **Summarize step**: `pipeline.py:run_summarize` walks
  `pending_summary_chunks` (chunks where `summary IS NULL` for the
  code-shaped kinds) and writes one-sentence LLM summaries to
  `chunk.summary`. Idempotent; opt-out via `gt sync --skip-summarize`.
- **Embed step**: `pipeline.py:run_embed` is kind-agnostic. It selects
  all chunks that don't yet have an embedding, runs each through
  `prefix_chunk` to splice in the per-kind header + leading docstring +
  summary, batches them through the configured `Embedder`, and writes
  vectors to `vec_chunk`. Detects `EMBED_TEXT_VERSION` bumps and
  re-embeds the whole corpus when the prefix shape changes — see
  `_embed_text_version_needs_bump`.
- **CLI**: `src/github_twin/cli.py` is the user-facing surface. `gt
  ingest` calls `run_ingest`, `gt summarize` calls `run_summarize`,
  `gt embed` calls `run_embed`, `gt sync` chains all three. `gt distill`
  runs the clustering + synthesis pipeline. `gt eval reviews|predictions`
  runs the held-out RAG-vs-baseline comparison; `gt eval search` runs
  the retrieval-quality dogfood harness. `gt serve` starts the MCP
  server. Both `gt` and `github-twin` resolve to the same Typer app.
- **MCP server**: `src/github_twin/mcp_server/server.py` opens the DB via
  the `db_session()` context manager (guaranteed clean close on exit),
  builds the `VectorStore` + `QueryExpander`, calls `init_otel()` (no-op
  unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set), and registers the tool
  implementations from `mcp_server/tools.py`. The server speaks MCP over
  stdio; the agent on the other side calls tools and gets back JSON.
- **The tools** are the user-visible RAG, split into two flavors:
  - **Dynamic retrieval** (`find_review_comments`, `find_style_examples`,
    `find_code`, `find_applicable_rules`, `predict_review_outcome`,
    `summarize_review_patterns`) — call once per query, each a thin
    function that embeds its query, calls `hybrid_search` (or
    `store.search` directly for `predict_review_outcome`), and returns
    a list of snippets. The retrieval tools accept a `scope` parameter
    (`"personal"` / `"project"` / `"all"`) that's sugar over the
    existing `repo=` / `author_login=` filters.
  - **Static memory blocks** (`house_rules`, `developer_profile`) —
    call once at session start, paste the returned Markdown into the
    agent's working context. `house_rules` renders the distilled
    `q.list_rules` output as one Markdown document;
    `developer_profile` synthesizes a 2–3-paragraph voice description
    via the same `eval.llm.TextLLM` dispatch used by `gt summarize` /
    `gt distill`, and caches the result in
    `developer_profile_cache` (invalidated when the sample of recent
    comments changes).

  None of the retrieval tools call an LLM directly. `developer_profile`
  is the only tool that does — and only when its cache is cold or
  invalidated. The *agent* is the language model in this architecture.

- **Scaffolding**: `gt init-claude-md` writes a `CLAUDE.md` template
  into the cwd that tells Claude Code *when* to call each tool. The
  template lives as a Python string constant in
  `src/github_twin/templates/claude_md.py` so it ships with the wheel
  without any package-data plumbing.

The CLAUDE.md file at the project root has a deeper module-by-module
layout and the schema invariants (`artifact.decision` is user-mode-only;
org-mode equivalent is `meta.reviewer_decisions`; `chunk.language` is
per-chunk and queries filter on it). Read it once you've internalized
this end-to-end picture.

## Further reading

- **Model Context Protocol — Specification** —
  [modelcontextprotocol.io](https://modelcontextprotocol.io). Skim the
  "Architecture" and "Tools" sections. Understanding how an MCP host
  enumerates and calls tools makes the `gt serve` surface obvious.
- **Anthropic — *Building tools for Claude*** —
  [docs.anthropic.com/en/docs/build-with-claude/tool-use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use).
  The general pattern of "model calls tool, tool returns text, model
  continues." MCP is a transport for this pattern.
- **Anthropic — *Contextual Retrieval*** —
  [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval).
  The recipe github-twin's retrieval pipeline implements end-to-end:
  BM25 + vector + RRF fusion AND per-chunk contextual prefixes
  embedded into the vector. Worth reading even after you've grasped
  github-twin's implementation — the numbers (e.g. "67% reduction in
  failure rate" with all three legs combined) come from there.
- **LlamaIndex documentation** — search for "LlamaIndex docs." Another
  Python RAG framework that wraps the same stages github-twin builds by
  hand. Worth a flip-through to see which seams the framework chose to
  expose; instructive even if you never adopt it.
