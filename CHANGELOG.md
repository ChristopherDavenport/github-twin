# Changelog

All notable changes to `github-twin` are recorded here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The canonical version source is the most recent `v*` git tag — see
`pyproject.toml`'s `hatch-vcs` configuration.

## [Unreleased]

## [0.0.4] — 2026-05-16

### Added
- Vertex AI / Application Default Credentials fallback path for Gemini
  auth. `GT_GEMINI_PROJECT` env var bootstraps via `gcloud auth
  application-default login`, alongside the existing `GEMINI_API_KEY` /
  `GOOGLE_API_KEY` path. All four Gemini surfaces (embed, distill,
  summarize, eval) share the same resolver in
  `gemini_client.make_gemini_client`.

### Documentation
- Getting-started guide covers the Gemini ADC path.

## [0.0.3] — 2026-05-16

### Added
- **Multi-target single DB.** One `data/` directory can now hold N
  targets — user, org, or repo scopes — instead of one DB per target.
- `gt targets` CLI surface for listing and selecting targets.
- `--target` flag on retrieval commands.
- Additive `gt init` so subsequent targets append rather than wipe.
- MCP tools accept `target=` for cross-target retrieval.

## [0.0.2] — 2026-05-16

### Added
- **Gemini embedder backend** alongside the existing Ollama and
  sentence-transformers options. Default model
  `gemini-embedding-001` at 3072 dims; 1536 and 768 supported via
  `output_dimensionality`.
- `gt init` flags `--embed-backend` / `--embed-model` / `--embed-dim`
  to stamp embed choice before the DB is opened. Idempotent on
  re-run with the same values; refuses silent overwrite on mismatch.
- Claude Code plugin manifest (`.claude-plugin/plugin.json`) with
  release-time version sync so the marketplace stays in lockstep with
  the published PyPI version.

### CI
- Bumped GitHub Actions versions to latest majors.

## [0.0.1] — 2026-05-16

Initial public release.

### Added
- Personal RAG over GitHub history (commits, code, review comments)
  served as an MCP server (`gt serve` / `uvx github-twin serve`).
- User mode and org mode ingest pipelines with process-and-purge
  shallow clones.
- AST-aware code chunking via tree-sitter (Python, Scala, JavaScript,
  TypeScript, Go, Rust) with line-window fallback.
- Hybrid retrieval (BM25 + vector via RRF) with optional Rule /
  Ollama BM25 query expansion.
- LLM chunk summaries (`gt summarize`) spliced into the embed-time
  prefix for contextual retrieval.
- Distillation of review comments into actionable rules
  (`gt distill`).
- Eval surfaces: `gt eval reviews`, `gt eval predictions`,
  `gt eval search`.
- Prompt-management tools: `house_rules`, `developer_profile`,
  `gt init-claude-md`.
- Pluggable backends for embedder (Ollama / ST / Gemini), vector
  store (sqlite-vec / FAISS), and LLM (Claude / Gemini / Ollama).
- OpenTelemetry tracing auto-detected from standard `OTEL_*` env
  vars.
- PyPI distribution as `github-twin` + console scripts `gt` and
  `github-twin`.
- Release pipeline via tag-cut → CI test gate → PyPI Trusted
  Publishing + Sigstore SLSA attestations + auto-drafted GitHub
  Release.

[Unreleased]: https://github.com/ChristopherDavenport/github-twin/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/ChristopherDavenport/github-twin/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/ChristopherDavenport/github-twin/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/ChristopherDavenport/github-twin/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ChristopherDavenport/github-twin/releases/tag/v0.0.1
