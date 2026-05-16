# Security Policy

## Supported versions

Only the latest released version of `github-twin` receives security updates.
Pre-1.0 means the API may shift; pin a specific version
(`uvx github-twin@0.X.Y`) if you need stability.

## Reporting a vulnerability

**Do not open a public GitHub issue for security reports.** Email
**chris@christopherdavenport.tech** with:

- A description of the issue
- Steps to reproduce
- The version of `github-twin` you're running (`github-twin --version`)
- Any logs or proof-of-concept (redact secrets first)

You can expect an acknowledgement within 7 days. I'll keep you in the
loop while a fix is prepared, and credit you in the release notes
unless you ask otherwise.

## Threat model

`github-twin` is a CLI / MCP server that handles three sensitive
surfaces:

1. **GitHub personal access tokens.** Read from `GITHUB_TOKEN` (env)
   only; never persisted to disk. Required scopes: `repo`,
   `user:email`, `read:org`. Use a fine-grained PAT and revoke it if
   it's ever exposed.
2. **LLM API keys.** `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` are read
   from env when the matching backend is selected. Like the GitHub
   token, these stay in process memory and are not logged.
3. **Local data directory.** `cfg.paths.data_dir` (default `./data`)
   contains the SQLite index, raw GitHub response cache, and (when
   org-mode persistent clones are enabled) shallow clones of every
   indexed repo. Treat it as containing every secret your indexed
   repos contain. Do not check it into source control — the default
   `.gitignore` excludes it.

### Reportable issues

- Any code path that exfiltrates `GITHUB_TOKEN` /
  `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` to a third party.
- Any way an untrusted GitHub artifact (a commit message, review
  comment, file at HEAD, PR title, …) can cause:
  - Arbitrary file write outside `cfg.paths.data_dir`.
  - Arbitrary code execution.
  - A SQL/FTS5 injection that escapes the parameterized query layer
    in `store/queries.py`.
- Any way the MCP server can be coerced into returning data outside
  the indexed corpus (prompt-injection routing aside — see "Out of
  scope" below).

### Out of scope

- **LLM prompt injection.** Adversarial review comments or commit
  messages can influence what `gt distill` / `gt summarize` generate.
  The retrieval layer surfaces those summaries verbatim. This is a
  general property of any RAG system and not a vulnerability in
  `github-twin` itself; we surface raw retrieved content to the
  caller, who is responsible for sandboxing downstream actions.
- **Local Ollama daemon.** If your `OLLAMA_HOST` points at an
  untrusted endpoint, that's your operator responsibility — not a
  bug class we triage.
- **GitHub API rate limits / cost.** The ingest paths respect the
  configured `since` cursor and back off on 429s; abuse from a
  compromised token is a GitHub-side concern.

## Build provenance

Releases published to PyPI carry SLSA build provenance
attestations via [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
Verify with `pip install` + `pip verify` (or the
[`pypi-attestations`](https://github.com/trailofbits/pypi-attestations)
tool) before installing in sensitive environments.
