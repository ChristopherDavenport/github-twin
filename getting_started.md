# Getting started

End-to-end walkthrough for setting up `github-twin` on a fresh machine. Reading time ~10 min, real setup time ~30 min plus a few hours of background ingest depending on how much GitHub history you have.

---

## 1. What you're building

A personal RAG over your own GitHub history (commits + PR review comments), exposed to Claude as an MCP server. Two retrieval modes:

- **Write like me** — pull your prior code as style exemplars when generating new code.
- **Review like me** — given a diff hunk, retrieve your past review comments on similar code.

Plus a distillation step that clusters review comments into reusable rules (e.g. *"Prefer Outcome over raceN/firstCompletedOf"*).

Everything stays local by default. SQLite + `sqlite-vec` on disk, Ollama for embeddings. Only the distillation step optionally calls a hosted LLM (Claude or Gemini), and even that has a local Ollama fallback. (If you have a Gemini API key but no Ollama install, you can opt in to remote Gemini embeddings at init time: `gt init --embed-backend gemini` — see [Embedder backends](#embedder-backends) in the README.)

---

## 2. Prerequisites

| Thing | Why | Install |
|---|---|---|
| **Linux/macOS/WSL** | Tested on WSL2 + Ubuntu | — |
| **Python 3.12+** | Required by `pyproject.toml` | `python3 --version` to check |
| **[Ollama](https://ollama.com)** | Local embeddings (default — skip if using the Gemini embedder) | `curl -fsSL https://ollama.com/install.sh \| sh` |
| **[`uv`](https://docs.astral.sh/uv/)** | Python project manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **~500 MB disk** | Embedding model + cached GitHub responses + DB | — |

> GitHub auth happens via the built-in `gt auth login` device flow — no `gh` CLI required. If you already use `gh`, github-twin will pick up its token automatically.

Optional (for `gt distill`):

| Thing | Why | Notes |
|---|---|---|
| **Anthropic API key** | Best-quality rule synthesis | `console.anthropic.com` → API Keys. Costs cents per run. |
| **Gemini API key** | Free-tier-friendly alternative | `aistudio.google.com` → Get API Key. 1500 req/day free on Flash. |

If you skip both, distill falls back to your local Ollama with `llama3.2`. It works but the rules will be noisier — see [Distillation quality](#7-distillation-optional) below.

---

## 3. One-time machine setup

### 3a. Authenticate to GitHub

The recommended path is the built-in OAuth device flow — no PAT to
mint, no `gh` CLI to install:

```sh
uvx github-twin auth login      # opens browser, you type a 6-char code
uvx github-twin auth status     # confirm which source is active
```

The token persists in the OS keyring (macOS Keychain / Linux Secret
Service / Windows Credential Manager) when available, otherwise a
0600 file under your data dir. Remove with `uvx github-twin auth logout`.

Required scopes (granted by default):

- `repo` — read private repos and private PR review comments. Without this, the index will silently miss anything private.
- `read:org` — list org membership for org-mode ingest.
- `user:email` — identity sweep picks up verified emails on the account. Optional but cleaner than relying on commit-history email recovery.

**Alternatives** that github-twin also accepts (in this order of preference):

- Existing `gh auth login` session — `gt` will use `gh auth token` automatically.
- `GITHUB_TOKEN` env var — useful for CI / headless / docker contexts.

### 3b. Start Ollama and pull the embedding model

```sh
# Ensure the daemon is running. On most installs it's already a system service:
ollama list

# Pull the embedding model used by github-twin.
ollama pull nomic-embed-text
```

This is a ~274 MB download. The model is locked at 2048 tokens of context; github-twin handles oversized chunks via client-side truncation + a per-item shrink fallback.

### 3c. (Optional) Set an LLM API key for distillation

Pick one. github-twin will auto-detect at `gt distill` time:

```sh
# Option A — Claude (best quality)
export ANTHROPIC_API_KEY=sk-ant-...

# Option B — Gemini (free tier covers this workload many times over)
export GEMINI_API_KEY=...
```

Append the export to your `~/.zshrc` / `~/.bashrc` so it persists across sessions. If both are set, Claude wins.

---

## 4. Install the project

```sh
git clone <your-fork-url> github-twin
cd github-twin
uv sync                # installs all deps into .venv/
```

Verify the CLI is wired:

```sh
uv run gt --help
```

You should see six commands: `init`, `ingest`, `embed`, `sync`, `distill`, `stats`, `serve`.

---

## 5. Identity discovery (`gt init`)

```sh
uv run gt init
```

This calls the GitHub API to discover:

1. Your canonical username (`gh api user`)
2. Verified emails attached to your account (`/user/emails`, needs `user:email` scope)
3. The synthetic GitHub `noreply` addresses
4. **Historical commit emails** — a search-API sweep over commits authored by your username, harvesting every distinct `commit.author.email` seen. This catches old work emails that aren't currently linked.

Expected output:

```
Identity: <your-username> (id 12345678)
Emails discovered:
  • 12345678+yourusername@users.noreply.github.com
  • current-work@example.com
  • personal@example.com
  • old-work@other-employer.com           ← surfaced by the historical sweep
  • yourusername@users.noreply.github.com

Data dir: ~/.local/share/github-twin
Config:   ~/.local/share/github-twin/config.toml
DB:       ~/.local/share/github-twin/db.sqlite
```

(Override with `GT_PATHS__DATA_DIR=...` — everything per-DB lives under
that root.)

Review the list. If anything looks wrong, edit `<data_dir>/config.toml`:

```toml
[identity]
extra_emails = ["address-the-sweep-missed@example.com"]
ignore_emails = ["bot-account@example.com"]
```

Then re-run `gt init`.

---

## 6. Ingest + embed (the long step)

### 6a. Quick smoke test first (recommended)

Before the full backfill, confirm a small slice works end-to-end:

```sh
uv run gt ingest --since 2024-01-01 --limit 5
uv run gt embed
uv run gt stats
```

You should see something like `vectors: ~10  pending embed: 0` and a few rows under each kind. If this works, the full run will work.

### 6b. Full backfill

```sh
uv run gt ingest --since 2018-01-01
uv run gt embed
```

What's happening, and what to expect:

| Phase | What it does | Time (very rough) |
|---|---|---|
| **Commit search** | One search per email; results are deduped by SHA. GitHub search API is 30 req/min, so this is rate-limit-bound. | 5–15 min |
| **Commit patch fetch** | One REST call per commit to get the diff. 5000 req/hr authenticated. | 30 min – 2 hr |
| **Review comment fetch** | Find PRs you've commented on, then per PR pull review comments, reviews, and issue comments. | 30 min – 1 hr |
| **Embed** | All chunks through Ollama `nomic-embed-text`. Typically 100–300 chunks/min on CPU, faster on GPU. | 10–40 min |

A run over a decade of activity in the typelevel/http4s/cats-effect ecosystem produced ~2700 commits, ~970 PRs, ~470 review-comment chunks, and ~8600 vectors in roughly 90 minutes. Your numbers will be smaller or larger depending on your authored volume.

To run it unattended, redirect to a log:

```sh
uv run gt ingest --since 2018-01-01 > data/backfill.log 2>&1 &
# follow along: tail -f data/backfill.log
```

Re-running is safe — artifacts are keyed by SHA / comment id; cached raw JSON under `data/raw/` avoids re-fetching anything you already have.

### 6c. Verify

```sh
uv run gt stats
```

You're looking for non-zero counts under every chunk kind and `pending embed: 0`.

---

## 7. Distillation (optional)

Clusters review-comment embeddings via HDBSCAN, then asks an LLM to synthesize a one-sentence rule per cluster. Each rule is stored as its own artifact and embedded into the same vector store, so it's queryable through the normal retrieval path *and* listable directly via `summarize_review_patterns`.

```sh
uv run gt distill
```

This will:

1. Cluster all `review_comment` chunks (no language split — clusters often discover the right grouping naturally).
2. Drop clusters smaller than 3 or larger than 40.
3. Synthesize one rule per cluster via the configured backend.
4. Store + embed each rule.

### Backend selection

`gt distill` auto-picks in this order based on what's available:

1. `ANTHROPIC_API_KEY` → Claude (`claude-sonnet-4-6` by default)
2. `GEMINI_API_KEY` / `GOOGLE_API_KEY` → Gemini (`gemini-2.5-flash`)
3. Neither → local Ollama (`llama3.2:3B`)

Force one with `gt distill --backend claude|gemini|ollama`.

### Quality expectations

- **Claude or Gemini**: rules are coherent, language-aware, and faithful to the underlying comments. ~10–25¢ on Claude with prompt caching; free under the Gemini Flash daily quota.
- **Ollama `llama3.2:3B`**: works, but expect 10–20% of rules to be artifacts — placeholder text leaking from the system prompt, invented language tags, or vague platitudes. Pull `qwen2.5-coder:14b` or similar and set `distill.ollama_model` in `<data_dir>/config.toml` for noticeably better results.

Re-running with a different backend overwrites the same rule artifacts in place (they're keyed by a hash of cluster-member IDs), so you can A/B backends without artifact churn.

### Dump rules to a markdown file

```sh
uv run python -c "
from github_twin.config import load_config
from github_twin.store import queries as q
from github_twin.store.db import open_db
cfg = load_config()
conn = open_db(cfg.paths.db_path, cfg.embed.dim)
for r in q.list_rules(conn, limit=500):
    print(f\"- [{r['language'] or '*'}] {r['rule']}\")
" > my-rules.md
```

---

## 8. Wire into Claude Code

Add the MCP server to your user-level Claude config (not the project-local `.mcp.json`):

```jsonc
// ~/.claude.json
{
  "mcpServers": {
    "github-twin": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/github-twin",
        "gt", "serve"
      ]
    }
  }
}
```

Restart Claude Code. Inside a session, `/mcp` should list `github-twin` with four tools:

| Tool | What it does |
|---|---|
| `find_review_comments(diff_hunk, language?, k=5)` | Past review comments on diffs similar to the one you paste in. |
| `find_style_examples(query, language?, k=5)` | Code you've written matching a natural-language description. |
| `summarize_review_patterns(language?, limit=20)` | Distilled rules, post-`gt distill`. |
| `sync(since?)` | Pull new commits + comments since the last cursor and embed them. |

You can also call them from outside Claude Code via any MCP-aware client.

---

## 9. Keeping it fresh

```sh
uv run gt sync
```

This is `ingest + embed` driven by the stored cursor — only deltas. Run it periodically (manually, via cron, or via the MCP `sync` tool). Distillation is *not* part of `sync`; re-run `gt distill` whenever you've accumulated meaningful new review comments and want the rules to reflect them.

---

## 10. Troubleshooting

**`gh auth token` fails / 401s on the API.** Run `gh auth status`. If the token has no `repo` scope, run `gh auth refresh -s repo,user:email`.

**`/user/emails` returns 404.** Your token doesn't have `user:email`. Harmless — the historical-commit sweep still picks up the addresses that matter. Add the scope if you want a cleaner email list.

**`the input length exceeds the context length` during embed.** Should be impossible now — there's a 4000-char client-side cap + per-item shrink fallback. If you hit it anyway, file an issue with the offending chunk's `id` and content type.

**Embed-time `ConnectError`.** Ollama isn't running. `ollama list` to check, or `ollama serve` to start it manually.

**Search API rate-limit warnings.** Normal during the first big ingest. The client respects `Retry-After` and `X-RateLimit-Reset`, so just leave it running.

**`No identity. Run \`gt init\` first.`** You skipped step 5. Run it.

**Distill rules look like nonsense ("Add Z when ...", "Proceed with the release as planned").** You're running on `llama3.2:3B`. Either set `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` and re-run, or pull a bigger Ollama model.

**Want to start over.** `rm -rf data/db.sqlite data/raw/`, then re-run from step 5.

---

## 11. Org mode (indexing a whole GitHub organization)

Everything above tracks one user. To mirror an entire org — files at HEAD plus
commits and reviews — initialize with `--kind org` instead:

```sh
uv run gt init --kind org --org typelevel
uv run gt repos          # confirm the repo list looks right
uv run gt ingest         # files-at-HEAD, then commits, then reviews per repo
uv run gt embed
```

`gt init --kind org` walks `/orgs/<name>/repos` and populates the per-repo
state used by ingest. `cfg.ingest.include_repos` / `exclude_repos`
(fnmatch on `owner/name`) let you scope to a subset.

### Clone strategy: process-and-purge vs. cached

By default the file walk **clones each repo to a tempdir, walks it, then
deletes it before moving on**. Peak disk usage is roughly the largest single
repo. That's safe on tight disks but means a re-sync re-clones every repo.

To keep clones around and update them with `git fetch --depth 1` on later
syncs, set in `<data_dir>/config.toml`:

```toml
[ingest]
cache_clones    = true
# Omit clones_dir to keep clones under <data_dir>/clones (recommended).
# Set it only when you want clones on a different disk than the DB.
# clones_dir    = "/mnt/big-disk/github-twin/clones"
max_repo_size_kb = 500000   # skip repos larger than this (default 500 MB)
```

When `cache_clones=true`, run periodic GC to drop repos that are no longer in
the org (or that you've excluded):

```sh
uv run gt clones prune --dry-run            # see what would be removed
uv run gt clones prune                       # actually remove
uv run gt clones prune --older-than-days 30  # also drop stale clones
```

### Per-author distilled rules

Org-mode review corpora include comments from every reviewer, so distilling
without a filter blurs everyone's preferences together. Scope to one
reviewer with `--author`:

```sh
uv run gt distill --author alice --backend claude
```

The MCP tools `find_review_comments` and `find_style_examples` accept
`author_login=` (and `repo=`) so a single agent session can ask, e.g.,
"how does alice review effect types in `typelevel/cats-effect`?".

## 12. What's next

- **P3 — `predict_review_outcome`**: scores a candidate PR against your historical approve/request-changes decisions on similar PRs. Not built yet; see the plan in `/home/chris/.claude/plans/using-github-history-build-robust-lantern.md`.
- **Tighten the chunker for minified bundles.** Right now there are a handful of 600KB minified-JS chunks that get truncated to 4000 chars and embedded as noise. Skipping them at chunk time would tidy the index.
- **Replace `llama3.2:3B`** for distill with a larger Ollama model if you want quality without paying.
- **Optional: sentence-transformers backend** for bulk embed throughput at org scale. The Embedder is a Protocol; adding `embed/sentence_transformers.py` is a localized change. Default stays on Ollama.
