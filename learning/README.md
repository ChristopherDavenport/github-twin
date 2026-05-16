# Learning github-twin

This folder is the on-ramp. github-twin is a small Retrieval-Augmented
Generation (RAG) system over your (or an org's) GitHub history. The code in
`src/github_twin/` is plumbing — these docs explain *what* the plumbing is
for, in roughly the order the data flows through it.

Each doc has the same shape: **what the concept is**, **how this codebase
implements it** (with file paths so you can jump straight to real code), and
**further reading** if you want more depth.

## Suggested reading order

1. **[What is a RAG?](00-what-is-rag.md)** — the umbrella idea. Why combine
   retrieval with a language model at all.
2. **[Ingest](01-ingest.md)** — pulling commits, PR reviews, and source
   files out of GitHub. The "where the data comes from" layer.
3. **[Process (chunking)](02-process.md)** — turning raw artifacts into
   small, embeddable pieces. The unit of retrieval.
4. **[Embed](03-embed.md)** — vector embeddings. How "semantic similarity"
   becomes math.
5. **[Store and retrieve](04-store-and-retrieve.md)** — where vectors live
   and how nearest-neighbor search works under SQL filters.
6. **[Distill](05-distill.md)** — clustering past review comments and
   asking an LLM to summarize them into reusable rules.
7. **[Eval](06-eval.md)** — measuring whether retrieval actually helps,
   versus a baseline of "just ask the model cold."
8. **[Putting it together](07-putting-it-together.md)** — how the pieces
   compose into the end-to-end MCP server an editor talks to.

A few cross-cutting concerns show up in more than one doc but don't get
their own file. They're worth knowing about up front:

- **Hybrid retrieval (BM25 + vector via RRF).** Default for every
  retrieval tool except `predict_review_outcome`. See `04-store-and-retrieve.md`.
- **Embed-time chunk prefix + LLM summaries.** Each chunk gets a
  deterministic header (path, symbol, doc) and an optional one-sentence
  LLM-written summary prepended *before embedding*, so vector queries
  can land on identifier-only code via NL queries. See `03-embed.md`
  and `02-process.md`.
- **Asymmetric BM25 query expansion.** The BM25 leg can pick up
  synonyms (`fn` ↔ `function`); the vector leg never does. See
  `04-store-and-retrieve.md`.
- **Static vs dynamic retrieval.** Most MCP tools are *dynamic* (one
  call per query). Two are *static memory blocks*:
  `house_rules()` returns all distilled rules as one Markdown paste,
  and `developer_profile()` synthesizes a 2–3-paragraph voice
  description. Call both at session start, treat the output as
  durable context. `gt init-claude-md` scaffolds a `CLAUDE.md` that
  wires this in. See `07-putting-it-together.md`.
- **`scope` parameter (`personal` / `project` / `all`).** Sugar over
  `repo=` / `author_login=` filters on retrieval tools, named after
  Claude Code's memdir tiers. See `04-store-and-retrieve.md`.
- **Eval harness has two halves.** `gt eval reviews|predictions`
  measures retrieval's effect on downstream generation; `gt eval
  search` is a tiered query-suite that scores retrieval directly. See
  `06-eval.md`.

When you've read these, the project-level docs (`README.md`, `CLAUDE.md`,
`getting_started.md`) will read as concrete configuration around concepts
you already understand.
