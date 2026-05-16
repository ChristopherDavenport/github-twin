# Distill

## What it is

Retrieval finds individual examples. Distillation summarizes a *pattern*
across many examples into something compact you can reuse.

Concretely: if you've left 4,000 review comments over five years, many of
them say variants of the same thing — "use the typed wrapper," "this needs
a test," "prefer `Async[F]` over `IO`." A query that returns five random
hits from those 4,000 is useful but noisy. A short "house rules" document
that says *"prefer the typed wrapper for HTTP responses (you said this
roughly 80 times over 3 years)"* is more useful, and more compact, than
any individual example.

The standard recipe is two-step:

1. **Cluster** the embeddings so semantically similar comments group
   together. HDBSCAN is a common pick because it doesn't require you to
   guess the number of clusters and it labels outliers as noise rather
   than forcing them into a cluster.
2. **Summarize** each cluster with a language model: feed it the
   representative members, ask it to articulate the rule they collectively
   express. Repeat for every cluster.

The output is a set of short rules, each linked back to the comments that
generated it (so you can audit the rule by re-reading its evidence).

## How github-twin does it

- **Clustering**: `src/github_twin/distill/cluster.py`. Two entry
  points: `cluster_review_comments` and `cluster_code_chunks`. Both
  load chunks with their embeddings (optionally filtered by
  `author_login`, `repo`, `language` for code), run HDBSCAN over the
  vectors, and return a list of `Cluster` objects. Outliers (HDBSCAN
  label `-1`) are dropped.
- **Synthesizer Protocol**: `src/github_twin/distill/synth.py` defines a
  `RuleSynthesizer` Protocol with one method: turn a cluster of members
  into a rule. Three implementations: `ClaudeSynthesizer`,
  `GeminiSynthesizer`, `OllamaSynthesizer`. Auto-pick precedence
  (Claude → Gemini → Ollama) based on which API key is set.
- **Orchestration**: `src/github_twin/distill/rules.py:distill_rules`.
  `chunk_kind="review_comment"` (default) clusters review comments and
  emits **`chunk.kind='rule'`** chunks; `chunk_kind="code"` clusters
  code chunks and emits **`chunk.kind='code_rule'`** chunks. The
  resulting rule chunk is embedded the same way every other chunk is,
  so retrieval is identical. `find_applicable_rules` is the MCP tool
  that surfaces `code_rule` chunks; `summarize_review_patterns` lists
  `rule` chunks. The artifact's `external_id` is a hash of the member
  chunk ids, so re-running distill with the same inputs overwrites in
  place — safe across backends.
- **Per-author scoping** (org mode): `gt distill --author <login>`
  filters the clustering step. Otherwise, an org with many active
  reviewers will blend everyone's style into mushy averages.

### `gt summarize` is the other LLM-uses-the-corpus path

Different goal: instead of clustering many chunks into one rule, write
**one short summary per chunk** to help retrieval find that chunk.
Implemented in `src/github_twin/process/summarize.py`. The summary
lands in `chunk.summary` (nullable TEXT column) and the embed-time
prefix in `embed/prefix.py` splices it into the header *before
embedding*. This is what gives natural-language queries ("Eq instance
for a wrapper case class") a shot at landing on chunks whose
code-only surface is just identifiers (`VaultSecretEq`).

Same LLM dispatch (Claude → Gemini → Ollama). Idempotent; safe to run
repeatedly. `gt sync` includes a summarize pass by default; pass
`--skip-summarize` to skip. After summarize fills `chunk.summary`, the
**next `gt embed` will detect the EMBED_TEXT_VERSION bump and re-embed
the whole corpus** so the vector index actually sees the summaries.

## Further reading

- **HDBSCAN documentation** —
  [hdbscan.readthedocs.io](https://hdbscan.readthedocs.io/). Start with
  "How HDBSCAN Works" — the visual explanations of mutual reachability
  distance and the condensed cluster tree are worth the time even if you
  never tune the parameters.
- **scikit-learn — *Clustering*** —
  [scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html).
  The clustering-algorithm chooser table at the top is a useful map of the
  space. (HDBSCAN's contrib package is in `hdbscan`, not sklearn proper,
  but the conceptual context is here.)
- **Anthropic — *Prompt caching*** —
  [docs.anthropic.com/en/docs/build-with-claude/prompt-caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).
  Why the Claude synthesizer is the default when an API key is available:
  prompt caching makes cluster-by-cluster synthesis cheap because the
  system prompt is reused.
- **HuggingFace — *Topic Modeling with BERTopic*** — search for
  "BERTopic." It's another implementation of the embed → cluster → label
  pipeline, packaged as a single library. Reading their docs is a useful
  way to see the pattern in a different shape.
