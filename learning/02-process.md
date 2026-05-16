# Process (chunking)

## What it is

Embedding models have a context window — usually a few thousand tokens, far
less than a full source file or a long PR description. If you embed a
whole 2,000-line file as one vector, you get a single point in vector
space that vaguely represents "this file," and a search for "how do we
parse YAML" will retrieve it or not retrieve it as a binary outcome. You
won't be able to point at the specific 30-line function that actually
parses YAML.

Chunking is how you fix that. You split the content into pieces small
enough to embed individually, large enough to be semantically meaningful
on their own, and shaped to the kind of content (a 5-line diff hunk needs
different treatment from a 1,000-line source file). Each chunk becomes
one row, one vector, one search hit.

Two design choices matter:

1. **Boundaries.** Where do you cut? Lines? Tokens? AST nodes? Function
   definitions? The cheap option — fixed-size line windows — gets you a
   working RAG quickly but emits chunks that are arbitrary segments of
   syntactically meaningful units. A class definition gets sliced
   mid-method; a long function spans three chunks whose individual
   embeddings are noise. The better option for code is **AST-aware
   chunking**: parse with tree-sitter and emit one chunk per declarable
   unit (function, class, impl block). Each chunk lines up with a thing
   a human would name and a query might describe.
2. **Overlap (line-window fallback).** When AST parsing doesn't apply —
   unknown language, parser failure, file with no chunkable nodes — you
   fall back to overlapping line windows so a function spanning a
   boundary still lives intact inside *some* window.

You also tag each chunk with metadata you might later filter on: the
language (Python? Scala?), the source path, the syntactic node kind
(`function_definition`? `class_declaration`?), the symbol name (`renew`?
`VaultSecretEq`?). Filters turn vector search from "find anything
similar" into "find anything similar *that's also a function in this
repo in this language by this author*."

There's also a separate concern that interacts with chunking: **what
text does the embedder actually see?** github-twin doesn't embed the raw
chunk text. It prepends a deterministic header (path, symbol, node kind,
leading docstring) and, when present, a one-sentence LLM summary. So the
embedder sees something like:

```
# core/.../VaultSecret.scala :: VaultSecretEq (function_definition)
# Implicit equality comparison for VaultSecret instances.

  implicit def VaultSecretEq[A : Eq] : Eq[VaultSecret[A]] = ...
```

instead of just the bare `implicit def` line. This is what lets vector
queries phrased in natural language ("Eq instance for a wrapper case
class") land on chunks whose code-only surface is just identifiers. See
`03-embed.md` for the embed-time prefix mechanism and EMBED_TEXT_VERSION
migration; `05-distill.md` for what `gt summarize` writes into the
`chunk.summary` column.

## How github-twin does it

All chunkers live in `src/github_twin/process/chunkers.py`. Each chunker
returns a typed dataclass (`CodeChunk`, `CommitMessageChunk`,
`PRSummaryChunk`, …) that the ingest writers turn into `chunk` rows.

- **`chunk_diff`** — walks a unified diff hunk by hunk. For languages
  with a tree-sitter grammar (see below), parses each hunk's post-image
  and emits **the deepest chunkable AST node that contains each
  significantly-added line** — so a method change inside a class emits
  the method, not the entire class. For languages without a grammar (or
  on parser failure), falls back to the original `_added_blocks` flow:
  contiguous runs of `+` lines bounded by `MAX_CODE_CHUNK_LINES`.
  Removed lines and context lines never reach the chunk text — the
  embedding reflects "what was introduced," not "what was around it."
- **`chunk_file`** — for org-mode file ingest. AST path emits every
  matching node in the tree (a class AND each of its methods, so a
  retrieval query can match either granularity); line-window fallback
  splits the file into ~80-line windows with a 10-line overlap.
- **AST grammar registry**: `src/github_twin/process/grammars/`. One
  file per language declares a `LanguageGrammar(chunk_node_kinds,
  symbol_name, leading_doc, ...)`. Supported today: python, scala,
  javascript, typescript (+ tsx as a separate registry entry), go,
  rust. Adding a language is one file plus an import in
  `grammars/__init__.py`. The `leading_doc` callback extracts a
  function's docstring (Python) or preceding doc-comment block (Go,
  Rust, JSDoc, ScalaDoc) — that becomes part of the embed prefix.
- **`chunk_pr_summary`** — title + body for a PR, with the body capped
  at `MAX_PR_BODY_CHARS` (2000). This is the chunk that
  `predict_review_outcome` queries against when asking "what happened to
  past PRs like this one."
- **`chunk_commit_message`** — one chunk per commit message. Small but
  useful for "have we discussed X before."
- **Per-chunk columns produced by AST chunking**: `chunk.node_kind`
  (the tree-sitter node type — `function_definition`,
  `class_declaration`, etc.) and `chunk.symbol_name` (the function /
  class name). The MCP `find_code` tool accepts a `node_kind=` filter
  that uses these directly. Line-window fallback chunks have both
  columns NULL and are silently excluded from those filtered queries.
- **`chunk.summary`** — written separately by `gt summarize` (see
  `05-distill.md`). Nullable; populated only for code-shaped kinds
  (`code`, `file`, `code_rule`, `commit_message`). Read by the
  embed-time prefix.
- **Language detection**: `process/language.py:language_for_path` is a
  pure extension → language map (`.py` → `python`, `.scala` → `scala`,
  etc.) backed by Pygments lexer aliases. The result lands on the
  `chunk.language` column, never on the artifact. This is why a single
  PR artifact can contribute chunks in multiple languages, and why
  queries filter `WHERE chunk.language = ?` rather than artifact-level.
- **Exclusions**: `process/chunkers.is_excluded_path` handles "don't
  index this" cases — generated files, vendored dependencies,
  lockfiles. Configured via `cfg.ingest.exclude_paths`.

## Further reading

- **LangChain — *Text Splitters* guide** — search for "langchain text
  splitters." Even if you never use LangChain, their splitter taxonomy
  (recursive character, token, language-aware, semantic) is a tidy
  vocabulary for chunking choices.
- **Greg Kamradt — *5 Levels of Text Splitting*** — search for "5 levels
  text splitting Kamradt." A practical ladder from "split by length" to
  "split by meaning," with worked examples.
- **Tree-sitter** — [tree-sitter.github.io](https://tree-sitter.github.io/tree-sitter/).
  Worth reading even if you never touch the API directly — github-twin
  uses it through the `tree-sitter-language-pack` wheel, but the
  concept of incremental parsing + named/anonymous nodes + field-keyed
  children shows up everywhere in our grammar files.
- **OpenAI — *Embedding long inputs*** — search for "OpenAI embeddings
  long inputs." Frames the size-vs-overlap tradeoff clearly.
