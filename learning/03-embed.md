# Embed

## What it is

An **embedding** is a function that turns a piece of text into a fixed-length
list of numbers (a vector) — typically 384 or 768 or 1024 dimensions — such
that texts with similar *meaning* end up near each other in that vector
space, and texts with different meanings end up far apart.

This is the trick that makes "semantic search" work. With plain keyword
search, "user authentication" doesn't match "sign-in handler" — different
words, same idea. With embeddings, the two phrases produce vectors that
sit close together because the model learned, during training, that they
mean roughly the same thing. Searching for one finds the other.

Two distance functions show up:

- **Cosine similarity** — the angle between two vectors. Range −1 to 1
  (1 = identical direction, 0 = orthogonal, −1 = opposite). The standard
  choice for normalized embeddings.
- **L2 (Euclidean) distance** — straight-line distance between vector
  tips. Range 0 to ∞. When vectors are unit-normalized (length 1), cosine
  and L2 give equivalent rankings, so the choice is operational rather
  than semantic.

The model that produces embeddings matters. Different models have
different dimensionalities, different speed/quality tradeoffs, and
different specializations (general-purpose vs. code-specific). Swapping
models means **re-embedding the whole corpus** — vectors from different
models live in incompatible spaces and can't be compared.

The *text* you embed matters at least as much as the model. Embedding the
raw chunk body works but loses information — a 10-line function body has
no signal that the function is named `renew` or that it lives in
`Vault.scala`. github-twin embeds a **prefixed string** that splices in
those bits of context before the model ever sees the code (see
`embed/prefix.py:prefix_chunk`). Headers are deterministic
(`# {path} :: {symbol} ({node_kind})`); if `gt summarize` has run, a
one-sentence LLM summary lands on the next line; the leading docstring
(when present) lands after that; then the original body. Whenever the
header shape changes, `EMBED_TEXT_VERSION` in `pipeline.py` is bumped
and the next `gt embed` wipes and re-creates `vec_chunk` automatically.

## How github-twin does it

- **The Protocol**: `src/github_twin/embed/base.py` defines an `Embedder`
  protocol with one method: `embed(items: list[str]) -> list[list[float]]`.
  Any class that implements this is a drop-in.
- **Default backend — Ollama**: `src/github_twin/embed/ollama.py`. Calls
  a local Ollama server. The **package default model** is
  `nomic-embed-text` (768 dims) — small download (~280 MB), works
  immediately after `ollama pull nomic-embed-text`. A code-capable
  upgrade is `qwen3-embedding:8b` (4096 dims, ~4.4 GB, Qwen Coder-derived
  MoE with 86 ms/call warm latency). Switching the embedder is a
  `config.toml` edit + `DROP TABLE vec_chunk; UPDATE chunk SET
  embed_model = NULL;` because sqlite-vec bakes the dimension into the
  table definition. On this corpus the swap moved Tier-2 hybrid eval
  from 60% → 100% — see [[project-chunk-summaries]] for the
  measurement. Includes a per-item **shrink-fallback** for oversized
  inputs: if a single chunk blows past the model's context limit, it's
  truncated and retried instead of failing the whole batch.
- **Opt-in backend — sentence-transformers**:
  `src/github_twin/embed/sentence_transformers.py`. Lazy-imports
  `sentence_transformers` so the dep is only required if you install the
  `[st]` extra. Useful for `BAAI/bge-*` models, the BGE-M3 family, or any
  HuggingFace embedding model you want to run without Ollama.
- **Embed-time prefix**: `src/github_twin/embed/prefix.py:prefix_chunk(c)`
  is called from `pipeline._flush` for every chunk just before
  `embedder.embed(...)`. It builds a per-kind header (different shape for
  `code` vs `commit_message` vs `review_comment`) and prepends it to
  `chunk.text`. **Headers are not written back to `chunk.text`** — they
  exist only in the embed-time string. This keeps BM25 (which indexes
  `chunk.text` via FTS5) unchanged.
- **`EMBED_TEXT_VERSION`** in `pipeline.py` (currently `3`) is the
  migration seam. Any change to `prefix.build_header` shape requires
  bumping it. The pipeline reads the stored version from `sync_cursor`;
  on mismatch it wipes `vec_chunk` and re-embeds with the new prefix.
- **Dispatch**: `src/github_twin/embed/__init__.py:make_embedder` reads
  `cfg.embed.backend` + `cfg.embed.model` and returns the right Embedder.
- **Storage**: vectors land in `vec_chunk` (sqlite-vec virtual table,
  one row per chunk, keyed by `chunk_id`), packed as raw float32 bytes
  via `store/queries.py:_pack_vec`. The dimension is parameterized at
  table creation in `db._ensure_vec_table`, so `data/db.sqlite` is
  pinned to one embedder's dim until you drop the table.
- **Distance metric**: sqlite-vec uses **L2 (Euclidean)** by default for
  ranking; that's what `SqliteVecStore.search` returns in
  `SearchHit.distance`. The eval harness in `eval/runner.py:_cosine_distance`
  is a separate concern — it scores LLM-generated comment text against
  ground truth using cosine, which is a different question from how
  retrieval ranks chunks.

## Further reading

- **HuggingFace MTEB Leaderboard** —
  [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
  The standard ranking of embedding models across retrieval, clustering,
  and classification benchmarks. Browse it when picking a model.
- **`nomic-embed-text` on Ollama** —
  [ollama.com/library/nomic-embed-text](https://ollama.com/library/nomic-embed-text).
  The package default model card. 768-dim, general-purpose, fast.
- **`qwen3-embedding:8b` on Ollama** —
  [ollama.com/library/qwen3-embedding](https://ollama.com/library/qwen3-embedding).
  The recommended upgrade when retrieval recall matters more than
  install size. 4096-dim, MoE-based so inference is fast despite the
  parameter count.
- **Anthropic — *Contextual Retrieval*** —
  [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval).
  The technique github-twin's embed-time prefix implements: prepend a
  short context string to each chunk before embedding. Their numbers
  put it at +35% retrieval lift on its own and -49% retrieval failure
  when combined with BM25 + RRF.
- **sentence-transformers documentation** — [sbert.net](https://www.sbert.net/).
  The library github-twin's opt-in path uses; "Computing Embeddings" and
  "Semantic Textual Similarity" are the relevant chapters.
- **Cosine similarity on Wikipedia** —
  [en.wikipedia.org/wiki/Cosine_similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
  The formula and a worked example. Short and worth knowing by heart.
