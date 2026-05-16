# Store and retrieve

## What it is

Once you have a million vectors, you need two things from your storage
layer: a way to **find the nearest neighbors** of a query vector quickly,
and a way to **filter** that search by metadata (only Python chunks, only
this repo, only this author).

The brute-force approach — compute the distance from the query to every
stored vector, sort, take the top k — is exact and dead simple. It scales
worse than "approximate nearest neighbor" (ANN) algorithms like HNSW or
IVF, but it's correct, and at the scale a single developer or even a
mid-sized org generates, it's fast enough. Sub-second up to about
500,000 vectors; a couple of seconds at a few million.

The metadata-filter dimension is where naive vector stores fall over. If
you do KNN first and *then* filter, you might get k=5 hits in your search
window but only 1 survives the language filter — the other 4 nearest hits
were in the wrong language and you missed them. The fix is **SQL
pre-filtering with overscan**: filter first to get the set of *eligible*
chunk ids, then do KNN within that set. You ask the vector index for
more than k hits (overscan) so that after intersecting with the eligible
set you still have enough.

## How github-twin does it

- **Schema**: `src/github_twin/store/schema.sql`. The relevant tables:
  - `artifact` — one row per ingested thing (commit, PR, review comment,
    file). Has metadata: `kind`, `author_login`, `repo`, `source_url`,
    `created_at`, free-form `meta` JSON.
  - `chunk` — one row per chunk extracted from an artifact. Has its own
    `kind` (`code`, `review_comment`, `commit_message`, `file`,
    `pr_summary`, `rule`, `code_rule`), per-chunk `language`.
  - `vec_chunk` — the sqlite-vec virtual table that holds packed vectors,
    keyed by `chunk_id`. Created at runtime in `db._ensure_vec_table` so
    the embedding dimension can be parameterized.
  - `chunk_fts` — the FTS5 virtual table for BM25 keyword search over
    `chunk.text`. See the "Hybrid retrieval" section below.
- **The retrieval call**: `store/queries.py:vector_search` is the
  canonical vector primitive. It builds a `chunk_id IN (subquery)`
  pre-filter from the optional `chunk_kind`, `language`, `repo`, and
  `author_login` arguments, then runs the KNN inside that set.
- **The Protocol seam**: `src/github_twin/store/vector_store.py` defines a
  `VectorStore` Protocol with a single `search(query_vec, *, filters, k)`
  method. Two implementations:
  - **`SqliteVecStore`** (default) — wraps `q.vector_search`. The
    `sqlite-vec` extension does the actual brute-force KNN inside SQLite,
    so the whole search is one SQL query.
  - **`FaissVectorStore`** (opt-in via the `[faiss]` extra) — loads all
    vectors into a Faiss `IndexIDMap2(IndexFlatL2)` at startup, runs KNN
    with overscan, then intersects with a SQL-side eligibility query for
    the metadata filter.
- **Dispatch**: `make_vector_store(conn, backend=..., dim=...)` at the
  bottom of `vector_store.py`. The MCP tools all call `store.search(...)`
  — they don't know which backend is underneath.

## Hybrid retrieval: BM25 + RRF

Pure vector search has a blind spot. Embeddings are great at "what does
this *mean*?" but mediocre at "does this contain *this exact identifier*?"
A query for `getUserById` returns chunks about user-fetching code in
general; the chunk that literally contains the function `getUserById` may
not even crack the top 10 because semantically it looks like every other
"fetch from a database" snippet in the corpus.

The fix is a second retrieval leg that scores on lexical match instead of
semantic similarity. BM25 is the standard: it ranks documents by how often
the query's terms appear, weighted by how rare those terms are corpus-wide
and normalized by document length. Identifiers like `SQLITE_OPEN_READWRITE`
or `deduplicate_results` are rare globally and score very high on the one
chunk that contains them.

In SQLite, BM25 lives in the FTS5 extension. github-twin sets up an
**external-content** FTS5 table that indexes `chunk.text` without
duplicating the storage:

```sql
CREATE VIRTUAL TABLE chunk_fts USING fts5(
  text,
  content='chunk',
  content_rowid='id',
  tokenize="porter unicode61 remove_diacritics 2 tokenchars '_'"
);
```

Three tokenizer choices matter:

- **`porter`** — English stemmer, so a comment about "tests" matches a
  query about "testing".
- **`unicode61`** — Unicode-aware word splitter (better than the default
  `ascii` for any non-ASCII text in commit messages or comments).
- **`tokenchars '_'`** — treats underscore as part of a word, so
  `user_session_id` stays one token instead of being split into three
  generic words.

Three triggers (`chunk_ai`, `chunk_au`, `chunk_ad` in `schema.sql`) keep
`chunk_fts` synchronized with the `chunk` table on insert/update/delete —
no separate write path to maintain.

### Reciprocal Rank Fusion

Now you have two ranked lists for any query: one from vector similarity,
one from BM25. How do you merge them? The two scores aren't comparable —
L2 distance and BM25 score live in different universes. The trick is to
throw away the scores entirely and just use the **ranks**:

```
fused_score(chunk) = 1/(k + rank_vec) + 1/(k + rank_bm25)
```

where `k=60` is the standard constant from the original RRF paper. A chunk
that ranks first in both lists gets `2/(60+1) ≈ 0.0328`. A chunk that
ranks first in one list and doesn't appear in the other gets
`1/(60+1) ≈ 0.0164`. Sort by fused score, return top-k.

Two things to notice about this formula:

1. **It's symmetric across retrievers.** No weights to tune. If you trust
   both signals equally (which is roughly true), you don't need to pick a
   λ.
2. **The constant `k=60` damps the contribution of low-rank hits.** The
   gap between rank 1 and rank 2 is large; the gap between rank 50 and
   rank 51 is tiny. Without this damping, a chunk that's mediocre in both
   lists could outrank a chunk that's #1 in one list.

To get enough overlap for the fusion to do real work, you **over-fetch**:
github-twin asks each retriever for 50 hits even when the final `k=5`. If
you only fetched 5 from each, there'd often be zero overlap and RRF would
degenerate to "union the two top-5s."

`hybrid_search` in `vector_store.py` does this work in ~15 lines of Python
that the MCP tools call. The `SearchHit.distance` field is reused to carry
`1 - rrf_score` so display code that just renders the distance keeps
working — but tools that interpret the distance numerically (currently
only `predict_review_outcome`) bypass `hybrid_search` and stay on raw
vector retrieval.

The technique comes from Anthropic's
[Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
post, which reports a ~49% reduction in retrieval failure from adding
BM25 + RRF to a vector-only baseline. That post also describes a
"contextual embedding" technique (prepending an LLM-generated chunk
description before embedding) — github-twin implements that too,
via the embed-time prefix mechanism in `embed/prefix.py:prefix_chunk`
(see `03-embed.md`). The deterministic part of the prefix (path,
symbol, node kind, leading docstring) lands automatically at chunk
creation time; the LLM-generated summary part is written by `gt
summarize` (see `05-distill.md`).

### Asymmetric BM25 query expansion

Modern embedding models already capture synonyms — a query for
"function" lands near `def f()` and `fn f()` and `func f()` in vector
space without any help. **BM25 does not**, because it scores on exact
token matches. A user typing "function" gets zero BM25 signal from a
Go file that spells it `func`.

The fix is to expand queries on the BM25 side only. github-twin's
`store/query_expansion.py` defines a `QueryExpander` protocol:

```python
def expand(query_text: str) -> list[list[str]]:
    # one OR-group per query token, original first
```

…with two backends:

- **`RuleExpander`** — a deterministic table of code-shaped synonyms
  (function ↔ func ↔ fn, search ↔ find ↔ lookup) plus case-variant /
  camelCase / snake_case splits. Always safe; zero deps. Default.
- **`OllamaExpander`** — wraps a local LLM call ("expand these
  tokens"), caches per-token in `data/query_expansion_cache.sqlite`.
  Opt-in via `cfg.retrieval.query_expansion = "ollama"`.

`hybrid_search(..., expander=...)` threads the expander to the BM25 leg
*only*. The vector leg receives `query_vec` unchanged. amanmcp's research
measured a ~15pp dense regression when both legs were expanded; a
regression test (`test_hybrid_passes_expander_only_to_bm25_leg`) pins
the asymmetry.

## Further reading

- **`sqlite-vec` README** —
  [github.com/asg017/sqlite-vec](https://github.com/asg017/sqlite-vec).
  The extension that gives SQLite vector-search primitives. Worth
  reading end-to-end; it's short.
- **Faiss wiki** —
  [github.com/facebookresearch/faiss/wiki](https://github.com/facebookresearch/faiss/wiki).
  Especially the "Faiss indexes" page, which explains why `IndexFlatL2`
  (exact) and `IndexHNSWFlat` (approximate) exist and when to use each.
- **Pinecone — *Hierarchical Navigable Small Worlds (HNSW)*** — search
  for "Pinecone HNSW." The clearest accessible writeup of the ANN
  algorithm Faiss and most modern vector DBs use under the hood.
- **Postgres pgvector** —
  [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
  Same idea as `sqlite-vec` but for Postgres. Useful comparison for
  understanding what "vector search inside a relational DB" looks like
  in production setups.
- **SQLite FTS5 docs** —
  [sqlite.org/fts5.html](https://www.sqlite.org/fts5.html). The
  external-content section explains why we use `INSERT INTO chunk_fts
  VALUES('rebuild')` to backfill and why `SELECT 1 FROM chunk_fts` lies
  about index state when the source table is populated.
- **Cormack, Clarke, Büttcher (2009) — *Reciprocal Rank Fusion outperforms
  Condorcet and individual rank learning methods*** — the original RRF
  paper. Short. The empirical finding that `k=60` works across many
  retrievers without per-domain tuning is the practical takeaway.
- **Anthropic — *Contextual Retrieval*** —
  [anthropic.com/engineering/contextual-retrieval](https://www.anthropic.com/engineering/contextual-retrieval).
  The full recipe github-twin's retrieval pipeline implements:
  BM25 + vector with RRF fusion, plus per-chunk contextual prefixes
  embedded into the vector. Worth reading for the empirical numbers
  and as the reference for why the asymmetric query-expansion
  contract exists.
