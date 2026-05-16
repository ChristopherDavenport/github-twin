# What is a RAG?

## What it is

**Retrieval-Augmented Generation** is a pattern, not a model. You take a
language model that's good at writing fluent text but doesn't know anything
about *your* code, *your* review history, or *your* team's conventions, and
you bolt a search step onto the front of it. When the model is asked to do
something — review a diff, predict whether a PR will be accepted, write a
function — your system first searches a private corpus for the most
relevant past examples and pastes them into the prompt as context. The
model then generates an answer informed by those examples instead of
inventing one from its general training.

The reason this matters is that the model on its own has two failure modes
that retrieval addresses directly. It hallucinates when asked about
specifics it doesn't know (your codebase, your reviewers' preferences), and
it's generic when asked to imitate a style (your voice, your team's house
rules). Retrieval grounds the model in real, retrievable facts and real,
retrievable examples. The model still does the writing; the corpus does
the remembering.

github-twin is a RAG whose corpus is GitHub history. The retrieval surface
is the set of MCP tools in `src/github_twin/mcp_server/tools.py`:
`find_review_comments`, `find_style_examples`, `find_code`,
`find_applicable_rules`, `predict_review_outcome`,
`summarize_review_patterns`, `house_rules`, `developer_profile`, and
`sync`. The first six are *dynamic* — call once per query, get a list
of hits. The next two (`house_rules` + `developer_profile`) are
*static memory blocks* — call once at session start, paste the
Markdown output directly into your working context. An agent (Claude
Code, Cursor, anything that speaks MCP) wires these up and uses them
to do its actual job.

## How github-twin does it

- **The corpus**: commits + PR review comments + (in org mode) source files
  at HEAD and PR summaries. Pulled from the GitHub REST API and from
  shallow git clones.
- **The store**: SQLite + `sqlite-vec` for vectors, all in one `data/db.sqlite`
  file. One install = one corpus.
- **The retriever**: `src/github_twin/store/vector_store.py` —
  **hybrid** retrieval (BM25 keyword + k-nearest-neighbor vector,
  fused by Reciprocal Rank Fusion), with SQL pre-filters (language,
  repo, author). Vector recall is boosted by an embed-time prefix
  on every chunk: a deterministic location header plus, when
  populated, a one-sentence LLM summary written by `gt summarize`.
  `predict_review_outcome` is the one tool that bypasses BM25 and
  uses raw vector retrieval, because its inverse-distance vote
  weighting needs calibrated L2 distance.
- **The reader**: any MCP-capable agent. github-twin itself does not call a
  big language model at query time — it returns snippets and lets the agent
  decide what to do with them. The only LLM calls inside the project are
  for *distillation* (turning many review comments into a few rules) and
  *evaluation* (measuring whether the retrieved context actually helps).

The MCP tool list in `src/github_twin/mcp_server/tools.py:30` (and below)
is the cleanest entry point for "what does a user actually get."

## Further reading

- **Lewis et al., 2020 — *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks*** —
  [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401). The
  original RAG paper. Skim the intro and section 2; the architecture
  details are dated but the framing is the canonical one.
- **Anthropic — *Introducing Contextual Retrieval*** —
  [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval).
  Modern, practical writeup of how retrieval quality affects downstream
  generation, with concrete numbers.
- **Model Context Protocol** —
  [modelcontextprotocol.io](https://modelcontextprotocol.io). The protocol
  this project's server speaks. Read the "concepts" section to see how an
  agent calls tools.
- **Pinecone — *Retrieval Augmented Generation*** — search for "Pinecone
  RAG series." Accessible written-for-engineers walkthrough of the moving
  parts (embedding, indexing, retrieval) without assuming an ML
  background.
