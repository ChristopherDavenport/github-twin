# Eval

## What it is

RAGs are tempting to ship without measurement: the retrieval looks
relevant, the generations look fluent, and you move on. But "looks
relevant" is not "actually helps." Sometimes retrieval is useless or even
harmful — the model would have produced a better answer cold than with
five distracting examples crammed into its context.

The fix is **held-out evaluation**: pick a slice of your data the system
hasn't seen during retrieval (typically: everything after a date cutoff),
ask the model the same question two ways — once cold (the baseline), once
with retrieval (the RAG) — and compare both against ground truth.

Two specific tools recur:

- **Paired comparison.** You run baseline and RAG on the *same* examples
  and look at the per-example delta. This controls for example difficulty:
  if some PRs are intrinsically harder to predict, both arms get hit
  equally and the delta isolates the retrieval effect. A **paired t-test**
  on the per-example deltas tells you whether the difference is real or
  noise.
- **Per-class F1.** For classification tasks (e.g. "approved" /
  "changes_requested" / "commented"), raw accuracy is misleading when
  classes are imbalanced — a model that always predicts the majority
  class can hit 70% accuracy and be useless. F1 per class shows you
  whether the model is actually distinguishing them.

Two specific traps to avoid:

- **Leakage.** If your retrieval can return examples from after the
  cutoff, the RAG arm gets to peek at the future. You must drop those
  hits.
- **Judge-retriever coupling.** If you measure semantic similarity to
  ground truth using the *same* embedding model that did retrieval, you're
  partly measuring how well the retriever clusters its own outputs. Use
  a different embedding model as the judge.

## How github-twin does it

- **Held-out iterators**: `src/github_twin/eval/holdout.py`.
  - `iter_held_out_review_comments` yields review-comment examples
    newer than `since`, optionally filtered by `author` and `repo`.
  - `iter_held_out_prs` yields PR examples with their decisions. In
    org mode, decisions live on `meta.reviewer_decisions` rather than
    `artifact.decision`, so the iterator extracts the per-author
    decision via `_decision_from_reviewer_meta`.
  - `count_eligible` returns the counts up-front so `gt eval` can
    pre-flight: a typo'd `--author` or empty slice fails fast instead
    of after a hundred LLM calls.
- **Runner**: `src/github_twin/eval/runner.py`.
  - `evaluate_reviews` — for each held-out comment, runs a baseline
    prompt and a RAG prompt (with retrieved similar review comments),
    embeds both outputs and the ground truth with the *judge*
    embedder, and computes cosine distance. The smaller distance wins.
  - `evaluate_predictions` — same idea for the decision classifier.
    Reports accuracy + per-class F1.
  - `_cosine_distance` and `_paired_t_one_sided` are the explicit
    math. The t-test is implemented with a normal approximation via
    `math.erf`, no SciPy dependency.
  - `_filter_post_cutoff` is the leakage guard — any retrieved hit
    whose artifact `created_at >= since` is dropped.
- **Report**: `src/github_twin/eval/report.py` renders the results as
  Rich tables: baseline mean, RAG mean, delta, paired-t verdict.
- **CLI**: `gt eval reviews --since <date>` and `gt eval predictions
  --since <date> --author <login>`. The judge embedder defaults to a
  *different* model than the retriever (sentence-transformers BGE-small)
  if the `[st]` extra is installed.

### `gt eval search` — measuring retrieval directly

The `eval reviews|predictions` harness measures *downstream effect* on
the model's output. It's expensive (one LLM call per held-out example,
two if you're comparing baseline vs RAG) and slow to iterate. When
you're tuning the retrieval pipeline itself — adding a chunker, swapping
embedders, tweaking the prefix shape — you want a faster signal that
asks "did the right chunk come back?" without involving an LLM.

`src/github_twin/eval/search_evals.py` is that harness. It loads a YAML
of queries (default at `evals/queries/default.yaml`), runs each query
through **all three retrieval modes** (bm25-only, vector-only, hybrid),
and scores each as pass/fail by checking whether any of the top-K hits
matches any of the query's `expect_any` clauses (`path_substr`,
`text_substr`, `symbol_name`, `url_substr`, ...).

Two design choices matter:

1. **Tiered targets.** Tier-1 queries must pass at 100% (CI gates on
   it); Tier-2 ≥85%; Tier-3 ≥70%. The renderer in `eval/report.py`
   prints per-tier × per-backend pass rates as a table.
2. **The gate fires on `hybrid` only.** BM25-only and vector-only are
   diagnostic columns. NL queries fundamentally can't satisfy BM25
   alone — the BM25 leg returns nothing when no query token appears in
   the chunk text — and forcing 100% BM25 would mean either rewriting
   the bank around BM25's limitations or dropping NL queries entirely.

Run with `gt eval search evals/queries/default.yaml --mode all`. The
`--expansion off|rule|ollama` flag overrides the configured BM25 query
expansion so you can A/B different expansion strategies on the same
corpus.

## Further reading

- **scikit-learn — *Model evaluation*** —
  [scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html).
  Authoritative reference for accuracy, precision, recall, F1, and why
  they differ. The "Classification metrics" section is the relevant one.
- **Paired difference test on Wikipedia** —
  [en.wikipedia.org/wiki/Paired_difference_test](https://en.wikipedia.org/wiki/Paired_difference_test).
  Short and complete. Explains why pairing tightens your statistical
  power when subjects vary.
- **Anthropic — *Evaluating prompts*** —
  [docs.anthropic.com/en/docs/build-with-claude/develop-tests](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests).
  Practical writeup of building evals for LLM-based features. Less about
  RAG specifically, more about how to think about ground truth and
  graders.
- **Ragas** — search for "Ragas evaluation library." A popular Python
  framework specifically for evaluating RAGs (faithfulness, answer
  relevancy, context precision). Worth reading their docs even if you
  don't adopt the library — the metric names are part of the working
  vocabulary.
- **Jason Liu — *Stop using LGTM as an evaluation metric*** — search for
  this exact phrase. Short opinionated post on why eyeballing retrievals
  isn't enough.
