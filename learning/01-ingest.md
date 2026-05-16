# Ingest

## What it is

Ingest is the boring-but-load-bearing part of any RAG: getting the raw data
out of its source system and into something you control. For github-twin,
the source is GitHub — both the REST API (commits, PRs, reviews) and git
itself (cloning a repo so you can walk every source file).

Three concerns dominate ingest design:

1. **Pagination and rate limits.** GitHub gives you 100 items per page and
   roughly 5000 API requests per hour. You can't fetch a 50,000-commit
   history in one call; you fetch a page at a time, with retries on 429s
   and 5xxs.
2. **Idempotency.** You will re-run ingest. New PRs land, old ones get
   updated, you fix a bug in your chunker. The system needs a way to say
   "this exact artifact exists already, just update it" instead of writing
   duplicates. The standard trick is a **stable external id** (e.g. a
   commit SHA, a PR number) used as the dedupe key.
3. **Incremental sync.** A backfill from 2018 is expensive; the daily
   delta is cheap. The system records a **cursor** ("we last saw reviews
   updated through 2026-05-01") and on the next run only fetches things
   newer than that.

For source files specifically there's a fourth concern: the REST API
doesn't expose every file's content efficiently. So github-twin does a
shallow `git clone` per repo, walks the working tree, and (by default)
deletes the clone before moving on. This caps peak disk at one repo at a
time.

## How github-twin does it

- **HTTP wrapper**: `src/github_twin/ingest/github_client.py` is a thin
  httpx layer that handles pagination, rate-limit headers, and 5xx retries.
  Anywhere you see `gh.paginate(url, params=...)`, that's this.
- **Commits**: `src/github_twin/ingest/commits.py` has two entry points —
  `ingest_commits` for user mode (search by author email) and
  `ingest_commits_org` for org mode (walk every repo). Each commit
  becomes one `artifact` row keyed by SHA. Commits ingest defaults to
  a **local git walker**: rather than paginating
  `/repos/{r}/commits?since=...` and fetching each patch with a second
  API call, we shallow-clone deep enough, run `git log --format=…
  --no-merges`, and `git show` the diff locally. Faster, much cheaper
  on rate limits, and lets us see commit metadata (author email,
  parent SHAs) the API doesn't always surface. Toggle off with
  `cfg.ingest.use_local_git_for_commits = false`.
- **Reviews and PRs**: `src/github_twin/ingest/reviews.py`. User mode
  searches by `commenter:<me>`; org mode walks
  `/repos/{r}/pulls?state=all&sort=updated&direction=desc` per repo and
  stops at the cursor. PRs become `artifact` rows; review comments become
  separate `artifact` rows linked back to the PR.
- **Files (org mode only)**: `src/github_twin/ingest/files.py` lists the
  repo table, skips anything whose `pushed_at` hasn't changed since last
  walk, then for each survivor uses the `cloned_repo` context manager from
  `src/github_twin/ingest/clone.py` to clone, walk, chunk, and clean up.
- **Idempotency seam**: `store/queries.py:upsert_artifact` plus
  `delete_chunks_for_artifact`. Every ingest writer follows the same
  pattern: upsert the artifact by `(kind, external_id)`, delete any
  existing chunks under that artifact, then insert fresh chunks. Safe to
  re-run, safe to interrupt.
- **Cursors**: `store/queries.set_cursor` / `get_cursor` for global
  resources, `store/queries.set_repo_cursor` for per-repo state
  (`last_commits_at`, `last_reviews_at`, `last_files_at`, `head_sha`).
- **Token hygiene**: `clone.py` builds the clone URL with the token
  inline (`https://oauth2:<token>@github.com/owner/repo`), then immediately
  runs `git remote set-url origin` to scrub it from `.git/config` before
  doing any other work.

## Further reading

- **GitHub REST API overview** —
  [docs.github.com/en/rest](https://docs.github.com/en/rest). Skim the
  pagination and conditional-requests pages; both inform how `ingest`
  decides what to fetch.
- **GitHub rate-limit reference** —
  [docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api).
  Explains the 5000/hr ceiling, the secondary rate limits, and the
  response headers (`X-RateLimit-Remaining` etc.) the client watches.
- **Git "shallow clone"** — `git help clone`, specifically `--depth=1`.
  Why this is the right tool when you only want HEAD and not the full
  history.
- **idempotency in data pipelines** — search for "idempotent ingest stable
  external id." The pattern (upsert by natural key, replace dependents)
  predates RAGs and is the foundation of any re-runnable extractor.
