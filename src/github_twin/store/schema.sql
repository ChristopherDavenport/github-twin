-- github-twin schema. Idempotent: every statement is CREATE ... IF NOT EXISTS.
-- vec_chunk is a virtual table (sqlite-vec); its dimension is parameterized at runtime.
--
-- Multi-target: one DB can hold N targets (user + orgs + repos). Every
-- per-target row carries `target_id`. Coalesced reads (target_id IS NULL
-- at the filter level) dedupe via (artifact.kind, artifact.external_id,
-- chunk.chunk_idx) so commits ingested under multiple targets don't
-- double-count.

CREATE TABLE IF NOT EXISTS target (
  id            INTEGER PRIMARY KEY,
  kind          TEXT NOT NULL,          -- 'user' | 'org' | 'repo'
  name          TEXT NOT NULL,          -- username, org login, or 'owner/name'
  external_id   INTEGER NOT NULL,       -- numeric GitHub id
  emails_json   TEXT,                   -- user-mode only; NULL for org/repo
  discovered_at TEXT NOT NULL,
  UNIQUE(kind, name)
);

CREATE TABLE IF NOT EXISTS artifact (
  id           INTEGER PRIMARY KEY,
  target_id    INTEGER NOT NULL REFERENCES target(id) ON DELETE CASCADE,
  kind         TEXT NOT NULL,        -- 'commit' | 'pr' | 'review_comment' | 'issue_comment' | 'file' | 'rule' | 'note'
  external_id  TEXT,                  -- commit SHA, PR node id, comment id, etc.
  source_url   TEXT,
  repo         TEXT,
  language     TEXT,
  author_email TEXT,
  author_login TEXT,                  -- GH login; populated in org-mode
  created_at   TEXT,                  -- ISO8601
  decision     TEXT,                  -- 'approved' | 'changes_requested' | 'commented' | NULL
  meta_json    TEXT,
  content_hash TEXT,                  -- sha256 of the source content (diff for commits, body for comments); NULL on legacy rows. Lets re-ingest skip chunk wipe+re-insert when the content hasn't changed.
  UNIQUE(target_id, kind, external_id)
);

CREATE INDEX IF NOT EXISTS artifact_target ON artifact(target_id);
CREATE INDEX IF NOT EXISTS artifact_kind_lang ON artifact(kind, language);
CREATE INDEX IF NOT EXISTS artifact_decision ON artifact(decision) WHERE decision IS NOT NULL;
CREATE INDEX IF NOT EXISTS artifact_repo ON artifact(repo);
CREATE INDEX IF NOT EXISTS artifact_author ON artifact(author_login);
-- Coalesce dedup leans on this composite for cheap GROUP BY / DISTINCT.
CREATE INDEX IF NOT EXISTS artifact_dedup ON artifact(kind, external_id);

CREATE TABLE IF NOT EXISTS chunk (
  id           INTEGER PRIMARY KEY,
  artifact_id  INTEGER NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  kind         TEXT NOT NULL,         -- 'code' | 'review_comment' | 'commit_message' | 'file' | 'pr_summary' | 'rule' | 'code_rule' | 'note'
  text         TEXT NOT NULL,
  context_json TEXT,
  language     TEXT,                  -- per-chunk language (file-level for code, comment's file for review)
  node_kind    TEXT,                  -- tree-sitter AST node type when AST-chunked; NULL for line-window / non-code chunks
  symbol_name  TEXT,                  -- function/class/method name extracted from the AST; NULL when unavailable
  summary      TEXT,                  -- NL summary written by `gt summarize`; NULL when not yet summarized
  embed_model  TEXT                   -- stamped when embedded; NULL means not yet embedded
);

CREATE INDEX IF NOT EXISTS chunk_artifact ON chunk(artifact_id);
CREATE INDEX IF NOT EXISTS chunk_kind ON chunk(kind);
CREATE INDEX IF NOT EXISTS chunk_kind_lang ON chunk(kind, language);
CREATE INDEX IF NOT EXISTS chunk_kind_node ON chunk(kind, node_kind) WHERE node_kind IS NOT NULL;
CREATE INDEX IF NOT EXISTS chunk_pending_embed ON chunk(id) WHERE embed_model IS NULL;
CREATE INDEX IF NOT EXISTS chunk_pending_summary ON chunk(kind, id) WHERE summary IS NULL;

-- BM25 keyword index over chunk.text. External-content so the text isn't
-- duplicated; triggers below keep it in sync. tokenchars '_' keeps snake_case
-- identifiers as one term; porter stems English for review-comment search.
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
  text,
  content='chunk',
  content_rowid='id',
  tokenize="porter unicode61 remove_diacritics 2 tokenchars '_'"
);

CREATE TRIGGER IF NOT EXISTS chunk_ai AFTER INSERT ON chunk BEGIN
  INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunk_ad AFTER DELETE ON chunk BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunk_au AFTER UPDATE ON chunk BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
  INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Per-target cursors plus a global tier (target_id=0) for cross-target
-- resources like `embed_text_version`.
CREATE TABLE IF NOT EXISTS sync_cursor (
  target_id  INTEGER NOT NULL,
  resource   TEXT NOT NULL,
  cursor     TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (target_id, resource)
);

-- Per-repo state for org-mode / repo-mode ingest: discovered, cloned, walked.
-- (target_id, full_name) is unique so two targets can hold the same repo
-- with their own walk cursors.
CREATE TABLE IF NOT EXISTS repo (
  target_id               INTEGER NOT NULL REFERENCES target(id) ON DELETE CASCADE,
  full_name               TEXT NOT NULL,  -- 'owner/name'
  default_branch          TEXT,
  head_sha                TEXT,              -- last indexed default-branch SHA
  pushed_at               TEXT,              -- from /repos response; skip if <= last_files_at
  archived                INTEGER NOT NULL DEFAULT 0,
  fork                    INTEGER NOT NULL DEFAULT 0,
  size_kb                 INTEGER,
  last_files_at           TEXT,              -- last successful file walk
  last_commits_at         TEXT,              -- last successful commits cursor (wall-clock)
  last_commits_walked_sha TEXT,              -- HEAD sha at end of last commits walk (git-local path)
  last_reviews_at         TEXT,              -- last successful reviews cursor
  PRIMARY KEY (target_id, full_name)
);

CREATE INDEX IF NOT EXISTS repo_pushed_at ON repo(pushed_at);
CREATE INDEX IF NOT EXISTS repo_full_name ON repo(full_name);

-- email → GitHub login cache. Populated lazily during git-local commits
-- ingest; misses (login IS NULL) are cached too so we don't re-query
-- unresolvable addresses. Global across targets — an email's mapping
-- doesn't change between orgs.
CREATE TABLE IF NOT EXISTS email_login_map (
  email       TEXT PRIMARY KEY,             -- lowercased
  login       TEXT,                          -- NULL if no linked GH account found
  resolved_at TEXT NOT NULL,                 -- ISO8601
  source      TEXT NOT NULL                  -- 'search_commits' | 'noreply' | 'manual'
);

-- Cached output of the `developer_profile` MCP tool. Cache key is a
-- composite string built by `_profile_cache_key` that already folds in
-- author / language / repo / target so we don't need separate columns
-- for them. The hash invalidates on any change to the underlying sample
-- set.
CREATE TABLE IF NOT EXISTS developer_profile_cache (
  login        TEXT PRIMARY KEY,             -- composite cache key (see _profile_cache_key)
  profile_md   TEXT NOT NULL,
  sample_hash  TEXT NOT NULL,                -- sha1(sorted comment chunk_ids)
  n_samples    INTEGER NOT NULL,
  generated_at TEXT NOT NULL                 -- ISO8601
);
