-- github-twin schema. Idempotent: every statement is CREATE ... IF NOT EXISTS.
-- vec_chunk is a virtual table (sqlite-vec); its dimension is parameterized at runtime.

CREATE TABLE IF NOT EXISTS artifact (
  id           INTEGER PRIMARY KEY,
  kind         TEXT NOT NULL,        -- 'commit' | 'pr' | 'review_comment' | 'issue_comment' | 'file' | 'rule'
  external_id  TEXT,                  -- commit SHA, PR node id, comment id, etc.
  source_url   TEXT,
  repo         TEXT,
  language     TEXT,
  author_email TEXT,
  author_login TEXT,                  -- GH login; populated in org-mode (added O-A)
  created_at   TEXT,                  -- ISO8601
  decision     TEXT,                  -- 'approved' | 'changes_requested' | 'commented' | NULL
  meta_json    TEXT,
  UNIQUE(kind, external_id)
);

CREATE INDEX IF NOT EXISTS artifact_kind_lang ON artifact(kind, language);
CREATE INDEX IF NOT EXISTS artifact_decision ON artifact(decision) WHERE decision IS NOT NULL;
CREATE INDEX IF NOT EXISTS artifact_repo ON artifact(repo);
CREATE INDEX IF NOT EXISTS artifact_author ON artifact(author_login);

CREATE TABLE IF NOT EXISTS chunk (
  id           INTEGER PRIMARY KEY,
  artifact_id  INTEGER NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
  kind         TEXT NOT NULL,         -- 'code' | 'review_comment' | 'commit_message' | 'file' | 'pr_summary' | 'rule' | 'code_rule'
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

CREATE TABLE IF NOT EXISTS sync_cursor (
  resource   TEXT PRIMARY KEY,
  cursor     TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- One row, kind discriminates user-mode vs org-mode targets.
CREATE TABLE IF NOT EXISTS target (
  id            INTEGER PRIMARY KEY CHECK (id = 1),
  kind          TEXT NOT NULL,          -- 'user' | 'org' | 'repo'
  name          TEXT NOT NULL,          -- username or org login
  external_id   INTEGER NOT NULL,       -- numeric GitHub id
  emails_json   TEXT,                   -- user-mode only; NULL for org
  discovered_at TEXT NOT NULL
);

-- Per-repo state for org-mode ingest: discovered, cloned, walked.
CREATE TABLE IF NOT EXISTS repo (
  full_name               TEXT PRIMARY KEY,  -- 'owner/name'
  default_branch          TEXT,
  head_sha                TEXT,              -- last indexed default-branch SHA
  pushed_at               TEXT,              -- from /repos response; skip if <= last_files_at
  archived                INTEGER NOT NULL DEFAULT 0,
  fork                    INTEGER NOT NULL DEFAULT 0,
  size_kb                 INTEGER,
  last_files_at           TEXT,              -- last successful file walk
  last_commits_at         TEXT,              -- last successful commits cursor (wall-clock)
  last_commits_walked_sha TEXT,              -- HEAD sha at end of last commits walk (git-local path)
  last_reviews_at         TEXT               -- last successful reviews cursor
);

CREATE INDEX IF NOT EXISTS repo_pushed_at ON repo(pushed_at);

-- email → GitHub login cache. Populated lazily during git-local commits
-- ingest; misses (login IS NULL) are cached too so we don't re-query
-- unresolvable addresses.
CREATE TABLE IF NOT EXISTS email_login_map (
  email       TEXT PRIMARY KEY,             -- lowercased
  login       TEXT,                          -- NULL if no linked GH account found
  resolved_at TEXT NOT NULL,                 -- ISO8601
  source      TEXT NOT NULL                  -- 'search_commits' | 'noreply' | 'manual'
);

-- Cached output of the `developer_profile` MCP tool. One row per author
-- login; sample_hash invalidates the cache when the set of recent review
-- comments changes (e.g. after a fresh `gt sync`). The MCP tool computes
-- the current sample_hash from chunk_ids of the N most-recent reviews and
-- compares; on mismatch it re-synthesizes and overwrites this row.
CREATE TABLE IF NOT EXISTS developer_profile_cache (
  login        TEXT PRIMARY KEY,             -- author login, or '__target__' for user-mode default
  profile_md   TEXT NOT NULL,
  sample_hash  TEXT NOT NULL,                -- sha1(sorted comment chunk_ids)
  n_samples    INTEGER NOT NULL,
  generated_at TEXT NOT NULL                 -- ISO8601
);
